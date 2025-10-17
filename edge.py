SVC="12345678-1234-5678-1234-56789abcdef0"
CHR="12345678-1234-5678-1234-56789abcdef1"
NAME="Nano33-FallServer"

MONGO_URL="mongodb+srv://steven:2121@iab330steven.t74aifz.mongodb.net/?retryWrites=true&w=majority&appName=IAB330Steven"
MONGO_DB="HARData"
MONGO_COLL="Data"
CSV_PATH="./data.csv"
MODEL_PATH="./har.joblib"

import os, sys, time, math, struct, asyncio, csv
from collections import deque
import numpy as np, pandas as pd
from bleak import BleakClient, BleakScanner
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
from pymongo import MongoClient
import certifi

SAMPLE_RATE_HZ=50
WINDOW_SAMPLES=2*SAMPLE_RATE_HZ
ACC_IMPACT_MG=2800
STILLNESS_STD_MG=80
IMPACT_WINDOW_S=1.5

async def find_device():
    devs=await BleakScanner.discover(timeout=5.0)
    for d in devs:
        n=(d.name or "")
        if NAME.lower() in n.lower():
            print("Found", n or "(no name)", d.address)
            return d.address

def basic_stats(v):
    return [float(np.mean(v)), float(np.std(v)), float(np.min(v)), float(np.max(v)), float(np.max(v)-np.min(v))]

def feature_vector(arr6xN):
    ax,ay,az,gx,gy,gz=[arr6xN[:,i] for i in range(6)]
    feats=[]
    for v in (ax,ay,az,gx,gy,gz): feats+=basic_stats(v)
    amag=np.sqrt(ax*ax+ay*ay+az*az)
    feats+=[float(np.mean(amag)), float(np.std(amag)), float(np.mean(np.abs(ax)+np.abs(ay)+np.abs(az)))]
    return np.array(feats,dtype=float)

async def collect_mode(label, csv_path=CSV_PATH):
    newfile=not os.path.exists(csv_path)
    f=open(csv_path,"a",newline=""); w=csv.writer(f)
    if newfile: w.writerow(["ts","ax","ay","az","gx","gy","gz","label"])
    addr=await find_device()
    c=BleakClient(addr, timeout=15)
    await c.connect()
    print("Connected", c.is_connected)
    print(f"Collecting '{label}' → {csv_path}")
    async def on_notify(_, data):
        ts,ax,ay,az,gx,gy,gz=struct.unpack("<I6h", data)
        w.writerow([ts,ax,ay,az,gx,gy,gz,label])
    await c.start_notify(CHR, on_notify)
    while True: await asyncio.sleep(1)

def train_mode(csv_path=CSV_PATH, model_path=MODEL_PATH):
    df=pd.read_csv(csv_path)
    X=[]; y=[]
    data=df[["ax","ay","az","gx","gy","gz","label"]].to_numpy()
    N=len(data)
    for s in range(0, N-WINDOW_SAMPLES+1, WINDOW_SAMPLES):
        seg=data[s:s+WINDOW_SAMPLES]
        labels=seg[:,-1]
        vals,cnts=np.unique(labels,return_counts=True)
        lbl=str(vals[int(np.argmax(cnts))])
        arr=seg[:,:6].astype(float)
        X.append(feature_vector(arr)); y.append(lbl)
    X=np.vstack(X); y=np.array(y)
    Xtr,Xte,Ytr,Yte=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
    clf=DecisionTreeClassifier(max_depth=12,class_weight="balanced",random_state=42)
    clf.fit(Xtr,Ytr)
    yp=clf.predict(Xte)
    print("ACC:", accuracy_score(Yte,yp))
    print(classification_report(Yte,yp,zero_division=0))
    dump(clf, model_path); print("Saved", model_path)

async def infer_mode(model_path=MODEL_PATH):
    clf=load(model_path)
    print("Model:", model_path, "| classes:", list(clf.classes_))
    addr=await find_device()
    c=BleakClient(addr, timeout=15)
    await c.connect()
    print("Connected", c.is_connected)
    win=deque(maxlen=WINDOW_SAMPLES)
    fallw=deque(maxlen=int(1.0*SAMPLE_RATE_HZ))
    last_imp=0.0
    since=0
    async def on_notify(_, data):
        nonlocal last_imp, since
        ts,ax,ay,az,gx,gy,gz=struct.unpack("<I6h", data)
        win.append([ax,ay,az,gx,gy,gz]); fallw.append([ax,ay,az]); since+=1
        m=math.sqrt(ax*ax+ay*ay+az*az); now=time.time()
        if m>ACC_IMPACT_MG: last_imp=now
        if last_imp and (now-last_imp)<IMPACT_WINDOW_S and len(fallw)>int(0.8*SAMPLE_RATE_HZ):
            mags=[math.sqrt(a*a+b*b+c*c) for (a,b,c) in list(fallw)[-int(0.8*SAMPLE_RATE_HZ):]]
            if float(np.std(mags))<STILLNESS_STD_MG:
                print(time.strftime("%H:%M:%S"), "| FALL"); last_imp=0.0
        if since>=WINDOW_SAMPLES and len(win)==WINDOW_SAMPLES:
            since=0
            arr=np.array(list(win),dtype=float)
            feats=feature_vector(arr)
            lab=str(clf.predict([feats])[0])
            print(time.strftime("%H:%M:%S"), "|", lab)
    await c.start_notify(CHR, on_notify)
    while True: await asyncio.sleep(1)

def mongo_db():
    return MongoClient(
        MONGO_URL,
        tls=True,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=30000,
        socketTimeoutMS=30000,
    )[MONGO_DB]

def cloud_upload(csv_path=CSV_PATH):
    db=mongo_db(); coll=db[MONGO_COLL]
    df=pd.read_csv(csv_path, usecols=["ts","ax","ay","az","gx","gy","gz","label"])
    df["ts"]=df["ts"].astype(int)
    for col in ["ax","ay","az","gx","gy","gz"]: df[col]=df[col].astype(int)
    df["label"]=df["label"].astype(str)
    rec=df.to_dict("records")
    step=1000
    for i in range(0,len(rec),step): coll.insert_many(rec[i:i+step])
    print("Uploaded", len(rec), "rows to", f"{MONGO_DB}.{MONGO_COLL}")

def cloud_update(csv_path=CSV_PATH):
    db=mongo_db(); coll=db[MONGO_COLL]
    r=coll.delete_many({})
    print("Cleared cloud rows:", r.deleted_count)
    cloud_upload(csv_path)

def cloud_retrieve(csv_path=CSV_PATH):
    db=mongo_db(); coll=db[MONGO_COLL]
    cur=coll.find({}, {"_id":0,"ts":1,"ax":1,"ay":1,"az":1,"gx":1,"gy":1,"gz":1,"label":1}).sort("ts",1)
    rows=list(cur)
    with open(csv_path,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["ts","ax","ay","az","gx","gy","gz","label"])
        for d in rows:
            w.writerow([int(d.get("ts",0)),int(d.get("ax",0)),int(d.get("ay",0)),int(d.get("az",0)),
                        int(d.get("gx",0)),int(d.get("gy",0)),int(d.get("gz",0)),str(d.get("label",""))])
    print("Downloaded", len(rows), "rows to", csv_path)

def cloud_delete():
    db=mongo_db(); coll=db[MONGO_COLL]
    r=coll.delete_many({})
    print("Deleted", r.deleted_count, "rows from", f"{MONGO_DB}.{MONGO_COLL}")

def menu():
    global MONGO_URL, MONGO_DB, MONGO_COLL, CSV_PATH, MODEL_PATH
    while True:
        print("\n1) collect  2) train  3) infer  4) upload  5) retrieve  6) update  7) delete  9) config  0) quit")
        s=input("> ").strip()
        if s=="1":
            lbl=input("label [walking/lying/fall/...]: ").strip()
            p=input(f"csv path [{CSV_PATH}]: ").strip() or CSV_PATH
            asyncio.run(collect_mode(lbl,p))
        elif s=="2":
            p=input(f"csv path [{CSV_PATH}]: ").strip() or CSV_PATH
            m=input(f"model path [{MODEL_PATH}]: ").strip() or MODEL_PATH
            train_mode(p,m)
        elif s=="3":
            m=input(f"model path [{MODEL_PATH}]: ").strip() or MODEL_PATH
            asyncio.run(infer_mode(m))
        elif s=="4":
            p=input(f"csv path [{CSV_PATH}]: ").strip() or CSV_PATH
            cloud_upload(p)
        elif s=="5":
            p=input(f"csv path to overwrite [{CSV_PATH}]: ").strip() or CSV_PATH
            cloud_retrieve(p)
        elif s=="6":
            p=input(f"csv path [{CSV_PATH}]: ").strip() or CSV_PATH
            cloud_update(p)
        elif s=="7":
            cloud_delete()
        elif s=="9":
            print("Current:", {"url":MONGO_URL,"db":MONGO_DB,"coll":MONGO_COLL,"csv":CSV_PATH,"model":MODEL_PATH})
            MONGO_URL=input("Mongo URL (SRV): ").strip() or MONGO_URL
            MONGO_DB=input(f"DB name [{MONGO_DB}]: ").strip() or MONGO_DB
            MONGO_COLL=input(f"Collection [{MONGO_COLL}]: ").strip() or MONGO_COLL
            CSV_PATH=input(f"Local CSV [{CSV_PATH}]: ").strip() or CSV_PATH
            MODEL_PATH=input(f"Model path [{MODEL_PATH}]: ").strip() or MODEL_PATH
        elif s=="0":
            sys.exit(0)

if __name__=="__main__":
    menu()