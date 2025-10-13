# edge_ml.py — one-file 2s HAR: collect/train/infer + heuristic & instant FALL
# BLE service must match your Arduino sketch:
SVC="12345678-1234-5678-1234-56789abcdef0"
CHR="12345678-1234-5678-1234-56789abcdef1"
NAME="Nano33-FallServer"

import argparse, asyncio, csv, os, struct, sys, time, math
from collections import deque, Counter
from datetime import datetime, timezone

# ---------- BLE (requires: pip install bleak pyobjc) ----------
from bleak import BleakClient, BleakScanner

# ---------- Common constants ----------
FS=50               # Hz from Arduino
WIN=2*FS            # 2s window
IMPACT=2800         # mg spike for fall
STILL_STD=80        # mg std after impact => lying still
IMPACT_WIN=1.5      # s window to check stillness post-impact

# ---------- Helpers ----------
async def find_addr():
    try: devs=await BleakScanner.discover(timeout=6.0, service_uuids=[SVC])
    except TypeError: devs=await BleakScanner.discover(timeout=6.0)
    for d in devs:
        uu=(getattr(d,"metadata",{}) or {}).get("uuids",[])
        if SVC.lower() in [u.lower() for u in uu] or (d.name or "").lower().find(NAME.lower())!=-1:
            print("Found", d.name or "(no name)", d.address); return d.address
    raise RuntimeError("Device not found — reset Nano & retry.")

def acc_mag(ax,ay,az): return math.sqrt(ax*ax+ay*ay+az*az)
def std(vals):
    if not vals: return 0.0
    m=sum(vals)/len(vals); return math.sqrt(sum((v-m)*(v-m) for v in vals)/len(vals))
def gyro_rms(gx,gy,gz):
    n=len(gx) or 1
    s=sum(gx[i]*gx[i]+gy[i]*gy[i]+gz[i]*gz[i] for i in range(n))
    return math.sqrt(s/n)

# ---------- Mode: heuristic (no ML) ----------
async def mode_heuristic():
    buf=deque(maxlen=WIN); fallbuf=deque(maxlen=int(1.0*FS))
    last_imp=0.0; last_print=0.0
    addr=await find_addr()
    async with BleakClient(addr, timeout=15) as c:
        await c.connect(); print("Connected?", c.is_connected)
        print("Heuristic mode: label every 2s; FALL is instant.")

        async def on_notify(_, data: bytearray):
            nonlocal last_imp, last_print
            if len(data)!=16: return
            ts,ax,ay,az,gx,gy,gz=struct.unpack("<I6h", data)
            buf.append([ax,ay,az,gx,gy,gz]); fallbuf.append([ax,ay,az])
            m=acc_mag(ax,ay,az); now=time.time()
            if m>IMPACT: last_imp=now
            # Instant FALL: impact then ~1s stillness
            if last_imp and (now-last_imp)<IMPACT_WIN and len(fallbuf)>int(0.8*FS):
                ms=[acc_mag(*s) for s in list(fallbuf)[-int(0.8*FS):]]
                if std(ms)<STILL_STD:
                    print(datetime.now(timezone.utc).isoformat(),"| FALL        conf=0.95"); last_imp=0.0
            # Print every 2s once buffer filled
            if len(buf)==WIN and (now-last_print)>=2.0:
                last_print=now
                axs=[b[0] for b in buf]; ays=[b[1] for b in buf]; azs=[b[2] for b in buf]
                gxs=[b[3] for b in buf]; gys=[b[4] for b in buf]; gzs=[b[5] for b in buf]
                mov = std([acc_mag(*b[:3]) for b in buf])>120 or gyro_rms(gxs,gys,gzs)>150
                # crude posture via mean Z only (kept simple):
                meanz=sum(azs)/len(azs)
                if   meanz < -300:        label,conf="lying",0.88
                elif mov:                 label,conf="moving",0.82
                else:                     label,conf="standing",0.82
                print(datetime.now(timezone.utc).isoformat(),f"| {label:<10} conf={conf:.2f}")

        await c.start_notify(CHR, on_notify)
        try:
            while True: await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await c.stop_notify(CHR)

# ---------- Mode: collect labeled raw to CSV ----------
async def mode_collect(label, csv_path):
    header = not os.path.exists(csv_path)
    f=open(csv_path,"a",newline=""); w=csv.writer(f)
    if header: w.writerow(["ts","ax","ay","az","gx","gy","gz","label"])
    addr=await find_addr()
    async with BleakClient(addr, timeout=15) as c:
        await c.connect(); print("Connected?", c.is_connected)
        print(f"Collecting label='{label}' → {csv_path}  (Ctrl+C to stop)")
        async def on_notify(_, data: bytearray):
            if len(data)!=16: return
            ts,ax,ay,az,gx,gy,gz=struct.unpack("<I6h", data)
            w.writerow([ts,ax,ay,az,gx,gy,gz,label])
        await c.start_notify(CHR, on_notify)
        try:
            while True: await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await c.stop_notify(CHR); f.close()

# ---------- Mode: train (CSV -> model) ----------
def mode_train(csv_path, model_path):
    import numpy as np, pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from joblib import dump

    df=pd.read_csv(csv_path)
    needed={"ts","ax","ay","az","gx","gy","gz","label"}
    if not needed.issubset(df.columns): sys.exit("CSV missing columns")

    X=[]; y=[]
    arr=df[["ax","ay","az","gx","gy","gz","label"]].to_numpy()
    N=len(arr)
    for s in range(0, N-WIN+1, WIN):       # non-overlap 2s
        seg=arr[s:s+WIN]
        lab=seg[:,-1]
        # majority label in this 2s
        vals,counts=np.unique(lab, return_counts=True)
        lbl=str(vals[np.argmax(counts)])
        dat=seg[:,:6].astype(float)
        ax,ay,az,gx,gy,gz = [dat[:,i] for i in range(6)]
        def stats(v): return [v.mean(), v.std(), v.min(), v.max(), v.max()-v.min()]
        feats=[]
        for v in (ax,ay,az,gx,gy,gz): feats+=stats(v)
        mag=np.sqrt(ax*ax+ay*ay+az*az)
        feats += [mag.mean(), mag.std(), np.mean(np.abs(ax)+np.abs(ay)+np.abs(az))]  # SMA
        X.append(feats); y.append(lbl)

    X=np.array(X); y=np.array(y)
    Xtr,Xte,Ytr,Yte=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
    clf=DecisionTreeClassifier(max_depth=12,class_weight="balanced",random_state=42)
    clf.fit(Xtr,Ytr)
    yp=clf.predict(Xte)
    print("ACC:", accuracy_score(Yte,yp))
    print(classification_report(Yte,yp,zero_division=0))
    from joblib import dump; dump(clf, model_path)
    print("Saved", model_path)

# ---------- Mode: infer (real-time ML; 2s windows + instant FALL) ----------
async def mode_infer(model_path):
    import numpy as np
    from joblib import load
    clf=load(model_path)
    print("Model:", model_path, "| classes:", list(clf.classes_))

    buf=deque(maxlen=WIN); fallbuf=deque(maxlen=int(1.0*FS))
    last_imp=0.0; since=0
    addr=await find_addr()
    async with BleakClient(addr, timeout=15) as c:
        await c.connect(); print("Connected?", c.is_connected)
        print("Infer mode: 2s windows; FALL is instant.")

        async def on_notify(_, data: bytearray):
            nonlocal last_imp, since
            if len(data)!=16: return
            ts,ax,ay,az,gx,gy,gz=struct.unpack("<I6h", data)
            buf.append([ax,ay,az,gx,gy,gz]); fallbuf.append([ax,ay,az]); since+=1

            # FALL: impact + ~1s stillness
            m=acc_mag(ax,ay,az); now=time.time()
            if m>IMPACT: last_imp=now
            if last_imp and (now-last_imp)<IMPACT_WIN and len(fallbuf)>int(0.8*FS):
                ms=[acc_mag(*s) for s in list(fallbuf)[-int(0.8*FS):]]
                if std(ms)<STILL_STD:
                    print(datetime.now(timezone.utc).isoformat(),"| FALL        conf=0.95"); last_imp=0.0

            # Every 2s (100 samples) -> classify last 2s window
            if since>=WIN and len(buf)==WIN:
                since=0
                dat=np.array(list(buf),dtype=float)
                ax,ay,az,gx,gy,gz = [dat[:,i] for i in range(6)]
                def stats(v): return [v.mean(), v.std(), v.min(), v.max(), v.max()-v.min()]
                feats=[]
                for v in (ax,ay,az,gx,gy,gz): feats+=stats(v)
                mag=np.sqrt(ax*ax+ay*ay+az*az)
                feats += [mag.mean(), mag.std(), np.mean(np.abs(ax)+np.abs(ay)+np.abs(az))]
                proba=getattr(clf,"predict_proba",None)
                if proba:
                    p=clf.predict_proba([feats])[0]; k=int(np.argmax(p)); lab=clf.classes_[k]; conf=float(p[k])
                else:
                    lab=str(clf.predict([feats])[0]); conf=1.0
                print(datetime.now(timezone.utc).isoformat(), f"| {lab:<10} conf={conf:.2f}")

        await c.start_notify(CHR, on_notify)
        try:
            while True: await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await c.stop_notify(CHR)

# ---------- CLI ----------
def main():
    ap=argparse.ArgumentParser("edge_ml")
    sub=ap.add_subparsers(dest="cmd")
    c=sub.add_parser("collect"); c.add_argument("--label",required=True); c.add_argument("--csv",required=True)
    t=sub.add_parser("train");   t.add_argument("--csv",required=True);   t.add_argument("--model",required=True)
    i=sub.add_parser("infer");   i.add_argument("--model",required=True)
    args=ap.parse_args()

    if args.cmd=="collect": asyncio.run(mode_collect(args.label,args.csv))
    elif args.cmd=="train":  mode_train(args.csv,args.model)
    elif args.cmd=="infer":  asyncio.run(mode_infer(args.model))
    else:                    asyncio.run(mode_heuristic())

if __name__=="__main__":
    main()