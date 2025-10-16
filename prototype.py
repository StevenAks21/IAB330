# edge.py — 2s live labels (standing / moving / transition / lying / FALL)
import asyncio, struct, time, math
from collections import deque
from datetime import datetime, timezone
from bleak import BleakClient, BleakScanner

SVC="12345678-1234-5678-1234-56789abcdef0"
CHR="12345678-1234-5678-1234-56789abcdef1"
NAME="Nano33-FallServer"
FS=50; WIN=2*FS                                     # 2 seconds
IMPACT=2800; STILL_STD=80; UPR=30; LIE=60           # thresholds (mg / deg)
buf=deque(maxlen=WIN); fallbuf=deque(maxlen=FS)     # ~1s fall check
gref=None; last_imp=0.0; last_print=0.0
cal_until=time.time()+1.0; cx=cy=cz=cN=0            # ~1s quick calibration

def mag(ax,ay,az): return math.sqrt(ax*ax+ay*ay+az*az)
def angle_to_ref(mx,my,mz):
    global gref
    if not gref:
        k=max((abs(mx),0),(abs(my),1),(abs(mz),2))[1]
        gref=(1,0,0) if k==0 else ((0,1,0) if k==1 else (0,0,1))
    n=math.sqrt(mx*mx+my*my+mz*mz) or 1.0
    a=(mx/n,my/n,mz/n); c=max(-1,min(1,a[0]*gref[0]+a[1]*gref[1]+a[2]*gref[2]))
    return math.degrees(math.acos(c))
def std(vals):
    if not vals: return 0.0
    m=sum(vals)/len(vals); return math.sqrt(sum((v-m)*(v-m) for v in vals)/len(vals))
def gyro_rms(gx,gy,gz):
    n=len(gx) or 1
    s=sum(gx[i]*gx[i]+gy[i]*gy[i]+gz[i]*gz[i] for i in range(n))
    return math.sqrt(s/n)

async def find_addr():
    try: devs=await BleakScanner.discover(timeout=6.0, service_uuids=[SVC])
    except TypeError: devs=await BleakScanner.discover(timeout=6.0)
    for d in devs:
        uu=(getattr(d,"metadata",{}) or {}).get("uuids",[])
        if SVC.lower() in [u.lower() for u in uu] or (d.name or "").lower().find(NAME.lower())!=-1:
            print(f"Found {d.name or '(no name)'} {d.address}"); return d.address
    raise RuntimeError("Device not found — reset the Nano and try again.")

async def main():
    addr=await find_addr()
    async with BleakClient(addr, timeout=15) as c:
        await c.connect(); print("Connected?", c.is_connected)

        async def on_notify(_, data: bytearray):
            global gref,last_imp,last_print,cx,cy,cz,cN,cal_until
            if len(data)!=16: return
            ts,ax,ay,az,gx,gy,gz=struct.unpack("<I6h", data)

            # quick 1s calibration (stand still upright)
            if time.time()<cal_until:
                cx+=ax; cy+=ay; cz+=az; cN+=1
                if time.time()+0.001>=cal_until and cN>0:
                    mx,my,mz=cx/cN,cy/cN,cz/cN
                    n=math.sqrt(mx*mx+my*my+mz*mz) or 1.0
                    gref=(mx/n,my/n,mz/n)
                    print(f"Calibrated up: {gref}")
                return

            buf.append([ax,ay,az,gx,gy,gz]); fallbuf.append([ax,ay,az])
            m=mag(ax,ay,az); now=time.time()
            if m>IMPACT: last_imp=now

            # instant FALL: impact + ~1s stillness
            if last_imp and (now-last_imp)<1.5 and len(fallbuf)>int(0.8*FS):
                ms=[mag(*s) for s in fallbuf][-int(0.8*FS):]
                if std(ms)<STILL_STD:
                    print(datetime.now(timezone.utc).isoformat(),"| FALL        conf=0.95"); last_imp=0.0

            # print every 2s when buffer filled
            if len(buf)==WIN and (now-last_print)>=2.0:
                last_print=now
                axs=[b[0] for b in buf]; ays=[b[1] for b in buf]; azs=[b[2] for b in buf]
                gxs=[b[3] for b in buf]; gys=[b[4] for b in buf]; gzs=[b[5] for b in buf]
                ang=angle_to_ref(sum(axs)/WIN, sum(ays)/WIN, sum(azs)/WIN)
                mov = std([mag(*b[:3]) for b in buf]) > 120 or gyro_rms(gxs,gys,gzs) > 150
                if   ang> LIE:           label,conf="lying",0.90
                elif ang<UPR and mov:    label,conf="moving",0.85
                elif ang<UPR:            label,conf="standing",0.85
                else:                    label,conf="transition",0.75
                print(datetime.now(timezone.utc).isoformat(),f"| {label:<10} conf={conf:.2f}")

        await c.start_notify(CHR, on_notify)
        try:
            while True: await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await c.stop_notify(CHR)

if __name__=="__main__":
    asyncio.run(main())