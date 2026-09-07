"""Start this project's loopback model servers with explicit PID/log files."""
import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", default="3")
    p.add_argument("--dino-config", required=True)
    p.add_argument("--only",choices=['grounding_dino','blip2itm','sam','yolov7'])
    args = p.parse_args()
    upstream = ROOT/"external/vlfm"
    logs = ROOT/"runs/services"
    logs.mkdir(parents=True,exist_ok=True)
    services = [("grounding_dino","GROUNDING_DINO_PORT",13181),("blip2itm","BLIP2ITM_PORT",13182),
                ("sam","SAM_PORT",13183),("yolov7","YOLOV7_PORT",13184)]
    if args.only:
        services=[s for s in services if s[0]==args.only]
    for _,key,default in services:
        port = int(os.environ.get(key,default))
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1",port)) == 0:
                raise SystemExit(f"Port {port} already in use; inspect existing project services first")
    env = os.environ.copy()
    env.setdefault('HF_HUB_OFFLINE','1')
    env.setdefault('TRANSFORMERS_OFFLINE','1')
    env.update(CUDA_VISIBLE_DEVICES=args.gpu,GROUNDING_DINO_CONFIG=str(Path(args.dino_config).resolve()),
               MOBILE_SAM_CHECKPOINT=str(upstream/"data/mobile_sam.pt"),
               GROUNDING_DINO_WEIGHTS=str(upstream/"data/groundingdino_swint_ogc.pth"),
               CLASSES_PATH=str(upstream/"vlfm/vlm/classes.txt"),
               PYTHONPATH=os.pathsep.join([str(ROOT),str(upstream),str(upstream/"yolov7"),env.get("PYTHONPATH","")]))
    processes = json.loads((logs/'pids.json').read_text()) if (logs/'pids.json').exists() else {}
    for name,key,default in services:
        port = int(env.get(key,default))
        command=[sys.executable,"-u","-m",f"vlfm.vlm.{name}","--port",str(port)]
        if name=='grounding_dino':
            command=[sys.executable,'-u','-m','av_nav.dino_service','--port',str(port),
                     '--config',env['GROUNDING_DINO_CONFIG'],'--weights',env['GROUNDING_DINO_WEIGHTS']]
        with (logs/f"{name}.log").open("w") as stream:
            child = subprocess.Popen(command,
                                     cwd=upstream,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        processes[name] = dict(pid=child.pid,port=port)
    (logs/"pids.json").write_text(json.dumps(processes,indent=2))
    print(json.dumps(processes,indent=2))
    print("Launched; inspect logs and health-check before evaluation. Launch is not readiness.")


if __name__ == "__main__":
    main()
