"""Probe actual inference for all four services; exits nonzero on any failure."""
import os
from pathlib import Path
import sys
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'external/vlfm'))
import requests
from vlfm.vlm.server_wrapper import image_to_str

image=np.zeros((128,128,3),dtype=np.uint8)
payload=image_to_str(image)
requests_by_service=[
    ('gdino','GROUNDING_DINO_PORT',13181,dict(image=payload,caption='chair .')),
    ('blip2itm','BLIP2ITM_PORT',13182,dict(image=payload,txt='a photo of a chair')),
    ('mobile_sam','SAM_PORT',13183,dict(image=payload,bbox=[20,20,100,100])),
    ('yolov7','YOLOV7_PORT',13184,dict(image=payload))]
results={}
for name,key,default,data in requests_by_service:
    port=int(os.environ.get(key,default))
    try:
        response=requests.post(f'http://127.0.0.1:{port}/{name}',json=data,timeout=90)
        response.raise_for_status()
        response.json()
        results[name]='ok'
    except Exception as exc:
        results[name]=str(exc)
print(json.dumps(results,indent=2))
raise SystemExit(0 if all(v=='ok' for v in results.values()) else 1)
