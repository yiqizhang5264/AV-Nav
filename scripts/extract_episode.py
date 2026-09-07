"""Extract one existing source episode for engineering diagnostics only."""
import argparse
import gzip
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--source',required=True)
p.add_argument('--index',required=True,type=int)
p.add_argument('--output',required=True)
args=p.parse_args()
with gzip.open(args.source,'rt') as f:
    data=json.load(f)
data['episodes']=[data['episodes'][args.index]]
dest=Path(args.output)
if dest.exists():
    p.error('Output exists')
dest.write_bytes(gzip.compress(json.dumps(data,sort_keys=True).encode(),mtime=0))
print(json.dumps(data['episodes'][0],ensure_ascii=False)[:300])
