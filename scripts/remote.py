"""Send LF-normalized Bash scripts through SSH; no password storage."""
import argparse
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument("script")
p.add_argument("--host",default="zyq@106.3.202.138")
p.add_argument("--port",default="50001")
args = p.parse_args()
text = Path(args.script).read_text(encoding="utf-8").replace("\r\n","\n")
raise SystemExit(subprocess.run(["ssh","-p",args.port,args.host,"bash","-s"],input=text.encode("utf-8")).returncode)
