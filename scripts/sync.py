"""Explicit GitHub-mediated synchronization, usable on Windows and Linux."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=["status","pull","push"])
    args = p.parse_args()
    print(git("status","--short","--branch"))
    if args.action == "status":
        return
    if git("status","--porcelain"):
        raise SystemExit("Working tree is dirty. Commit explicit files before synchronization; nothing was stashed or overwritten.")
    if args.action == "pull":
        subprocess.run(["git","pull","--ff-only"],cwd=ROOT,check=True)
        subprocess.run(["git","submodule","update","--init","--recursive"],cwd=ROOT,check=True)
    else:
        subprocess.run(["git","push"],cwd=ROOT,check=True)


if __name__ == "__main__":
    main()
