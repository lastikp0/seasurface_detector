#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names = [x.strip() for x in args.names.split(",") if x.strip()]
    d = {
        "path": ".",
        "train": str((Path(args.train_dir) / "images").as_posix()),
        "val": str((Path(args.val_dir) / "images").as_posix()),
        "names": {i: n for i, n in enumerate(names)},
    }
    Path(args.out).write_text(yaml.safe_dump(d, sort_keys=False))
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
