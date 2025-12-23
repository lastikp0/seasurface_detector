#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    out_path = Path(args.out)

    lines = ["source,frame,class_id,x,y,w,h"]
    for img_path in sorted([p for p in images_dir.iterdir() if p.is_file()]):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        for row in label_path.read_text().splitlines():
            if not row.strip():
                continue
            cls, xc, yc, bw, bh = row.split()[:5]
            cls = int(float(cls))
            xc = float(xc) * w
            yc = float(yc) * h
            bw = float(bw) * w
            bh = float(bh) * h
            x = int(round(xc - bw / 2))
            y = int(round(yc - bh / 2))
            bw_i = int(round(bw))
            bh_i = int(round(bh))
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            bw_i = max(1, min(bw_i, w - x))
            bh_i = max(1, min(bh_i, h - y))
            lines.append(f"{img_path.name},0,{cls},{x},{y},{bw_i},{bh_i}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path} ({len(lines)-1} boxes)")

if __name__ == "__main__":
    main()
