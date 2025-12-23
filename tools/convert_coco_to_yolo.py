#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

IGNORED_NAMES = {"ignored"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_json", required=True, help="Path to COCO instances_*.json")
    ap.add_argument("--images_dir", required=True, help="Directory with images referenced by COCO JSON")
    ap.add_argument("--out_dir", required=True, help="Output split dir (creates images/ and labels/)")
    ap.add_argument("--force", action="store_true", help="Delete out_dir if exists")
    args = ap.parse_args()

    coco_json = Path(args.coco_json)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)

    if not coco_json.exists():
        raise SystemExit(f"ERROR: coco_json not found: {coco_json}")
    if not images_dir.exists():
        raise SystemExit(f"ERROR: images_dir not found: {images_dir}")

    if out_dir.exists():
        if not args.force:
            raise SystemExit(
                f"ERROR: out_dir already exists: {out_dir}\n"
                f"Delete it or rerun with --force"
            )
        import shutil
        shutil.rmtree(out_dir)

    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    data = json.loads(coco_json.read_text())

    cats = data.get("categories", [])
    if not cats:
        raise SystemExit("ERROR: COCO json has no 'categories'")

    cats_kept = []
    for c in cats:
        name = str(c.get("name", "")).strip().lower()
        if name in IGNORED_NAMES:
            continue
        cats_kept.append(c)

    if not cats_kept:
        raise SystemExit("ERROR: after filtering ignored, no categories left")

    cats_sorted = sorted(cats_kept, key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats_sorted)}
    idx_to_name = {i: c.get("name", str(c["id"])) for i, c in enumerate(cats_sorted)}

    images = data.get("images", [])
    if not images:
        raise SystemExit("ERROR: COCO json has no 'images'")

    img_id_to_info = {}
    for im in images:
        img_id_to_info[im["id"]] = (im["file_name"], im.get("width", None), im.get("height", None))

    for img_id, (fname, _, _) in img_id_to_info.items():
        (out_labels / (Path(fname).stem + ".txt")).write_text("")

    anns = data.get("annotations", [])
    if not anns:
        print("WARN: no annotations in COCO json (labels will be empty)")

    per_image_lines = {img_id: [] for img_id in img_id_to_info.keys()}

    skipped_unknown_cat = 0
    skipped_no_wh = 0

    for a in anns:
        img_id = a.get("image_id")
        cat_id = a.get("category_id")
        bbox = a.get("bbox")

        if img_id not in img_id_to_info or bbox is None:
            continue

        if cat_id not in cat_id_to_idx:
            skipped_unknown_cat += 1
            continue

        fname, W, H = img_id_to_info[img_id]
        if not W or not H:
            skipped_no_wh += 1
            continue

        x, y, bw, bh = bbox

        xc = (x + bw / 2.0) / W
        yc = (y + bh / 2.0) / H
        wn = bw / W
        hn = bh / H

        cls = cat_id_to_idx[cat_id]
        per_image_lines[img_id].append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

    for img_id, lines in per_image_lines.items():
        fname, _, _ = img_id_to_info[img_id]
        out_txt = out_labels / (Path(fname).stem + ".txt")
        out_txt.write_text("\n".join(lines) + ("\n" if lines else ""))
    
    (out_dir / "classes.txt").write_text("\n".join([idx_to_name[i] for i in range(len(idx_to_name))]) + "\n")

    linked = 0
    missing = 0
    for img_id, (fname, _, _) in img_id_to_info.items():
        src = images_dir / fname
        dst = out_images / Path(fname).name
        if not src.exists():
            missing += 1
            continue
        try:
            dst.symlink_to(src.resolve())
        except Exception:
            import shutil
            shutil.copy2(src, dst)
        linked += 1

    print(f"Done. out_dir={out_dir}")
    print(f"Images linked/copied: {linked}, missing: {missing}")
    print(f"Classes: {len(idx_to_name)}")
    print("Class mapping (index -> name):")
    for i in range(len(idx_to_name)):
        print(f"  {i}: {idx_to_name[i]}")
    if skipped_unknown_cat:
        print(f"WARN: skipped annotations with filtered/unknown category: {skipped_unknown_cat}")
    if skipped_no_wh:
        print(f"WARN: skipped annotations due to missing width/height: {skipped_no_wh}")

if __name__ == "__main__":
    main()
