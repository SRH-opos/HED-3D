"""Export one frozen DualKD prediction and all three scores per labeled ROI."""
import argparse
import csv
import math
from pathlib import Path

from PIL import Image
import torch

from .student import load_student, predict


def read_manifest(path, root, expected_count):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"sample_id", "image", "Z_rel_cam"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest must have sample_id, image, and Z_rel_cam columns")
        rows = list(reader)
    if not rows or (expected_count is not None and len(rows) != expected_count):
        raise ValueError(f"Unexpected manifest size: {len(rows)} (expected {expected_count})")
    root = root.resolve(strict=True)
    ids, paths = set(), set()
    for row in rows:
        sample_id = row["sample_id"].strip()
        path = (root / row["image"]).resolve()
        target = float(row["Z_rel_cam"])
        if not sample_id or sample_id in ids or path in paths:
            raise ValueError(f"An image and a nonempty sample ID must each occur once: {sample_id}")
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing image or path outside --root: {row['image']}")
        if not math.isfinite(target) or not 0 <= target <= 1:
            raise ValueError(f"Target outside [0, 1]: {sample_id}")
        ids.add(sample_id)
        paths.add(path)
        yield sample_id, path, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/dualkd_student.pth"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or (args.expected_count is not None and args.expected_count < 1):
        parser.error("Thread and expected sample counts must be positive")
    torch.set_num_threads(args.threads)
    manifest = list(read_manifest(args.manifest, args.root, args.expected_count))
    student, gate = load_student(args.checkpoint, args.device)
    rows = []
    for sample_id, path, target in manifest:
        with Image.open(path) as image:
            score = predict(student, gate, image, args.device)
        rows.append({
            "sample_id": sample_id, "target": target,
            "prediction": score.pop("Z_rel_cam"),
            "disagreement": abs(score["mu_det"] - score["mu_nig"]),
            **score,
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} ROI predictions to {args.out}")


if __name__ == "__main__":
    main()
