"""Detect a fill port and export ellipse, relative depth, and optional teacher data."""
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from PIL import Image

from .geometry import detect as detect_gated
from .hierarchy import detect as detect_legacy
from .hierarchy.paper import detect as detect_paper
from .images import YoloDetector, aligned_crop, image_paths, read_bgr, student_crop, write_bgr
from .pointcloud import write_relative_ply
from .student import load_student, predict
from .teacher import DepthAnythingTeacher, sample_ring


def run(args):
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    detector = None if args.roi_input else YoloDetector(
        args.yolo, device, args.imgsz, args.conf)
    student = load_student(args.student, device) if args.student else None
    teacher = DepthAnythingTeacher(args.dav2_repo, args.dav2_weights, device) if args.dav2_repo else None
    sources = image_paths(args.source)
    # A fresh directory keeps results from different weights/runs separate.
    args.out.mkdir(parents=True, exist_ok=False)
    detectors = {"paper": detect_paper, "gated": detect_gated, "legacy": detect_legacy}
    fit_ellipse = detectors[args.geometry]
    records, targets = [], []
    for src in sources:
        random.seed(42)
        np.random.seed(42)
        cv2.setRNGSeed(42)
        frame = read_bgr(src)
        h, w = frame.shape[:2]
        proposal = detector(frame) if detector else ([0, 0, w, h], None)
        record = {"image": src.name, "status": "no_detection", "geometry_mode": args.geometry}
        records.append(record)
        if proposal is None:
            print(src.name, record["status"], flush=True)
            continue
        box, score = proposal
        record.update(box_xyxy=box, detection_confidence=score)
        ellipse = fit_ellipse(frame, box)
        record["status"] = "ok" if ellipse else "no_ellipse"
        record["ellipse"] = asdict(ellipse) if ellipse else None
        sample_dir = args.out / src.name
        sample_dir.mkdir()
        if ellipse or args.roi_input:
            if args.roi_input:
                roi = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                record["roi_mode"] = "pre_cropped"
            elif args.aligned:
                roi = aligned_crop(frame, ellipse)
                record["roi_mode"] = "ellipse_aligned"
            else:
                roi = student_crop(frame, box)
                record["roi_mode"] = "detection_box"
            roi.save(sample_dir / "roi.png")
            if student:
                record["student"] = predict(*student, roi, device)
                if ellipse is None:
                    record["status"] = "roi_only"
        if ellipse:
            annotated = frame.copy()
            cv2.ellipse(annotated, ellipse.opencv, (0, 255, 0), 2)
            center = (round(ellipse.cx), round(ellipse.cy))
            cv2.drawMarker(annotated, center, (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
            write_bgr(sample_dir / "ellipse.png", annotated)
        if teacher and ellipse:
            depth = teacher(frame)
            np.save(sample_dir / "teacher_proxy.npy", depth)
            z, confidence, mask = sample_ring(depth, ellipse)
            record["teacher"] = {"Z_rel_cam": z, "confidence": confidence}
            Image.fromarray(mask).save(sample_dir / "ring_mask.png")
            write_bgr(sample_dir / "teacher_proxy.png",
                      cv2.applyColorMap(np.uint8(depth * 255), cv2.COLORMAP_TURBO))
            targets.append({"image": f"{src.name}/roi.png", "Z_rel_cam": z,
                            "confidence": confidence})
            if args.cloud:
                # Convert the dense proxy to the distance convention of Z_rel_cam.
                relative_depth = 1.0 - depth
                reference = record["student"]["Z_rel_cam"]
                retained = (mask != 0) & (np.abs(relative_depth - reference) <= args.cloud_tolerance)
                record["cloud"] = {"retained_points": int(retained.sum()),
                                   "relative_depth_tolerance": args.cloud_tolerance}
                if retained.any():
                    write_relative_ply(sample_dir / "screened_relative.ply",
                                       relative_depth, frame, stride=1, mask=retained)
        print(src.name, record["status"], flush=True)
    with (args.out / "results.json").open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False, allow_nan=False)
    if teacher:
        with (args.out / "targets.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["image", "Z_rel_cam", "confidence"])
            writer.writeheader()
            writer.writerows(targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="image or a directory of images")
    parser.add_argument("--out", required=True, type=Path, help="new output directory")
    parser.add_argument("--yolo", type=Path, default=Path("models/yolo_safe5.pt"))
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", help="cpu, cuda, or cuda:0")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--student", type=Path, help="run the DualKD student on the ROI")
    parser.add_argument("--roi-input", action="store_true", help="input is already an ROI; skip YOLO")
    parser.add_argument("--geometry", choices=("paper", "gated", "legacy"), default="paper",
                        help="paper: Section 2.2 reference; gated: previous fast path; legacy: archive")
    parser.add_argument("--aligned", action=argparse.BooleanOptionalAction, default=True,
                        help="ellipse-align full-image ROI crops; use --no-aligned for box crops")
    parser.add_argument("--dav2-repo", type=Path)
    parser.add_argument("--dav2-weights", type=Path)
    parser.add_argument("--cloud", action="store_true",
                        help="export teacher thin-ring points screened by the fused student estimate")
    parser.add_argument("--cloud-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    if bool(args.dav2_repo) != bool(args.dav2_weights):
        parser.error("Set both --dav2-repo and --dav2-weights for teacher inference")
    if args.cloud and (not args.dav2_repo or not args.student):
        parser.error("--cloud needs both the teacher and the student")
    if not 0 < args.cloud_tolerance <= 1:
        parser.error("--cloud-tolerance must be in (0, 1]")
    if args.threads < 1 or args.imgsz < 32 or not 0 < args.conf <= 1:
        parser.error("Expected threads >= 1, imgsz >= 32 and 0 < conf <= 1")
    run(args)


if __name__ == "__main__":
    main()


