from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

SIZE = 434
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_paths(source):
    if source.is_file():
        return [source]
    paths = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(f"No images in {source}")
    return paths


def read_bgr(path):
    with Image.open(path) as image:
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def write_bgr(path, frame):
    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(path)


def student_crop(frame, box):
    """Re-create the 1.5× box crop and 434 px letterbox from 图像调整.py."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * 0.25, (y2 - y1) * 0.25
    crop = image.crop((max(0, x1 - dx), max(0, y1 - dy),
                       min(image.width, x2 + dx), min(image.height, y2 + dy)))
    ratio = SIZE / max(crop.size)
    crop = crop.resize((max(1, int(crop.width * ratio)), max(1, int(crop.height * ratio))),
                       Image.Resampling.LANCZOS)
    w, h = SIZE - crop.width, SIZE - crop.height
    return ImageOps.expand(crop, (w // 2, h // 2, w - w // 2, h - h // 2), fill=0)


def aligned_crop(frame, ellipse):
    matrix = cv2.getRotationMatrix2D((ellipse.cx, ellipse.cy), ellipse.angle_deg,
                                     SIZE / (1.5 * ellipse.major_px))
    matrix[:, 2] += np.array([SIZE / 2 - ellipse.cx, SIZE / 2 - ellipse.cy])
    crop = cv2.warpAffine(frame, matrix, (SIZE, SIZE))
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


class YoloDetector:
    def __init__(self, weights, device, imgsz, confidence):
        from ultralytics import YOLO

        self.model = YOLO(str(Path(weights).resolve(strict=True)))
        self.device = device
        self.imgsz = imgsz
        self.confidence = confidence

    def __call__(self, frame):
        result = self.model.predict(frame, imgsz=self.imgsz, conf=self.confidence,
                                    max_det=1, device=self.device, verbose=False)[0]
        if len(result.boxes) == 0:
            return None
        return result.boxes.xyxy[0].cpu().tolist(), float(result.boxes.conf[0])

