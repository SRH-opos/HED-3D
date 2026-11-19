"""Ellipse fitting for the runnable pipeline, with checks against the YOLO box."""
import cv2
import numpy as np

from .hierarchy import Ellipse
from .hierarchy.axes import fit_major_first


def _gradient_edges(crop):
    rgb = cv2.GaussianBlur(crop.astype(np.float32) / 255, (5, 5), 1.0)
    dx = cv2.Sobel(rgb, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(rgb, cv2.CV_32F, 0, 1, ksize=3)
    a = (dx**2).sum(axis=2)
    b = (dx * dy).sum(axis=2)
    c = (dy**2).sum(axis=2)
    magnitude = np.sqrt((a + c + np.sqrt((a - c)**2 + 4 * b**2)) / 2)
    strong = np.quantile(magnitude, 0.90)
    if strong <= 1e-6:
        return np.zeros(crop.shape[:2], np.uint8)
    scaled = np.uint8(np.clip(magnitude * (128 / strong), 0, 255))
    local = cv2.boxFilter(scaled.astype(np.float32), -1, (31, 31))
    weak = (scaled >= 51) & (scaled > 1.15 * local)
    _, labels = cv2.connectedComponents(weak.astype(np.uint8))
    keep = np.unique(labels[(scaled >= 128) & weak])
    return np.uint8(np.isin(labels, keep)) * 255


def _measure(ellipse, points, box_center, box_size):
    (cx, cy), (major, minor), angle = ellipse
    aspect = minor / max(major, 1e-6)
    center_offset = np.linalg.norm((np.array([cx, cy]) - box_center) / box_size)
    if (not np.isfinite([cx, cy, major, minor, angle]).all() or
            not 0.50 * max(box_size) <= major <= 1.2 * max(box_size) or
            not 0.35 * min(box_size) <= minor or aspect < 0.28 or center_offset > 0.25):
        return None
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    unit = ((points - [cx, cy]) @ [[c, -s], [s, c]]) / [major / 2, minor / 2]
    deviation = np.abs(np.linalg.norm(unit, axis=1) - 1)
    inliers = deviation < max(0.025, 5 / minor)
    angles = np.mod(np.arctan2(unit[inliers, 1], unit[inliers, 0]), 2 * np.pi)
    if len(angles) < 24:
        return None
    bins = np.floor(angles * (36 / (2 * np.pi))).astype(int)
    cover = np.unique(bins).size / 36
    ordered = np.sort(angles)
    max_gap = np.rad2deg(np.diff(np.r_[ordered, ordered[0] + 2 * np.pi]).max())
    residual = float(np.median(deviation[inliers]))
    if cover < 0.80 or max_gap > 60 or residual > 0.02:
        return None
    size = min(major / max(box_size), 1.0)
    score = cover + 0.35 * size - 2.0 * center_offset - 5.0 * residual
    return score, inliers, cover, max_gap, residual


def detect(frame, box, pad=20):
    """Merge short edge gaps, fit closed/partial arcs, and rank supported candidates."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    box_size = np.array([x2 - x1, y2 - y1])
    if min(box_size) < 20:
        return None
    left, top = max(0, int(np.floor(x1)) - pad), max(0, int(np.floor(y1)) - pad)
    right, bottom = min(w, int(np.ceil(x2)) + pad), min(h, int(np.ceil(y2)) + pad)
    crop = frame[top:bottom, left:right]
    center = np.array([(x1 + x2) / 2 - left, (y1 + y2) / 2 - top])
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(cv2.createCLAHE(2, (8, 8)).apply(gray), (5, 5), 1.0)
    edges = cv2.Canny(gray, 30, 85)
    ys, xs = np.where(edges)
    if len(xs) < 24:
        return None
    points = np.column_stack([xs, ys]).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidates = []
    for mask in (edges, _gradient_edges(crop)):
        connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        arcs, _ = cv2.findContours(connected, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for arc in arcs:
            if len(arc) < max(24, max(box_size) * 0.35):
                continue
            ellipse = fit_major_first(arc)
            quality = _measure(ellipse, points, center, box_size)
            if quality is not None:
                candidates.append((quality, ellipse))
    if not candidates:
        return None
    quality, ellipse = max(candidates, key=lambda item: item[0][0])
    # Refine with edge points around the accepted ellipse. Keep the original on regression.
    refined = fit_major_first(points[quality[1]])
    new_quality = _measure(refined, points, center, box_size)
    if new_quality is not None and new_quality[0] > quality[0]:
        quality, ellipse = new_quality, refined
    (cx, cy), (major, minor), angle = ellipse
    _, inliers, cover, gap, residual = quality
    return Ellipse(float(cx + left), float(cy + top), float(major), float(minor), float(angle),
                   float(inliers.mean()), float(cover), float(gap), 3, float(residual))

