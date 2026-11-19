"""In-memory adaptation of the original four ellipse-screening scripts."""
from dataclasses import dataclass

import cv2
import numpy as np

from . import contour, fit, recover, refine


@dataclass(frozen=True)
class Ellipse:
    cx: float
    cy: float
    major_px: float
    minor_px: float
    angle_deg: float
    support: float
    coverage: float
    max_gap_deg: float
    stage: int
    radial_residual: float | None = None

    @property
    def opencv(self):
        return ((self.cx, self.cy), (self.major_px, self.minor_px), self.angle_deg)


def detect(frame, box, pad=20):
    """Find the outermost accepted candidate in the highest-ranked cluster."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(np.floor(x1)) - pad), max(0, int(np.floor(y1)) - pad)
    x2, y2 = min(w, int(np.ceil(x2)) + pad), min(h, int(np.ceil(y2)) + pad)
    if min(x2 - x1, y2 - y1) < 20:
        return None
    crop = frame[y1:y2, x1:x2]
    outlines = contour.process_one(crop, [(0, 0, crop.shape[1], crop.shape[0])])
    mask = refine.load_mask_from_green(outlines)
    if not np.any(mask):
        return None
    mask, _, pixels = refine.refine_with_relax(mask)
    if pixels == 0:
        return None
    canvas = cv2.cvtColor(255 - mask, cv2.COLOR_GRAY2BGR)
    _, accepted = fit.process_one(canvas, fit.STRICT, [fit.RELAX1, fit.RELAX2])
    stage = 3
    if accepted:
        ellipse, metrics = accepted[0]["ellipse"], accepted[0]["metrics"]
    else:
        recovered = recover.process_one_image(canvas)
        if not recovered:
            return None
        ellipse, metrics, _ = recovered[0]
        stage = 4
    (cx, cy), (major, minor), angle = ellipse
    return Ellipse(float(cx + x1), float(cy + y1), float(major), float(minor),
                   float(angle), float(metrics["support"]), float(metrics["ang_cover"] / 360),
                   float(metrics["max_gap"]), stage)



