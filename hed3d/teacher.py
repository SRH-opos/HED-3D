import sys
from pathlib import Path

import cv2
import numpy as np
import torch


class DepthAnythingTeacher:
    def __init__(self, repo, weights, device):
        repo = Path(repo).resolve(strict=True)
        if not (repo / "depth_anything_v2" / "dpt.py").is_file():
            raise ValueError(f"Pass the relative Depth Anything V2 repository root: {repo}")
        sys.path.insert(0, str(repo))
        from depth_anything_v2.dpt import DepthAnythingV2

        self.model = DepthAnythingV2(encoder="vitl", features=256,
                                     out_channels=[256, 512, 1024, 1024])
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model.to(device).eval()

    @torch.inference_mode()
    def __call__(self, frame):
        # Upstream infer_image converts its BGR input to RGB internally.
        proxy = self.model.infer_image(frame)
        if not np.isfinite(proxy).all() or np.ptp(proxy) <= 1e-6:
            raise ValueError("Depth Anything V2 produced a non-finite or constant map")
        return ((proxy - proxy.min()) / np.ptp(proxy)).astype(np.float32)


def sample_ring(depth, ellipse, half_width=2, angles=120):
    """Sample equal parameter angles along outward unit normals, in full-image pixels."""
    t = np.arange(angles) * (2 * np.pi / angles)
    a, b = ellipse.major_px / 2, ellipse.minor_px / 2
    rim = np.stack([a * np.cos(t), b * np.sin(t)], axis=-1)
    normal = np.stack([np.cos(t) / a, np.sin(t) / b], axis=-1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    offsets = np.arange(-half_width, half_width + 1)
    coords = rim[:, None, :] + normal[:, None, :] * offsets[None, :, None]
    angle = np.deg2rad(ellipse.angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    coords = np.rint(coords @ [[c, s], [-s, c]] + [ellipse.cx, ellipse.cy]).astype(int)
    x, y = coords[..., 0], coords[..., 1]
    valid = (x >= 0) & (x < depth.shape[1]) & (y >= 0) & (y < depth.shape[0])
    mask = np.zeros(depth.shape, dtype=np.uint8)
    mask[y[valid], x[valid]] = 255
    profile = np.array([np.median(depth[y[i, ok], x[i, ok]])
                        for i, ok in enumerate(valid) if ok.any()])
    if len(profile) < angles * 0.30:
        raise ValueError("Too few in-frame ring angles")
    lo, hi = np.quantile(profile, [0.05, 0.95])
    retained = profile[(profile >= lo) & (profile <= hi)]
    value = float(np.median(retained)) if retained.size else float(np.median(profile))
    # A documented heuristic for preparing new supervision; original trainer used no weights.
    confidence = float(ellipse.coverage * len(profile) / angles /
                       (1 + np.median(np.abs(profile - value))))
    return 1.0 - value, confidence, mask

