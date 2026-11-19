import cv2
import numpy as np


def write_relative_ply(path, depth, frame=None, stride=4, z_scale=200, origin=(0, 0), mask=None):
    """Export pixel x/y and scaled relative z from a dense depth proxy."""
    if depth.ndim != 2 or not np.isfinite(depth).all() or depth.min() < 0 or depth.max() > 1:
        raise ValueError("Expected a dense finite 2D relative depth array in [0,1]")
    if stride < 1 or not np.isfinite(z_scale) or z_scale <= 0:
        raise ValueError("stride and z_scale must be positive")
    if mask is None:
        ys, xs = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    else:
        if mask.shape != depth.shape:
            raise ValueError("Point mask must align with the teacher depth")
        ys, xs = np.nonzero(mask)
        ys, xs = ys[::stride], xs[::stride]
        if not len(ys):
            raise ValueError("No teacher points passed screening")
    zs = depth[ys, xs]
    colors = (
        cv2.applyColorMap(np.uint8(zs.reshape(-1, 1) * 255), cv2.COLORMAP_TURBO).reshape(-1, 3)
        if frame is None else frame[ys, xs].reshape(-1, 3)
    )
    points = np.column_stack([xs.ravel() + origin[0], ys.ravel() + origin[1],
                              zs.ravel() * z_scale, colors[:, ::-1]])
    header = (
        f"ply\nformat ascii 1.0\ncomment x y in pixels and z a scaled relative proxy\n"
        f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header"
    )
    np.savetxt(path, points, fmt="%d %d %.6f %d %d %d", header=header, comments="")

