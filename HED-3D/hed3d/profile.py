"""Extract a measured-in-pixels section from one or more dense model outputs."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("maps", nargs="+", type=Path, help="aligned .npy depth maps in [0, 1]")
    parser.add_argument("--line", nargs=4, type=float, required=True,
                        metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 2 or not np.isfinite(args.line).all():
        parser.error("Expected finite line coordinates and at least two samples")
    x0, y0, x1, y1 = args.line
    x = np.linspace(x0, x1, args.samples)
    y = np.linspace(y0, y1, args.samples)
    profiles, shape = [], None
    for path in args.maps:
        array = np.load(path, allow_pickle=False)
        if array.ndim != 2 or not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
            raise ValueError(f"Expected a finite 2D map in [0,1]: {path}")
        if shape and array.shape != shape:
            raise ValueError("All maps must have the same shape and coordinate frame")
        shape = array.shape
        if min(x) < 0 or min(y) < 0 or max(x) > shape[1] - 1 or max(y) > shape[0] - 1:
            raise ValueError(f"Line lies outside map: {path}")
        profiles.append(map_coordinates(array.astype(float), [y, x], order=1))
    with args.out.open("x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["fraction", "x_px", "y_px", *[p.name for p in args.maps]])
        writer.writerows(zip(np.linspace(0, 1, args.samples), x, y, *profiles))


if __name__ == "__main__":
    main()

