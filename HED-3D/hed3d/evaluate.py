"""Evaluate normalized predictions, or calibrate and test physical distance."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def read_column(rows, name):
    data = np.array([float(row[name]) for row in rows], dtype=float)
    if not data.size or not np.isfinite(data).all():
        raise ValueError(f"Empty or invalid column: {name}")
    return data


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def errors(reference, prediction):
    residual = prediction - reference
    absolute = np.abs(residual)
    return {"n": len(residual), "mae": float(absolute.mean()),
            "rmse": float(np.sqrt(np.mean(residual**2))), "max_ae": float(absolute.max())}


def reliability(error, sigma, threshold):
    if threshold is None:
        return {}
    positive = error >= threshold
    p, n = int(positive.sum()), int((~positive).sum())
    if not p or not n:
        return {"threshold": threshold, "n_high_error": p, "auroc": None, "auprc_ap": None}
    auc = (rankdata(sigma)[positive].sum() - p * (p + 1) / 2) / (p * n)
    levels = np.unique(sigma)[::-1]
    order = np.argsort(-sigma, kind="stable")
    sorted_scores = sigma[order]
    top = np.searchsorted(-sorted_scores, -levels, side="right")
    tp = np.cumsum(positive[order])[top - 1]
    ap = np.sum(np.diff(np.r_[0, tp / p]) * (tp / top))
    return {"threshold": threshold, "n_high_error": p, "auroc": float(auc), "auprc_ap": float(ap)}


def relative(args):
    rows = read_rows(args.csv)
    target, pred = (read_column(rows, key) for key in ("target", "prediction"))
    if any(np.any((col < 0) | (col > 1)) for col in (target, pred)):
        raise ValueError("Relative target and prediction must be in [0,1]")
    summary = errors(target, pred)
    absolute = np.abs(pred - target)
    for tolerance in (0.05, 0.07, 0.10):
        summary[f"pct_abs_error_lt_{tolerance:.2f}"] = float(100 * np.mean(absolute < tolerance))
    denominator = np.sum((target - target.mean())**2)
    summary["r2_against_target"] = (
        float(1 - np.sum((pred - target)**2) / denominator) if denominator > 0 else None)
    return summary


def metric(args):
    calibration = read_rows(args.calibration)
    test = read_rows(args.test)
    calibration_ids = [row["sample_id"] for row in calibration]
    test_ids = [row["sample_id"] for row in test]
    for ids in (calibration_ids, test_ids):
        if any(not item.strip() for item in ids) or len(set(ids)) != len(ids):
            raise ValueError("sample_id must be present and unique within each CSV")
    if set(calibration_ids) & set(test_ids):
        raise ValueError("Calibration and test sample IDs overlap")
    z_cal = read_column(calibration, "relative_depth")
    d_cal = read_column(calibration, "distance_mm")
    z_test = read_column(test, "relative_depth")
    actual = read_column(test, "distance_mm")
    if any(np.any((col < 0) | (col > 1)) for col in (z_cal, z_test)):
        raise ValueError("Relative depth must be in [0,1]")
    if np.any(d_cal <= 0) or np.any(actual <= 0):
        raise ValueError("Measured distances must be positive")
    if len(z_cal) < 2 or np.ptp(z_cal) < 1e-9:
        raise ValueError("Calibration requires at least two distinct relative depths")
    design = np.column_stack([z_cal, np.ones(len(calibration))])
    a, b = np.linalg.lstsq(design, d_cal, rcond=None)[0]
    predicted = a * z_test + b
    score = errors(actual, predicted)
    total = np.sum((actual - actual.mean())**2)
    score["r2"] = float(1 - np.sum((actual - predicted)**2) / total) if total > 0 else None
    return {"calibration_n": len(calibration), "a": float(a), "b": float(b), "test_mm": score,
            "samples": [{"sample_id": identifier, "predicted_mm": float(pred), "measured_mm": float(gt)}
                        for identifier, pred, gt in zip(test_ids, predicted, actual)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    rel = commands.add_parser("relative")
    rel.add_argument("csv", type=Path)
    rel.set_defaults(evaluate=relative)
    mm = commands.add_parser("metric")
    mm.add_argument("calibration", type=Path)
    mm.add_argument("test", type=Path)
    mm.set_defaults(evaluate=metric)
    for command in (rel, mm):
        command.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = json.dumps(args.evaluate(args), indent=2, ensure_ascii=False, allow_nan=False)
    if args.out:
        with args.out.open("x", encoding="utf-8") as file:
            file.write(result + "\n")
    else:
        print(result)


if __name__ == "__main__":
    main()


