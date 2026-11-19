"""Compute depth errors, high-error detection metrics, and risk–coverage curves."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .evaluate import errors, read_column, read_rows, reliability


SCORES = ("u_gate", "sigma_pred", "disagreement")


def evaluate(rows, expected_count=None):
    required = {"sample_id", "target", "prediction", *SCORES}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Expected columns: {', '.join(sorted(required))}")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} samples, got {len(rows)}")
    ids = [row["sample_id"].strip() for row in rows]
    if any(not sample_id for sample_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Every sample_id must be nonempty and unique")
    target, prediction = (read_column(rows, key) for key in ("target", "prediction"))
    if any(np.any((col < 0) | (col > 1)) for col in (target, prediction)):
        raise ValueError("Target and fused prediction must lie in [0, 1]")
    absolute = np.abs(prediction - target)
    threshold = float(np.quantile(absolute, 0.80))
    positive = absolute >= threshold
    if positive.all() or not positive.any():
        raise ValueError("The percentile rule must produce both positive and negative examples")
    summary = errors(target, prediction)
    summary.update(high_error_threshold=threshold, n_high_error=int(positive.sum()))
    for tolerance in (0.05, 0.07, 0.10):
        summary[f"pct_abs_error_lt_{tolerance:.2f}"] = float(100 * np.mean(absolute < tolerance))

    metrics, curve = [], []
    for name in SCORES:
        score = read_column(rows, name)
        if np.any(score < 0):
            raise ValueError(f"{name} must be nonnegative")
        result = reliability(absolute, score, threshold)
        correlation = None
        if np.ptp(score) and np.ptp(absolute):
            correlation = float(np.corrcoef(score, absolute)[0, 1])
        metrics.append({"score": name, "auroc": result["auroc"],
                        "auprc_ap": result["auprc_ap"], "pearson_r": correlation})
        if name == "disagreement":
            continue
        order = np.argsort(score, kind="stable")
        for tenth in range(1, 11):
            retained = max(1, len(rows) * tenth // 10)
            curve.append({"score": name, "coverage": tenth / 10, "retained": retained,
                          "actual_coverage": retained / len(rows),
                          "mae": float(absolute[order[:retained]].mean())})
    metrics.append({"score": "chance_reference", "auroc": 0.5,
                    "auprc_ap": float(positive.mean()), "pearson_r": None})
    labels = [{"sample_id": sample_id, "absolute_error": float(error), "high_error": int(flag)}
              for sample_id, error, flag in zip(ids, absolute, positive)]
    return summary, metrics, curve, labels


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()
    summary, metrics, curve, labels = evaluate(read_rows(args.predictions), args.expected_count)
    args.out.mkdir(parents=True, exist_ok=False)
    payload = json.dumps(summary, indent=2, allow_nan=False)
    (args.out / "depth_metrics.json").write_text(payload + "\n", encoding="utf-8")
    write_csv(args.out / "metrics.csv", metrics)
    write_csv(args.out / "risk_coverage.csv", curve)
    write_csv(args.out / "high_error_labels.csv", labels)
    print(payload)


if __name__ == "__main__":
    main()
