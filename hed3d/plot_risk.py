"""Render a supplied risk–coverage CSV as PNG and PDF."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

from .evaluate import read_rows


def plot(rows, stem, title=None):
    styles = {
        "u_gate": (r"SoftGate score $u_{\mathrm{gate}}$", "#225883", "-", "o"),
        "sigma_pred": (r"NIG predictive standard deviation $\sigma_{\mathrm{pred}}$",
                       "#a7471e", "--", "s"),
    }
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                         "font.size": 13, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(10.6, 5.9), facecolor="white")
    if title:
        ax.set_title(title, pad=12)
    endpoints = []
    for name, (label, color, linestyle, marker) in styles.items():
        points = sorted((float(row["coverage"]), float(row["mae"]))
                        for row in rows if row["score"] == name)
        x, y = np.asarray(points, dtype=float).T if points else ([], [])
        if (not np.array_equal(x, np.arange(1, 11) / 10)
                or not np.isfinite(y).all() or np.any(y < 0)):
            raise ValueError(f"{name} needs one finite nonnegative MAE at each 10% coverage step")
        endpoints.append(y[-1])
        ax.plot(100 * x, y, color=color, linestyle=linestyle, marker=marker,
                markerfacecolor="none" if name == "sigma_pred" else "white",
                markeredgewidth=1.5, linewidth=2.4, label=label)
    if not np.isclose(*endpoints, rtol=0, atol=1e-12):
        raise ValueError("The two scores must give the same MAE at full coverage")
    ax.annotate(f"{endpoints[0]:.5f}", (100, endpoints[0]), xytext=(-9, 8),
                textcoords="offset points", ha="right")
    ax.set(xlim=(7, 104.5), xticks=np.arange(10, 101, 10),
           xlabel="Coverage (%)", ylabel="MAE of retained predictions")
    ax.margins(y=0.13)
    ax.grid(axis="y", color="#d9dee3")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for end in ((1.01, 0), (0, 1.015)):
        ax.add_patch(FancyArrowPatch((0, 0), end, transform=ax.transAxes,
                                    arrowstyle="-|>", mutation_scale=14, color="black",
                                    linewidth=1.5, clip_on=False, shrinkA=0, shrinkB=0))
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(pad=1.5)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("risk_coverage", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="Output filename stem")
    parser.add_argument("--title")
    args = parser.parse_args()
    plot(read_rows(args.risk_coverage), args.out, args.title)


if __name__ == "__main__":
    main()
