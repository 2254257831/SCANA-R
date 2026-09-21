from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "scana_experiments"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    report = json.loads((OUT / "noise_generator_report.json").read_text(encoding="utf-8"))
    generator = read_csv(OUT / "exp422_generator_validation.csv")
    alpha = read_csv(OUT / "exp421_constraint_curve.csv")
    ablation = read_csv(OUT / "exp421_projection_ablation.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0), dpi=180)
    fig.patch.set_facecolor("#fbfcfe")
    colors = {
        "Clean-DT-zero-delta": "#64748b",
        "Manual-Gaussian": "#dc2626",
        "Real-statistic": "#0891b2",
        "SCANA-Generator": "#16a34a",
    }

    ax = axes[0, 0]
    epochs = [x["epoch"] for x in report["loss_trace"]]
    train_loss = [x["train_loss"] for x in report["loss_trace"]]
    val_loss = [x["val_loss"] for x in report["loss_trace"]]
    ax.plot(epochs, train_loss, marker="o", lw=2.2, color="#2563eb", label="train")
    ax.plot(epochs, val_loss, marker="o", lw=2.2, color="#f97316", label="validation")
    ax.set_title("A. Conditional noise generator training", loc="left", fontweight="bold")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    names = [r["method"] for r in generator]
    mmd = [f(r, "mmd_to_real_delta") for r in generator]
    bar_colors = [colors.get(n, "#16a34a") for n in names]
    ax.barh(np.arange(len(names)), mmd, color=bar_colors, alpha=0.88)
    ax.set_yticks(np.arange(len(names)), labels=names)
    ax.invert_yaxis()
    ax.set_xlabel("MMD to real-success residual")
    ax.set_title("B. Distribution alignment before projection", loc="left", fontweight="bold")
    for i, v in enumerate(mmd):
        ax.text(v + max(mmd) * 0.02, i, f"{v:.3f}", va="center", fontsize=8)
    ax.grid(True, axis="x", alpha=0.25)

    ax = axes[1, 0]
    a = [f(r, "alpha") for r in alpha]
    m = [f(r, "mmd_to_real_delta") for r in alpha]
    rv = [f(r, "raw_range_violation_rate") * 100.0 for r in alpha]
    ax.plot(a, m, color="#16a34a", lw=2.4, marker="o", label="MMD")
    ax.set_xlabel("noise scale alpha")
    ax.set_ylabel("MMD", color="#16a34a")
    ax.tick_params(axis="y", labelcolor="#16a34a")
    ax2 = ax.twinx()
    ax2.plot(a, rv, color="#dc2626", lw=2.0, marker="s", label="raw violation")
    ax2.set_ylabel("raw range violation (%)", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")
    ax.set_title("C. Noise scale trade-off after projection", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ab_names = [r["method"] for r in ablation]
    x = np.arange(len(ab_names))
    dir_drop = [1.0 - f(r, "direction_consistency") for r in ablation]
    violation = [f(r, "range_violation_against_full_constraints") for r in ablation]
    width = 0.38
    ax.bar(x - width / 2, dir_drop, width, color="#f97316", label="1 - direction consistency")
    ax.bar(x + width / 2, violation, width, color="#dc2626", label="range violation")
    ax.set_xticks(x, labels=ab_names, rotation=22, ha="right")
    ax.set_title("D. Projection ablation risk signals", loc="left", fontweight="bold")
    ax.set_ylabel("rate")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("SCANA offline experiments: generator calibration and label-validity trade-offs", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = OUT / "fig_scana_offline_results_summary.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(out_png)


if __name__ == "__main__":
    main()
