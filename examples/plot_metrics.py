"""Distribution plots of a cohort's metrics table.

    python examples/plot_metrics.py OUT_DIR/metrics.csv [more.csv ...] -o plots/

Several tables (e.g. batches or sites) are combined and coloured by source.
Needs matplotlib (pip install openheart[plot]).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PANELS = [
    ("lv_edv_ml", "LV end-diastolic volume (ml)"),
    ("lv_esv_ml", "LV end-systolic volume (ml)"),
    ("lv_ef_pct", "LV ejection fraction (%)"),
    ("rv_ef_pct", "RV ejection fraction (%)"),
    ("lv_mass_g", "LV mass (g)"),
    ("wt_mean_ed_mm", "Mean wall thickness, ED (mm)"),
    ("wt_max_ed_mm", "Max wall thickness, ED (mm)"),
    ("wt_mean_es_mm", "Mean wall thickness, ES (mm)"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("tables", nargs="+")
    p.add_argument("-o", "--out", default="plots")
    args = p.parse_args()

    frames = [pd.read_csv(t).assign(source=Path(t).parent.name) for t in args.tables]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["status"] == "ok"]
    panels = [(c, title) for c, title in PANELS if c in df]

    ncols = 4
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)
    for ax, (column, title) in zip(axes.flat, panels):
        for source, group in df.groupby("source"):
            ax.hist(group[column].dropna(), bins=30, alpha=0.6, label=f"{source} (n={len(group)})")
        ax.set_title(title, fontsize=10)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(args.out) / "metrics_distributions.png", dpi=200)
    print(df[[c for c, _ in panels]].describe().round(1).to_string())


if __name__ == "__main__":
    main()
