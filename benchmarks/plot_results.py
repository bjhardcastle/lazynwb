"""Plot autoresearch iteration results for get_table_schema speedup."""
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

results_path = pathlib.Path(__file__).parent / "autoresearch_results.tsv"

rows = []
with open(results_path) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        rows.append(row)

iterations = [int(r["iteration"]) for r in rows]
times = [float(r["time_s"]) for r in rows]
descriptions = [r["description"] for r in rows]
keeps = [r["keep"] for r in rows]

colors = []
for r in rows:
    if r["keep"] == "baseline":
        colors.append("#888888")
    elif r["keep"] == "keep":
        colors.append("#2ecc71")
    else:
        colors.append("#e74c3c")

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(iterations, times, color=colors, edgecolor="white", linewidth=0.5, width=0.6)

# Annotate bars
for i, (bar, t, desc) in enumerate(zip(bars, times, descriptions)):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{t:.2f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(bar.get_x() + bar.get_width() / 2, -0.6,
            f"Iter {iterations[i]}", ha="center", va="top", fontsize=8, color="#555")

# Baseline reference line
baseline = times[0]
ax.axhline(baseline, color="#888888", linestyle="--", linewidth=1, alpha=0.6, label=f"Baseline {baseline:.2f}s")

ax.set_xlabel("Iteration", fontsize=11)
ax.set_ylabel("Time (seconds)", fontsize=11)
ax.set_title("get_table_schema() — autoresearch speedup iterations\n(2 S3 HDF5 files, 69 columns)", fontsize=12)
ax.set_xticks(iterations)
ax.set_ylim(0, baseline * 1.2)

legend_patches = [
    mpatches.Patch(color="#888888", label="Baseline"),
    mpatches.Patch(color="#2ecc71", label="Keep (improvement)"),
    mpatches.Patch(color="#e74c3c", label="Discard (regression)"),
]
ax.legend(handles=legend_patches, loc="upper right", fontsize=9)

# Wrap long descriptions for x-tick labels
import textwrap
labels = ["\n".join(textwrap.wrap(d, 20)) for d in descriptions]
ax.set_xticklabels(labels, fontsize=7, ha="center")

plt.tight_layout()
out = pathlib.Path(__file__).parent / "autoresearch_results.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved plot to {out}")
