"""Plot autoresearch iteration results for get_table_schema speedup."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Results from git log + benchmark runs (2 S3 HDF5 files, units table, 69 columns)
iterations = [
    {"label": "Baseline\n(ThreadPool+\nremfile)", "time": 9.73, "color": "#d62728", "iter": 0},
    {"label": "Iter 1\nProcessPool\nmulti-file", "time": 4.57, "color": "#ff7f0e", "iter": 1},
    {"label": "Iter 2\ncolnames\nfast-path", "time": 4.18, "color": "#ff7f0e", "iter": 2},
    {"label": "Iter 4\nh5coro batch\nmetadata read", "time": 2.38, "color": "#2ca02c", "iter": 4},
    {"label": "Iter 6\nconcurrent\nh5coro+h5py", "time": 1.60, "color": "#2ca02c", "iter": 6},
    {"label": "Iter 7\nThreadPool\n(no ProcessPool)", "time": 2.16, "color": "#1f77b4", "iter": 7},
    {"label": "Iter 8a\ninspectPath\nfast-path", "time": 1.72, "color": "#1f77b4", "iter": "8a"},
    {"label": "Iter 8b\n+monkey-patch\nbool cols", "time": 1.43, "color": "#1f77b4", "iter": "8b"},
    {"label": "Iter 8c\n+REFERENCE→\nString (no h5py)", "time": 0.74, "color": "#9467bd", "iter": "8c"},
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: bar chart of iteration times
ax = axes[0]
x = np.arange(len(iterations))
times = [it["time"] for it in iterations]
colors = [it["color"] for it in iterations]
labels = [it["label"] for it in iterations]

bars = ax.bar(x, times, color=colors, edgecolor="black", linewidth=0.5, width=0.7)

# Annotate bars
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{t:.2f}s", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("Time (seconds)", fontsize=12)
ax.set_title("get_table_schema() Speed: Autoresearch Iterations\n(2 S3 HDF5 files, units table, 69 columns)", fontsize=11)
ax.set_ylim(0, 11)
ax.axhline(y=9.73, color="#d62728", linestyle="--", alpha=0.4, linewidth=1, label="Baseline 9.73s")
ax.axhline(y=0.74, color="#9467bd", linestyle="--", alpha=0.4, linewidth=1, label="Best 0.74s")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# Add improvement %
for bar, t in zip(bars, times):
    pct = (1 - t / 9.73) * 100
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
            f"-{pct:.0f}%", ha="center", va="center", fontsize=7,
            color="white" if t > 2 else "black", fontweight="bold")

# Right plot: cumulative speedup line
ax2 = axes[1]
iter_nums = [0, 1, 2, 4, 6, 7, "8a", "8b", "8c"]
iter_labels_short = ["Base", "Iter1", "Iter2", "Iter4", "Iter6", "Iter7", "Iter8a", "Iter8b", "Iter8c"]
speedups = [9.73 / t for t in times]

ax2.plot(range(len(times)), speedups, "o-", color="#2ca02c", linewidth=2.5, markersize=8, zorder=3)
ax2.fill_between(range(len(times)), 1, speedups, alpha=0.15, color="#2ca02c")

for i, (sp, t) in enumerate(zip(speedups, times)):
    ax2.annotate(f"{sp:.1f}x\n({t:.2f}s)", (i, sp),
                 textcoords="offset points", xytext=(0, 8),
                 ha="center", fontsize=7.5, color="#2ca02c", fontweight="bold")

ax2.set_xticks(range(len(times)))
ax2.set_xticklabels(iter_labels_short, fontsize=9, rotation=20)
ax2.set_ylabel("Speedup vs Baseline (×)", fontsize=12)
ax2.set_title("Cumulative Speedup Over Baseline (9.73s)\nAutoresearch/Schema Branch", fontsize=11)
ax2.set_ylim(0.5, 16)
ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax2.grid(alpha=0.3)

# Annotate technique categories
category_patches = [
    mpatches.Patch(color="#ff7f0e", label="ProcessPool + colnames fast-path"),
    mpatches.Patch(color="#2ca02c", label="h5coro batch reads + concurrent open"),
    mpatches.Patch(color="#1f77b4", label="ThreadPool + inspectPath"),
    mpatches.Patch(color="#9467bd", label="Monkey-patch + REFERENCE→String"),
]
ax2.legend(handles=category_patches, fontsize=8, loc="upper left")

plt.tight_layout()
output_path = "benchmarks/schema_autoresearch_results.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
plt.close()
