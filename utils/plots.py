"""Shared plotting helpers used by the evaluation and analysis stages.

Keeping these in one place means the confusion matrix heatmap and the per class bar charts look
the same wherever they are produced.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402


def save_confusion_heatmap(matrix, class_names, out_path):
    """Save a row normalized confusion matrix as a heatmap PNG."""
    matrix = np.asarray(matrix, dtype=np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros(matrix.shape, dtype=np.float64),
        where=row_sums > 0,
    )
    size = max(6, len(class_names))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    annotate = len(class_names) <= 15
    sns.heatmap(
        normalized,
        annot=annotate,
        fmt=".2f",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "fraction of true class"},
        ax=ax,
    )
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title("Row normalized confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_bar_chart(values, class_names, title, ylabel, out_path, ylim=(0.0, 1.0)):
    """Save a per class bar chart, drawing missing values as zero height bars."""
    heights = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), 4))
    ax.bar(range(len(class_names)), heights, color="steelblue")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
