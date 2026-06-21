"""Class size analysis.

This groups classes by how much of the dataset they cover and reports accuracy per group, so
it becomes clear whether the model systematically underperforms on rare classes. Classes are
ordered by ground truth pixel support and split into a configurable number of bins, from the
most frequent group to the rarest group.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _bin_label(position, count):
    """Return a readable label for a bin given its position and the number of bins."""
    if count == 1:
        return "all classes"
    if position == 0:
        return "most frequent"
    if position == count - 1:
        return "rarest"
    return f"middle {position}"


def analyze_class_size(confusion, class_names, num_bins=3):
    """Group classes by support and return per bin mean IoU and mean recall."""
    support = confusion.mat.sum(axis=1).astype(np.int64)
    iou = confusion.iou_per_class()
    _, recall, _ = confusion.precision_recall_f1()

    order = np.argsort(-support)
    groups = np.array_split(order, min(int(num_bins), len(order)))

    bins = []
    for position, group in enumerate(groups):
        members = [int(i) for i in group]
        bins.append(
            {
                "bin": _bin_label(position, len(groups)),
                "class_ids": members,
                "class_names": [class_names[i] for i in members],
                "total_support": int(support[members].sum()),
                "mean_iou": _safe_mean(iou[members]),
                "mean_recall": _safe_mean(recall[members]),
            }
        )
    return {"num_bins": len(groups), "bins": bins}


def _safe_mean(values):
    """Return the mean of the finite entries or None when none are finite."""
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else None


def save_class_size_figure(result, out_path):
    """Save a grouped bar chart of mean IoU and mean recall per bin."""
    labels = [b["bin"] for b in result["bins"]]
    mean_iou = [0.0 if b["mean_iou"] is None else b["mean_iou"] for b in result["bins"]]
    mean_recall = [
        0.0 if b["mean_recall"] is None else b["mean_recall"] for b in result["bins"]
    ]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2), 4))
    ax.bar(x - width / 2, mean_iou, width, label="mean IoU", color="steelblue")
    ax.bar(x + width / 2, mean_recall, width, label="mean recall", color="indianred")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Accuracy by class frequency bin")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
