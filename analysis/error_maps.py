"""Per image error maps.

For the worst scoring images this module saves a four panel figure showing the input, the
ground truth, the prediction and an error overlay marking every pixel where the prediction
differs from the ground truth. Looking at the actual failures is often the fastest way to form
a hypothesis about what the model struggles with.

To keep memory bounded on large datasets the WorstImageCollector keeps only the current worst
N samples while a single pass streams through the data, so there is no need to store every
prediction.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def get_palette(num_classes):
    """Return an array of RGB colors, one per class, drawn from a categorical colormap."""
    cmap = plt.get_cmap("tab20", num_classes)
    return (np.array([cmap(i)[:3] for i in range(num_classes)]) * 255).astype(np.uint8)


def colorize(mask, num_classes, palette, ignore_index, ignore_color=(0, 0, 0)):
    """Map a class index mask to an RGB image using the palette."""
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_index in range(num_classes):
        rgb[mask == class_index] = palette[class_index]
    rgb[mask == ignore_index] = np.array(ignore_color, dtype=np.uint8)
    return rgb


class WorstImageCollector:
    """Keep the worst N samples by score while streaming over a dataset."""

    def __init__(self, num_keep):
        self.num_keep = int(num_keep)
        self.items = []

    def consider(self, score, index, image, gt, pred):
        """Offer one sample, keeping it only if it is among the worst seen so far."""
        if self.num_keep <= 0:
            return
        entry = {"score": score, "index": index, "image": image, "gt": gt, "pred": pred}
        if len(self.items) < self.num_keep:
            self.items.append(entry)
            return
        worst_kept = max(range(len(self.items)), key=lambda k: self.items[k]["score"])
        if score < self.items[worst_kept]["score"]:
            self.items[worst_kept] = entry

    def sorted_items(self):
        """Return the kept samples ordered from worst to best score."""
        return sorted(self.items, key=lambda entry: entry["score"])


def render_error_map(entry, num_classes, ignore_index, palette, out_path):
    """Save the four panel input, ground truth, prediction and error overlay figure."""
    image = entry["image"]
    gt = entry["gt"]
    pred = entry["pred"]
    valid = gt != ignore_index
    error = valid & (pred != gt)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    axes[0].imshow(np.clip(image, 0.0, 1.0))
    axes[0].set_title("input")
    axes[1].imshow(colorize(gt, num_classes, palette, ignore_index))
    axes[1].set_title("ground truth")
    axes[2].imshow(colorize(pred, num_classes, palette, ignore_index))
    axes[2].set_title("prediction")
    axes[3].imshow(np.clip(image, 0.0, 1.0))
    overlay = np.zeros((error.shape[0], error.shape[1], 4), dtype=np.float64)
    overlay[error] = [1.0, 0.0, 0.0, 0.75]
    axes[3].imshow(overlay)
    error_fraction = float(error.sum()) / max(int(valid.sum()), 1)
    axes[3].set_title(f"errors  mIoU {entry['score']:.3f}  error frac {error_fraction:.2f}")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
