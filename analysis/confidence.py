"""Confidence versus correctness analysis.

For every valid pixel the model gives a predicted class and a confidence, the largest softmax
probability. This module accumulates a histogram of that confidence separately for pixels the
model got right and pixels it got wrong. If the wrong pixels still carry high confidence the
model is overconfident on its mistakes, which matters for how much its outputs can be trusted.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _to_numpy(array):
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array)


class ConfidenceAnalyzer:
    """Accumulate confidence histograms for correct and incorrect pixels."""

    def __init__(self, num_bins=20, ignore_index=255):
        self.ignore_index = ignore_index
        self.edges = np.linspace(0.0, 1.0, int(num_bins) + 1)
        self.correct_counts = np.zeros(int(num_bins), dtype=np.int64)
        self.incorrect_counts = np.zeros(int(num_bins), dtype=np.int64)
        self.sum_correct = 0.0
        self.n_correct = 0
        self.sum_incorrect = 0.0
        self.n_incorrect = 0

    def update(self, gt, pred, confidence):
        """Add one image given ground truth, prediction and per pixel confidence."""
        gt = _to_numpy(gt).reshape(-1)
        pred = _to_numpy(pred).reshape(-1)
        confidence = _to_numpy(confidence).reshape(-1).astype(np.float64)
        valid = gt != self.ignore_index
        correct = valid & (pred == gt)
        incorrect = valid & (pred != gt)

        self.correct_counts += np.histogram(confidence[correct], bins=self.edges)[0]
        self.incorrect_counts += np.histogram(confidence[incorrect], bins=self.edges)[0]
        self.sum_correct += float(confidence[correct].sum())
        self.n_correct += int(correct.sum())
        self.sum_incorrect += float(confidence[incorrect].sum())
        self.n_incorrect += int(incorrect.sum())

    def results(self):
        """Return mean confidence and pixel counts for each group plus the histograms."""
        mean_correct = self.sum_correct / self.n_correct if self.n_correct else None
        mean_incorrect = (
            self.sum_incorrect / self.n_incorrect if self.n_incorrect else None
        )
        return {
            "num_bins": len(self.correct_counts),
            "bin_edges": self.edges.tolist(),
            "correct_counts": self.correct_counts.tolist(),
            "incorrect_counts": self.incorrect_counts.tolist(),
            "mean_confidence_correct": mean_correct,
            "mean_confidence_incorrect": mean_incorrect,
            "n_correct": self.n_correct,
            "n_incorrect": self.n_incorrect,
        }

    def save_figure(self, out_path):
        """Save overlaid density histograms of confidence for correct and incorrect pixels."""
        centers = (self.edges[:-1] + self.edges[1:]) / 2
        width = (self.edges[1] - self.edges[0]) * 0.9
        correct = self.correct_counts.astype(np.float64)
        incorrect = self.incorrect_counts.astype(np.float64)
        correct_density = correct / correct.sum() if correct.sum() else correct
        incorrect_density = incorrect / incorrect.sum() if incorrect.sum() else incorrect

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(centers, correct_density, width=width, alpha=0.6, label="correct", color="seagreen")
        ax.bar(
            centers,
            incorrect_density,
            width=width,
            alpha=0.6,
            label="incorrect",
            color="indianred",
        )
        ax.set_xlabel("predicted confidence")
        ax.set_ylabel("fraction of pixels in group")
        ax.set_title("Confidence on correct versus incorrect pixels")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
