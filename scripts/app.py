"""Interactive upload app for inspecting model predictions.

Run from the project root.

    python scripts/app.py --checkpoint results/dummy_run/checkpoints/best.pth

This launches a browser GUI where an image can be uploaded and the model prediction is shown as
a colored overlay together with a confidence map. A ground truth label image can be uploaded as
an optional second input. When it is present the app also shows the true correct versus
incorrect map and reports pixel accuracy and mean IoU. Without a ground truth the app switches
to the label free failure view, namely the entropy, margin risk and combined failure score maps
plus a novelty score against the training distribution when an ood_reference.pt file sits next
to the checkpoint. Build that file once with scripts/fit_ood.py.

Note that a model trained only on the synthetic dummy data will produce meaningless predictions
on real photographs. The app becomes useful once the model is trained on real land cover data.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
from PIL import Image  # noqa: E402

from analysis import ood  # noqa: E402
from analysis.error_maps import colorize, get_palette  # noqa: E402
from analysis.uncertainty import (  # noqa: E402
    failure_score,
    margin,
    max_softmax_confidence,
    predictive_entropy,
)
from models.registry import build_model  # noqa: E402
from utils.config import Config, load_config  # noqa: E402
from utils.device import get_device  # noqa: E402
from utils.metrics import ConfusionMatrix  # noqa: E402

NEAREST = TF.InterpolationMode.NEAREST
BILINEAR = TF.InterpolationMode.BILINEAR


class Predictor:
    """Holds a loaded model and turns an uploaded image into prediction visuals."""

    def __init__(self, checkpoint_path, config_path, device_pref):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if config_path is not None:
            cfg = load_config(config_path)
        else:
            cfg = Config(checkpoint["config"])

        self.cfg = cfg
        self.device = get_device(device_pref or str(cfg.get_path("device", "auto")))
        self.num_classes = int(cfg.dataset.num_classes)
        self.ignore_index = int(cfg.dataset.get_path("ignore_index", 255))
        size = cfg.dataset.image_size
        self.image_size = (int(size[0]), int(size[1]))
        self.mean = list(cfg.augmentation.get_path("mean", [0.485, 0.456, 0.406]))
        self.std = list(cfg.augmentation.get_path("std", [0.229, 0.224, 0.225]))
        self.dataset_name = cfg.dataset.name
        self.checkpoint_path = checkpoint_path

        self.model = build_model(cfg).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        # Novelty scoring works only when a fitted reference sits next to the checkpoint.
        self.ood_reference = ood.load_reference(
            Path(checkpoint_path).parent / "ood_reference.pt"
        )

        names = self._class_names()
        self.class_names = names
        self.palette = get_palette(self.num_classes)
        self.legend = self._make_legend()

    def _class_names(self):
        from data.registry import build_dataset

        try:
            dataset = build_dataset(self.cfg, "train")
            names = getattr(dataset, "class_names", None)
        except Exception:
            names = None
        if not names:
            names = [f"class_{i}" for i in range(self.num_classes)]
        return list(names)[: self.num_classes]

    def _make_legend(self):
        fig, ax = plt.subplots(figsize=(3.2, 0.4 * self.num_classes + 0.4))
        for i, name in enumerate(self.class_names):
            ax.add_patch(plt.Rectangle((0, i), 0.8, 0.85, color=self.palette[i] / 255.0))
            ax.text(1.0, i + 0.45, f"{i}  {name}", va="center", fontsize=10)
        ax.set_xlim(0, 4)
        ax.set_ylim(0, self.num_classes)
        ax.invert_yaxis()
        ax.axis("off")
        fig.tight_layout()
        fig.canvas.draw()
        legend = np.asarray(fig.canvas.renderer.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return legend

    def _infer(self, image_rgb):
        """Return the prediction map, the label free risk maps and a novelty summary.

        The prediction and every risk map come back at the resolution of the uploaded
        image. The novelty summary is None when no fitted reference is available.
        """
        original_hw = image_rgb.shape[:2]
        tensor = torch.from_numpy(image_rgb.copy()).permute(2, 0, 1).float() / 255.0
        resized = TF.resize(tensor, list(self.image_size), interpolation=BILINEAR, antialias=True)
        normalized = TF.normalize(resized, self.mean, self.std).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(normalized)
            logits = output["out"] if isinstance(output, dict) else output
            prediction = logits.argmax(dim=1)
            maps = {
                "confidence": max_softmax_confidence(logits),
                "entropy": predictive_entropy(logits),
                "margin risk": margin(logits),
                "failure score": failure_score(logits),
            }
        prediction_full = (
            TF.resize(prediction.float().unsqueeze(1).cpu(), list(original_hw), interpolation=NEAREST)
            .squeeze(1)
            .squeeze(0)
            .long()
            .numpy()
        )
        maps_full = {
            name: (
                TF.resize(value.unsqueeze(1).cpu(), list(original_hw), interpolation=BILINEAR)
                .squeeze(1)
                .squeeze(0)
                .numpy()
            )
            for name, value in maps.items()
        }
        return prediction_full, maps_full, self._novelty(normalized)

    def _novelty(self, normalized):
        """Score one normalized batch against the reference, or return None without one."""
        if self.ood_reference is None:
            return None
        distances, novelty = ood.score_images(self.model, normalized, self.ood_reference)
        distance = float(distances[0])
        return {
            "distance": distance,
            "novelty": float(novelty[0]),
            "verdict": ood.verdict(distance, self.ood_reference),
        }

    def _read_ground_truth(self, path, target_hw):
        """Read a label image and resize it to the target size with nearest neighbor."""
        array = np.asarray(Image.open(path))
        if array.ndim == 3:
            array = array[..., 0]
        array = array.astype(np.int64)
        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).float()
        resized = TF.resize(tensor, list(target_hw), interpolation=NEAREST)
        return resized.squeeze(1).squeeze(0).long().numpy()

    @staticmethod
    def _heatmap(values):
        """Render a zero to one map with the magma colormap as an RGB image."""
        return (plt.get_cmap("magma")(np.clip(values, 0.0, 1.0))[..., :3] * 255).astype(np.uint8)

    def predict(self, image_rgb, ground_truth_path):
        """Run the model and return a gallery of result panels and a metrics summary."""
        if image_rgb is None:
            return [], "Upload an image to begin."
        image_rgb = np.asarray(image_rgb)[..., :3].astype(np.uint8)
        prediction, maps, novelty = self._infer(image_rgb)

        prediction_color = colorize(prediction, self.num_classes, self.palette, self.ignore_index)
        overlay = (0.5 * image_rgb + 0.5 * prediction_color).astype(np.uint8)

        gallery = [
            (image_rgb, "input"),
            (overlay, "prediction overlay"),
            (prediction_color, "prediction"),
            (self._heatmap(maps["confidence"]), "confidence, bright is confident"),
        ]

        if ground_truth_path:
            ground_truth = self._read_ground_truth(ground_truth_path, image_rgb.shape[:2])
            valid = ground_truth != self.ignore_index
            error = valid & (prediction != ground_truth)
            error_image = image_rgb.copy()
            error_image[error] = [255, 0, 0]

            confusion = ConfusionMatrix(self.num_classes, self.ignore_index)
            confusion.update(ground_truth, prediction)
            total = int(valid.sum())
            correct = int((valid & (prediction == ground_truth)).sum())
            accuracy = correct / total if total else 0.0

            gallery.insert(1, (colorize(ground_truth, self.num_classes, self.palette, self.ignore_index), "ground truth"))
            gallery.append((error_image, "errors in red"))
            summary = (
                f"### Scored against the uploaded ground truth\n\n"
                f"- pixel accuracy {accuracy:.4f}\n"
                f"- mean IoU {confusion.mean_iou():.4f}\n"
                f"- error fraction {(1.0 - accuracy):.4f}\n"
                f"- correct pixels {correct} of {total}"
            )
        else:
            gallery.append((self._heatmap(maps["entropy"]), "entropy, bright is uncertain"))
            gallery.append(
                (self._heatmap(maps["margin risk"]), "margin risk, bright means two classes compete")
            )
            gallery.append(
                (self._heatmap(maps["failure score"]), "failure score, bright is likely wrong")
            )
            summary_lines = [
                "### No ground truth uploaded",
                "",
                "Showing the label free failure view. Bright areas of the entropy, margin risk "
                "and failure score maps mark the pixels where the model is least trustworthy and "
                "most likely wrong. Upload a label image to get the true correct versus "
                "incorrect map and the scores.",
            ]
            if novelty is not None:
                summary_lines += [
                    "",
                    f"Novelty against the training distribution is {novelty['novelty']:.2f} on a "
                    f"zero to one scale, verdict {novelty['verdict']} (Mahalanobis distance "
                    f"{novelty['distance']:.1f}). A high verdict means the scene looks unlike the "
                    "training data, so even confident predictions deserve little trust there.",
                ]
            else:
                summary_lines += [
                    "",
                    "No ood_reference.pt was found next to the checkpoint, so the novelty score "
                    "is unavailable. Build it once with scripts/fit_ood.py to enable it.",
                ]
            summary = "\n".join(summary_lines)

        gallery.append((self.legend, "class colors"))
        return gallery, summary


def build_interface(predictor):
    import gradio as gr

    header = (
        f"# Segmentation prediction viewer\n\n"
        f"Loaded checkpoint {predictor.checkpoint_path} trained on dataset "
        f"{predictor.dataset_name} with {predictor.num_classes} classes. Upload an image to see "
        f"the prediction. Optionally upload a label image to see correct versus incorrect."
    )
    if predictor.dataset_name == "dummy":
        header += (
            "\n\nThis checkpoint is trained on the synthetic dummy data, so predictions on real "
            "photographs will not be meaningful until the model is trained on real land cover."
        )
    if predictor.dataset_name == "loveda":
        header += (
            "\n\nThe model was trained only on LoveDA imagery of China at 0.3 m per pixel, so a "
            "scene from a very different geography or ground resolution will degrade. Without a "
            "ground truth the app shows the label free failure maps, which mark where the model "
            "is least trustworthy."
        )

    with gr.Blocks(title="Segmentation prediction viewer") as demo:
        gr.Markdown(header)
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="numpy", label="image")
                gt_input = gr.Image(type="filepath", image_mode="L", label="ground truth label, optional")
                run_button = gr.Button("Run", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Markdown()
                gallery = gr.Gallery(label="results", columns=3, height="auto")
        run_button.click(predictor.predict, inputs=[image_input, gt_input], outputs=[gallery, summary])
    return demo


def main():
    parser = argparse.ArgumentParser(description="Interactive prediction viewer")
    parser.add_argument(
        "--checkpoint",
        default="results/dummy_run/checkpoints/best.pth",
        help="path to a saved checkpoint",
    )
    parser.add_argument("--config", default=None, help="optional config path override")
    parser.add_argument("--device", default=None, help="auto, cpu or cuda")
    parser.add_argument("--port", type=int, default=7860, help="local server port")
    parser.add_argument("--share", action="store_true", help="create a public gradio link")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run prediction on a random image and exit without launching the server",
    )
    args = parser.parse_args()

    predictor = Predictor(args.checkpoint, args.config, args.device)

    if args.selftest:
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, size=(180, 200, 3), dtype=np.uint8)
        gallery, summary = predictor.predict(image, None)
        print(f"selftest produced {len(gallery)} panels")
        for _, caption in gallery:
            print(" panel", caption)
        print("summary head", summary.splitlines()[0])
        print("selftest OK")
        return

    demo = build_interface(predictor)
    demo.launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
