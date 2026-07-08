"""Cut ISPRS Potsdam orthophotos into patches for the potsdam dataset loader.

Run from the project root.

    python scripts/prepare_potsdam.py --images path/to/rgb --labels path/to/labels --output data/raw/PotsdamPatches

Potsdam ships 6000 by 6000 orthophotos at 0.05 m per pixel with RGB color coded labels.
The loader in data/potsdam.py expects pre cut patches under root/<split>/images and
root/<split>/labels with matching file names. This script cuts them and assigns whole
source photos to train or val, so patches from one photo never leak across splits.

Patch size matters because of scale. LoveDA is 0.3 m per pixel, six times coarser than
Potsdam. A 2304 pixel patch resized to a 384 network input therefore lands exactly at
0.3 m per pixel, which is what a LoveDA trained model expects. Use 3072 when the model
input is 512. Download Potsdam yourself from the ISPRS benchmark site, it requires a
short registration. The RGB orthophotos and the label tiles usually follow the naming
pattern top_potsdam_2_10_RGB.tif and top_potsdam_2_10_label.tif.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from data.potsdam import POTSDAM_COLORS  # noqa: E402

Image.MAX_IMAGE_PIXELS = None  # the source orthophotos are large on purpose

SOURCE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def find_label(labels_dir, image_path):
    """Return the label file matching one orthophoto, or None when there is none.

    The usual ISPRS convention replaces RGB with label in the file name. The exact
    image name is also tried so pre renamed datasets keep working.
    """
    candidates = []
    for suffix in (image_path.suffix, ".tif", ".tiff", ".png"):
        candidates.append(image_path.stem.replace("RGB", "label") + suffix)
    candidates.append(image_path.name)
    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        candidate = labels_dir / name
        if candidate.exists():
            return candidate
    return None


def cut_pair(image_path, label_path, out_root, split, patch_size, stride):
    """Cut one orthophoto and its label into aligned patches and return the count."""
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    label = np.asarray(Image.open(label_path).convert("RGB"), dtype=np.uint8)
    if image.shape[:2] != label.shape[:2]:
        raise ValueError(
            f"{image_path.name} and {label_path.name} differ in size, "
            f"{image.shape[:2]} against {label.shape[:2]}"
        )

    image_dir = out_root / split / "images"
    label_dir = out_root / split / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    height, width = image.shape[:2]
    count = 0
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            name = f"{image_path.stem}_y{y:05d}_x{x:05d}.png"
            Image.fromarray(image[y : y + patch_size, x : x + patch_size]).save(image_dir / name)
            Image.fromarray(label[y : y + patch_size, x : x + patch_size]).save(label_dir / name)
            count += 1
    return count


def run(images_dir, labels_dir, out_root, patch_size, stride, val_fraction):
    """Pair the source photos with their labels, split them and cut every patch."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_root = Path(out_root)

    sources = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in SOURCE_EXTENSIONS
    )
    pairs = []
    for image_path in sources:
        label_path = find_label(labels_dir, image_path)
        if label_path is None:
            print(f"no label found for {image_path.name}, skipping it")
            continue
        pairs.append((image_path, label_path))
    if not pairs:
        raise FileNotFoundError(
            f"no image and label pairs found under {images_dir} and {labels_dir}"
        )

    # Whole photos are assigned to a split so no photo leaks patches into both.
    num_val = max(1, round(val_fraction * len(pairs))) if val_fraction > 0 else 0
    num_val = min(num_val, len(pairs))
    val_pairs = pairs[len(pairs) - num_val :]
    train_pairs = pairs[: len(pairs) - num_val]

    totals = {"train": 0, "val": 0}
    for split, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        for image_path, label_path in split_pairs:
            count = cut_pair(image_path, label_path, out_root, split, patch_size, stride)
            totals[split] += count
            print(f"{split}  {image_path.name}  {count} patches")

    print(
        f"Cut {totals['train']} train and {totals['val']} val patches of size "
        f"{patch_size} into {out_root}"
    )
    print(
        "Point the loader at them with "
        f"--set dataset.name=potsdam dataset.root={out_root}"
    )
    return totals


def run_selftest():
    """Cut synthetic orthophotos and check the produced layout, no real data needed."""
    root = Path("results/potsdam_prepare_selftest")
    source_images = root / "source" / "images"
    source_labels = root / "source" / "labels"
    source_images.mkdir(parents=True, exist_ok=True)
    source_labels.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    palette = np.array(POTSDAM_COLORS, dtype=np.uint8)
    for name in ("top_potsdam_2_10_RGB.png", "top_potsdam_3_11_RGB.png"):
        image = rng.integers(0, 255, size=(300, 300, 3), dtype=np.uint8)
        classes = rng.integers(0, len(palette), size=(10, 10))
        label = palette[np.repeat(np.repeat(classes, 30, axis=0), 30, axis=1)]
        Image.fromarray(image).save(source_images / name)
        Image.fromarray(label).save(source_labels / name.replace("RGB", "label"))

    out_root = root / "patches"
    totals = run(source_images, source_labels, out_root, patch_size=128, stride=128, val_fraction=0.5)
    for split in ("train", "val"):
        images = sorted((out_root / split / "images").glob("*.png"))
        labels = sorted((out_root / split / "labels").glob("*.png"))
        assert images and len(images) == len(labels)
        assert [p.name for p in images] == [p.name for p in labels]
        with Image.open(images[0]) as patch:
            assert patch.size == (128, 128)
    assert totals["train"] == 4 and totals["val"] == 4
    print(f"selftest wrote patches under {out_root}")
    print("selftest OK")


def main():
    parser = argparse.ArgumentParser(
        description="Cut Potsdam orthophotos and labels into loader ready patches",
        epilog=(
            "Potsdam is 0.05 m per pixel. A 2304 pixel patch resized to a 384 network "
            "input lands at 0.3 m per pixel, matching LoveDA. Use 3072 for a 512 input."
        ),
    )
    parser.add_argument("--images", default=None, help="folder holding the RGB orthophotos")
    parser.add_argument("--labels", default=None, help="folder holding the RGB label tiles")
    parser.add_argument(
        "--output", default="data/raw/PotsdamPatches", help="root folder for the patches"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=2304,
        help="patch side in source pixels, 2304 matches a 384 input at 0.3 m",
    )
    parser.add_argument(
        "--stride", type=int, default=None, help="step between patches, defaults to the patch size"
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="fraction of source photos assigned to the val split",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="cut synthetic photos and verify the layout, no data needed",
    )
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
        return

    if args.images is None or args.labels is None:
        raise SystemExit("--images and --labels are required unless --selftest is used")

    run(
        args.images,
        args.labels,
        args.output,
        int(args.patch_size),
        int(args.stride or args.patch_size),
        float(args.val_fraction),
    )


if __name__ == "__main__":
    main()
