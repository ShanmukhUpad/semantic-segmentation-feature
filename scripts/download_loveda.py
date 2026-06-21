"""Download and arrange the LoveDA dataset.

This fetches the LoveDA zip files from the official Zenodo record and extracts them so the
result matches the layout the loader expects, namely a root holding Train, Val and Test, each
with Rural and Urban subfolders that contain images_png and masks_png. Train and Val carry the
labels. The Test split has images only because its labels are withheld for the public
benchmark, so train on Train and evaluate on Val.

    python scripts/download_loveda.py --root data/raw/LoveDA

The files are large, a few gigabytes in total, so the download takes a while. Already present
splits are skipped.
"""
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

ZENODO = "https://zenodo.org/records/5706578/files"
ZIPS = {
    "Train": f"{ZENODO}/Train.zip?download=1",
    "Val": f"{ZENODO}/Val.zip?download=1",
    "Test": f"{ZENODO}/Test.zip?download=1",
}


def download(url, dest):
    """Stream a URL to a destination file showing a progress bar."""
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        with open(dest, "wb") as handle, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
                bar.update(len(chunk))


def main():
    parser = argparse.ArgumentParser(description="Download the LoveDA dataset")
    parser.add_argument("--root", default="data/raw/LoveDA", help="where to place the data")
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="also download the Test split, which has images but no labels",
    )
    parser.add_argument(
        "--keep-zips", action="store_true", help="keep the downloaded zip files"
    )
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    downloads = root / "_downloads"
    downloads.mkdir(exist_ok=True)

    wanted = ["Train", "Val"] + (["Test"] if args.include_test else [])
    for split in wanted:
        if (root / split).exists():
            print(f"{split} already present at {root / split}, skipping")
            continue
        zip_path = downloads / f"{split}.zip"
        if not zip_path.exists():
            print(f"Downloading {split}")
            download(ZIPS[split], zip_path)
        print(f"Extracting {split}")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(root)
        if not args.keep_zips:
            zip_path.unlink()

    if not args.keep_zips:
        try:
            downloads.rmdir()
        except OSError:
            pass

    print("\nDone. Set dataset.root in your config to this path.")
    print(f"  dataset.root: {root}")
    expected = root / "Train" / "Rural" / "images_png"
    if expected.exists():
        print(f"Verified layout, found {expected}")
    else:
        print(
            "Warning, expected layout not found. Check that the zip extracted to "
            f"{root}/Train/Rural/images_png. The extracted folder names may differ."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
