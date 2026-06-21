# Semantic Segmentation Failure Analysis

Research codebase for training a standard semantic segmentation model on land cover remote
sensing imagery and then analyzing where and why the model fails. The aim is to go beyond a
single overall accuracy number and characterize failure modes such as boundary errors and
weak performance on rare classes.

## Status

The project is built in phases. Phase 1 sets up the structure and Phase 2 provides the data
pipeline with a synthetic dummy dataset so the pipeline can be verified before real data is
available. Later phases add the model and training loop, per class evaluation, and the
failure analysis tools.

## Project layout

```
data/      dataset loaders, augmentation, exploration
models/    model definitions (added in a later phase)
analysis/  failure analysis utilities (added in a later phase)
configs/   YAML configuration files
scripts/   train, evaluate and analyze entry points (added in later phases)
utils/     config loading, seeding and device helpers
notebooks/ exploration notebooks
results/   outputs, checkpoints and figures (gitignored)
```

## Setup

Use Python 3.10 or newer. Create a virtual environment and install the dependencies.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The pinned torch build is the CPU build. For a CUDA machine install the matching CUDA build of
torch and torchvision from the official PyTorch index before installing the rest.

## Quick start with the dummy dataset

No download is needed for the dummy dataset. Run the data exploration utility to confirm the
pipeline works end to end.

```
python -m data.explore --config configs/dummy.yaml
```

This prints the per class pixel distribution, prints the image and mask tensor shapes and
dtypes, and saves a sample grid figure together with a class distribution CSV under the run
output directory inside results.

## Running the full pipeline

The four stages all read one config and write under the run output directory.

```
python scripts/train.py --config configs/dummy.yaml
python scripts/evaluate.py --checkpoint results/dummy_run/checkpoints/best.pth --split val
python scripts/evaluate.py --checkpoint results/dummy_run/checkpoints/best.pth --split test
python scripts/analyze.py --checkpoint results/dummy_run/checkpoints/best.pth
```

Evaluation writes one folder per split, so val and test scores live side by side under
results and never overwrite each other.

## Viewing results

There are three ways to look at a run.

One. Build a single HTML gallery that shows every figure and every metric table, including the
test scores, on one page.

```
python scripts/gallery.py --run-dir results/dummy_run
```

Open the resulting results/dummy_run/gallery.html in a browser. Add --embed to inline the
images so the file can be shared on its own.

Two. Open the markdown report results/dummy_run/analysis/report.md in a viewer that renders
images, for example the VS Code preview.

Three. Launch TensorBoard for the training curves and the sample prediction images that are
logged each epoch.

```
tensorboard --logdir results/dummy_run/tensorboard
```

The Images tab shows the input next to the ground truth and the prediction so the prediction
can be watched improving over epochs. The number of logged samples is set by train.log_images
in the config.

## Using real data

Two real datasets are supported out of the box. Set dataset.name and dataset.root in a config
file to point at your local copy.

### LoveDA

Download LoveDA with the helper script, which fetches the official Zenodo files and arranges
them for you.

```
python scripts/download_loveda.py --root data/raw/LoveDA
```

This creates a root holding Train, Val and Test, each with Rural and Urban subfolders holding
images_png and masks_png. LoveDA labels use value 0 for no data and values 1 through 7 for the
seven land cover classes. Train and Val carry labels. The Test split has images only because
its labels are withheld for the public benchmark, so train on Train and evaluate on Val. The
default config already points dataset.root at data/raw/LoveDA.

Then train, evaluate and analyze the same way as the dummy run but with the LoveDA config.

```
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --checkpoint results/loveda_deeplabv3/checkpoints/best.pth --split val
python scripts/analyze.py --checkpoint results/loveda_deeplabv3/checkpoints/best.pth --split val
```

Training the full dataset on a CPU is not practical. Use a CUDA GPU, where the code runs with
no change. For a quick check that the real data path works on a CPU, train on a small subset for
a couple of epochs.

```
python scripts/train.py --config configs/default.yaml --set dataset.subset=40 train.epochs=2 dataloader.batch_size=2
```

If you have no local GPU, open notebooks/colab_loveda.ipynb in Google Colab, switch the runtime
to a GPU, and run the cells. It clones the code, downloads LoveDA, trains, analyzes and shows
the results, then saves the run to Google Drive.

### ISPRS Potsdam

Cut the Potsdam orthophotos and labels into patches ahead of time and place them in an images
folder and a labels folder per split. Labels are expected as RGB color coded PNG tiles using
the standard six class Potsdam palette.

## Adding a new dataset

Subclass BaseSegmentationDataset in the data package, implement length, read_image and
read_mask, and register the class with the register_dataset decorator. The new name then works
through the config without any other change.

## Reproducibility

Every run sets seeds for python, numpy and torch from the seed field in the config. Data
loader workers are seeded as well so shuffling and augmentation are repeatable.
