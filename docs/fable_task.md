# Fable task, semantic segmentation failure analysis repo

You are working in the repo `semantic-segmentation-feature` (PyTorch land cover semantic segmentation plus failure analysis, DeepLabV3 ResNet50, trained on LoveDA which is China only, 0.3 m aerial, 7 classes background building road water barren forest agriculture). The trained baseline runs on a Colab T4. There are three problems to fix. Read the whole task before you start.

## Hard writing rule, applies to every file you touch

In all prose you write (docstrings, comments, markdown, generated reports, help text, gradio labels) do NOT use em-dashes and do NOT use colons. Restructure sentences instead. Use "such as", "namely", or split into two sentences. YAML and Python syntax that require a colon are code, not prose, so they are exempt. Docstrings are plain sentences with no "Args" or "Returns" colon sections, matching the existing style in this repo. Match the surrounding code style, comment density, and naming.

## General constraints

- Keep everything device agnostic. Use `utils/device.get_device`. It must run on CPU and on a CUDA GPU with no code change, same as the current code.
- Preserve the existing registry pattern for models, losses, datasets. Do not break the dummy or LoveDA flows.
- Reuse existing helpers instead of duplicating them. `analysis/error_maps.colorize` and `get_palette` for colored maps, `utils/plots.save_bar_chart` and `save_confusion_heatmap` for figures, `utils/config` for config load and dotted overrides, `utils/seed.set_seed`.
- Config comes from the checkpoint by default, same convention as `scripts/evaluate.py` and `scripts/analyze.py`.
- Do not add heavy new dependencies. numpy, torch, torchvision, PIL, matplotlib, pandas are already available. scipy is acceptable if genuinely needed, otherwise avoid it.
- On Colab the torch build is the CUDA build. Never reinstall torch or torchvision. The notebook already installs only the light extras.

---

## Problem 1, training takes about 4 hours on a T4, it should take about 2

Root cause. The Colab notebook `notebooks/colab_loveda.ipynb` train cell runs

```
!python scripts/train.py --config configs/default.yaml --set train.epochs=15 dataloader.batch_size=8 dataloader.num_workers=2
```

but it never enables the two knobs that make a T4 fast. `configs/default.yaml` sets `amp: false` (line 43) and `image_size: [512, 512]` (line 12). So the run trains at full 512 with mixed precision off, which is about 22 minutes per epoch, roughly 5 to 6 hours for 15 epochs. The known good T4 combination is AMP on at image size 384, which is about 8 minutes per epoch and gives an equivalent baseline (validated earlier, mIoU about 0.50 on LoveDA val). The notebook markdown even claims each epoch takes a few minutes, which only holds with these knobs on, so the text and the cell currently disagree.

Fix.

1. In `notebooks/colab_loveda.ipynb`, change the train cell (`cell-11`) to add the speed knobs.

```
!python scripts/train.py --config configs/default.yaml --set train.epochs=15 dataloader.batch_size=8 dataloader.num_workers=2 train.amp=true dataset.image_size=[384,384]
```

2. Update the surrounding notebook markdown (`cell-10`) so the wording matches what the cell actually does. State that AMP on at 384 gives about 8 minutes per epoch on a T4 and that raising to 512 with AMP off is much slower.

3. In `scripts/train.py` `main`, when the device is CUDA set `torch.backends.cudnn.benchmark = True` before the training loop. Input size is fixed per run, so cudnn autotune picks the fastest kernels and gives a small free speedup. Guard it so it only runs on CUDA.

4. Do not change the defaults in `configs/default.yaml`. That file stays the full resolution reference config. The speed knobs belong in the notebook override where they already conceptually live.

Acceptance for Problem 1. On a T4 an epoch at 384 with AMP on completes in well under 12 minutes and 15 epochs finish in roughly 2 to 2.5 hours. Training still writes the same checkpoints, metrics.csv, and TensorBoard logs as before.

---

## Problem 2, identify segmentation failures anywhere in the world, with no ground truth labels

Context. The model is trained only on LoveDA, which is China. It can run inference on aerial imagery from anywhere (Tokyo, Cairo, Iowa, the Sahara), but two things make it fail off distribution, and those are exactly what we want to surface. One, domain gap, the model was never trained on those geographies so it is often confidently wrong. Two, the class set is locked to the 7 LoveDA classes, so a scene that contains something outside those 7 gets forced into one of them.

The existing failure analysis in `scripts/analyze.py` compares prediction against a ground truth mask, so it only works on LoveDA val. Global imagery has no labels, so we need label free failure detection. Build the following.

### 2a. New module `analysis/uncertainty.py`, label free pixel uncertainty maps

Pure functions that take model logits of shape `(B, C, H, W)` and return per pixel maps in `[0, 1]`, no labels required. Reuse these from both the app and the batch scanner so the logic lives in one place.

- `max_softmax_confidence(logits)`, the current max softmax probability. This already exists inline in `scripts/app.py._infer`, move the logic here and have the app call it.
- `predictive_entropy(logits)`, Shannon entropy of the softmax over classes, normalized by `log(num_classes)` so it lands in `[0, 1]`. High entropy means the model is spread across classes.
- `margin(logits)`, the difference between the top 1 and top 2 softmax probabilities. Low margin means two classes are competing, which flags ambiguous pixels. Return it so that higher value means more failure risk, namely return `1 - (p1 - p2)`.
- `failure_score(logits, weights=...)`, a single combined per pixel failure heatmap in `[0, 1]` that blends normalized entropy, the margin risk, and `1 - confidence`. Document the default weights in the docstring and make them overridable. Higher means more likely wrong.

Keep these vectorized in torch, device agnostic, and covered by a tiny smoke check (run on a random logits tensor and assert output shapes and value range).

### 2b. New module `analysis/ood.py`, out of distribution novelty score against the LoveDA training distribution

The uncertainty maps above catch pixel ambiguity, but a model can be confidently and uniformly wrong on a truly novel scene. Add a distribution level novelty score that answers "how far is this image from what the model trained on".

Default approach, feature space Mahalanobis distance.

- Extract a backbone feature vector per image. For `deeplabv3_resnet50` the torchvision backbone output is a 2048 channel feature map. Register a forward hook or call the backbone directly, then global average pool over height and width to get a 2048 vector per image.
- Fit a reference Gaussian on a sample of LoveDA train images, namely a mean vector and a covariance matrix with shrinkage regularization for stability. Cap the number of sampled training images for speed (for example 400) and make it a CLI arg.
- Novelty score for a new image is the Mahalanobis distance of its feature vector to that Gaussian. Also store a set of training distances so the raw distance can be normalized to a 0 to 1 novelty by comparison against the training percentiles. Higher means more out of distribution.
- Save the reference (mean, inverse covariance, training distance percentiles, plus a cheap RGB channel mean and std of the training tiles) to a file next to the checkpoint, for example `results/loveda_deeplabv3/ood_reference.pt`. Load it lazily and degrade gracefully when it is missing.

Keep it dependency light. numpy and torch are enough. Only reach for scipy if you must.

### 2c. New script `scripts/fit_ood.py`, build the reference once

CLI mirrors the other scripts. `--checkpoint` required, `--config` optional falling back to the checkpoint config, `--num-images` to cap the training sample, `--output` defaulting to `ood_reference.pt` next to the checkpoint. It loads the model, runs the LoveDA train split through the backbone, fits the Gaussian and percentiles from 2b, and saves the reference. Print a short summary. Add a `--selftest` style fast path or support `dataset.subset` so it can be smoke tested without the full dataset.

### 2d. New script `scripts/scan_failures.py`, run label free failure detection on any imagery

This is the main worldwide tool. It takes a folder of arbitrary aerial images (or a single image) with no labels and produces failure visuals and a ranked report.

- CLI. `--checkpoint` required, `--images` a file or a directory, `--output-dir`, `--ood-reference` defaulting to the file next to the checkpoint, `--tile-size` and `--overlap` for large images, `--device`.
- Scale handling. LoveDA is 0.3 m ground sample distance. A large orthophoto squashed to 384 destroys that scale and creates fake failure. So run a sliding window at the model input size across the native image, overlap the windows, and stitch the prediction and the uncertainty maps back into a full resolution result. For an image already near tile size, one window is fine. Document the ground sample distance assumption in the help text and the report.
- For each image produce and save panels, namely the input, the prediction overlay using `colorize` and `get_palette`, the combined `failure_score` heatmap, the entropy map, and a per tile novelty heatmap from the OOD score. Reuse the magma style heatmap rendering already in `scripts/app.py`.
- Produce a ranked table (CSV and JSON) of the worst images or tiles sorted by mean failure score and by novelty, so the user can see where in the world the model breaks worst.
- Write a short `report.md` that summarizes the run, the ground sample distance caveat, the top failing images, and what the entropy, margin, and novelty signals mean. Follow the no colon no em-dash rule. Optionally reuse `scripts/gallery.py` to assemble a single gallery.html.
- No ground truth anywhere in this script.
- Add a `--selftest` path that runs the pipeline on a couple of random noise images and exits, so it can be verified with no data and no GPU, matching the `--selftest` convention already in `scripts/app.py`.

### 2e. Extend `scripts/app.py`, richer label free view when no ground truth is uploaded

Today the no ground truth branch shows only a max softmax confidence map. Upgrade it.

- When no ground truth is uploaded, show the entropy map, the margin risk map, the combined `failure_score` heatmap, and a single scalar novelty score with a plain language verdict such as low, medium, or high novelty relative to the LoveDA training data. Load `ood_reference.pt` if it exists next to the checkpoint, and if it is absent skip the novelty score but still show the uncertainty maps.
- Add a note in the app header that the model was trained on China at 0.3 m, so imagery at a very different scale or geography will degrade, and that the failure maps show where the model is least trustworthy.
- Keep the existing ground truth path unchanged. When a label is uploaded it still scores accuracy and mean IoU as before.
- Reuse the new `analysis/uncertainty.py` functions so the app and the scanner share one implementation.

### 2f. Notebook and docs

- Add notebook cells after analysis that run `scripts/fit_ood.py` on the trained checkpoint and then `scripts/scan_failures.py` on a small set of sample non China tiles, then display the failure heatmaps and the ranked worst list inline. If you cannot bundle sample tiles, add a clearly marked cell where the user drops in their own images or a download of a few public sample tiles, and explain in markdown that any aerial image from anywhere works because this path needs no labels.
- Update `README.md` with a short section on label free worldwide failure detection describing `fit_ood.py`, `scan_failures.py`, and the new app panels. Keep to the no colon no em-dash rule.

## Acceptance for Problem 2

- `scripts/fit_ood.py` produces an `ood_reference.pt` next to the checkpoint.
- `scripts/scan_failures.py --selftest` runs end to end on random images with no data and no GPU and prints panel and ranking summaries.
- `scripts/scan_failures.py` on a folder of real non China aerial images produces per image failure heatmaps, entropy maps, novelty maps, and a ranked CSV and JSON of the worst images, with no labels and no crash.
- `scripts/app.py` with no uploaded ground truth shows entropy, margin, combined failure heatmap, and a novelty score, and still degrades cleanly when `ood_reference.pt` is absent.
- `analysis/uncertainty.py` functions return maps in `[0, 1]` with correct shapes, verified by a small smoke check.
- Existing dummy and LoveDA train, evaluate, analyze flows still work unchanged.
- Every new or edited piece of prose obeys the no em-dash no colon rule.

---

## Problem 3, prove the label free signals actually predict true error under domain shift

Context. Problem 2 builds signals that claim to find failures without labels. That claim is the scientific heart of the project and it must be tested, not assumed. The test needs labeled data from a different geography than the training data. The repo already ships a loader for ISPRS Potsdam (German urban imagery, 6 classes, 0.05 m ground sample distance), which a LoveDA trained model has never seen, so it serves as the geographic domain shift set. LoveDA val serves as the in domain control.

### 3a. New module `analysis/validation.py`, error masks, ranking metrics and the class mapping

- `POTSDAM_TO_LOVEDA`, a coarse mapping from the six Potsdam class ids onto the seven LoveDA class ids. Building maps to building, tree maps to forest, impervious surface maps to road, low vegetation maps to background (LoveDA annotates grass as background, its agriculture class means cropland), and car and clutter map to the ignore index because they have no counterpart. Document in prose that the mapping is coarse, that absolute error rates on Potsdam are therefore approximate, and that the ranking metrics only need the error mask to be roughly right.
- `remap_mask(mask, mapping, ignore_index)` rewriting ground truth ids through the mapping, unmapped ids become ignore.
- `pixel_error_mask(pred, gt, ignore_index)` returning boolean error and valid masks.
- `error_auroc(scores, errors)` and `error_aupr(scores, errors)`, thin wrappers over sklearn returning None in the degenerate all correct or all wrong case. sklearn is already pinned in requirements, this adds no dependency.
- `risk_coverage(scores, errors, num_points)` returning coverage fractions, selective risks and the area under the curve, keeping pixels from the lowest failure score upward.
- A `__main__` smoke check on synthetic scores where an informative signal must beat a random one.

### 3b. New script `scripts/validate_signals.py`, the experiment

- CLI mirrors the other scripts. `--checkpoint` required, `--config` and `--set` with the checkpoint fallback, `--split` default val, `--label-map` with choices none and potsdam_to_loveda, `--limit` to cap images, `--max-pixels-per-image` (seeded subsample for the pixel metrics, default 20000), `--ood-reference` defaulting to the file next to the checkpoint, `--baseline-csv` pointing at the per_image.csv of a previous in domain run, `--output-dir` defaulting to `validation/<dataset>_<split>` under the run dir, `--device`, `--selftest`.
- One pass over the labeled split. Per image compute the four uncertainty signals from `analysis/uncertainty.py`, the prediction, the optionally remapped ground truth, the error mask, a subsample of valid pixels, and the per image novelty when the reference exists. Reuse the backbone hook so novelty needs no second forward pass.
- Report per signal the pixel level AUROC, AUPR and risk coverage area, the per image Spearman correlation between mean signal and true error rate, the novelty Spearman, and, when `--baseline-csv` is given, the AUROC of novelty at separating the two datasets.
- Write `metrics.json`, `per_image.csv`, a risk coverage figure, a failure score decile bar chart, a mean failure against error scatter, a novelty histogram when the baseline is given, and a `report.md` that explains every number in plain language. Follow the no colon no em-dash rule.
- `--selftest` runs the whole thing on the dummy dataset with a randomly initialized model, no data and no GPU needed.
- The intended experiment is two runs. First LoveDA val for the in domain check, then Potsdam patches with `--label-map potsdam_to_loveda --baseline-csv <loveda run>/per_image.csv`. If the ranking metrics stay clearly above chance on Potsdam, the label free signals are validated for worldwide use.

### 3c. New script `scripts/prepare_potsdam.py`, cut the orthophotos at the right scale

Potsdam ships 6000 by 6000 orthophotos at 0.05 m per pixel, six times finer than LoveDA. Feeding them naively creates a scale confound. The script cuts image and label pairs into aligned patches in the layout `data/potsdam.py` expects, assigns whole source photos to train or val so no photo leaks across splits, and defaults the patch size to 2304 because a 2304 patch resized to a 384 input lands exactly at 0.3 m per pixel. Document that 3072 matches a 512 input. `--images`, `--labels`, `--output`, `--patch-size`, `--stride`, `--val-fraction`, plus a `--selftest` that cuts synthetic photos and checks the layout. The user downloads Potsdam themselves, it requires registration.

### 3d. Docs for Problem 3

- Notebook cells after the scan section that run `validate_signals.py` on LoveDA val with a small `--limit`, display the metrics and figures inline, and describe the optional Potsdam cross domain recipe in markdown.
- README subsection describing the validation and the two run recipe.

## Acceptance for Problem 3

- `scripts/validate_signals.py --selftest` runs end to end with no data and no GPU.
- On LoveDA val the script produces metrics.json, per_image.csv, the figures and report.md, and the failure score AUROC lands clearly above 0.5.
- With prepared Potsdam patches, `--label-map potsdam_to_loveda` runs without a crash and reports the same metrics plus the novelty domain AUROC when a baseline csv is given.
- `scripts/prepare_potsdam.py --selftest` verifies the cutting and the layout with no real data.

## Suggested order

1. Problem 1 speed fix, small and self contained, do it first and confirm the notebook wording matches.
2. `analysis/uncertainty.py`, then wire it into `scripts/app.py` no ground truth branch.
3. `analysis/ood.py` and `scripts/fit_ood.py`.
4. `scripts/scan_failures.py` with tiling, the ranked report, and the `--selftest` path.
5. `analysis/validation.py`, `scripts/validate_signals.py` and `scripts/prepare_potsdam.py`.
6. Notebook cells and README sections.
