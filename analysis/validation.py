"""Tools that test whether the label free failure signals predict true error.

The uncertainty and novelty signals are only worth trusting worldwide if they actually
rank wrong pixels above correct ones. This module holds the pieces that check that
claim on a labeled dataset, namely error masks, ranking quality metrics and a class
mapping so ISPRS Potsdam labels can be scored against a model trained on the LoveDA
classes. The full experiment lives in scripts/validate_signals.py.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# Coarse mapping from the six Potsdam classes onto the seven LoveDA class ids. Potsdam
# impervious surfaces are mostly roads and pavement, so they map to road. Low
# vegetation is lawn and grass, which LoveDA annotates as background because its
# agriculture class means cropland. Cars and clutter have no LoveDA counterpart and are
# ignored. The mapping is deliberately coarse. Absolute error rates on Potsdam are
# therefore approximate, but the ranking metrics only need the error mask to be roughly
# right, so they stay meaningful.
POTSDAM_TO_LOVEDA = {
    0: 2,  # impervious surface to road
    1: 1,  # building to building
    2: 0,  # low vegetation to background
    3: 5,  # tree to forest
    4: None,  # car has no counterpart
    5: None,  # clutter has no counterpart
}

LABEL_MAPS = {"potsdam_to_loveda": POTSDAM_TO_LOVEDA}


def remap_mask(mask, mapping, ignore_index):
    """Return the mask with source class ids rewritten through the mapping.

    Ids mapped to None and ids missing from the mapping become the ignore index, so
    unmappable classes drop out of the error computation instead of polluting it.
    """
    mask = np.asarray(mask)
    out = np.full(mask.shape, ignore_index, dtype=np.int64)
    for source, target in mapping.items():
        if target is not None:
            out[mask == source] = target
    return out


def pixel_error_mask(pred, gt, ignore_index):
    """Return boolean error and valid masks comparing a prediction to a ground truth."""
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    valid = gt != ignore_index
    error = valid & (pred != gt)
    return error, valid


def error_auroc(scores, errors):
    """Return the AUROC of the scores at ranking error pixels first, or None.

    0.5 means the signal is no better than chance and 1.0 means every wrong pixel
    scores above every correct one. None is returned when the pixels are all correct
    or all wrong, where the metric is undefined.
    """
    errors = np.asarray(errors, dtype=bool)
    if errors.all() or not errors.any():
        return None
    return float(roc_auc_score(errors, np.asarray(scores, dtype=np.float64)))


def error_aupr(scores, errors):
    """Return the average precision of the scores at ranking error pixels first, or None.

    Unlike AUROC this weighs the positive class, so it is the stricter number when
    errors are rare. The base error rate is the chance level to compare against.
    """
    errors = np.asarray(errors, dtype=bool)
    if errors.all() or not errors.any():
        return None
    return float(average_precision_score(errors, np.asarray(scores, dtype=np.float64)))


def risk_coverage(scores, errors, num_points=50):
    """Return coverage fractions, selective risks and the area under the curve.

    Pixels are kept from the lowest failure score upward, mimicking a system that
    trusts the model only where the signal says it is safe. At coverage c the risk is
    the error rate over the kept fraction. A signal that ranks errors last pushes the
    curve down, so a lower area under it is better. The area at chance level equals
    roughly the base error rate.
    """
    scores = np.asarray(scores, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    order = np.argsort(scores, kind="stable")
    cumulative_errors = np.cumsum(errors[order])
    total = len(scores)
    coverages = np.linspace(1.0 / num_points, 1.0, num_points)
    kept = np.maximum((coverages * total).astype(np.int64), 1)
    risks = cumulative_errors[kept - 1] / kept
    return coverages, risks, float(np.mean(risks))


def _smoke_check():
    """Check the metrics behave sensibly on a synthetic score and error set."""
    rng = np.random.default_rng(0)
    errors = rng.random(5000) < 0.3
    informative = np.where(errors, rng.uniform(0.5, 1.0, 5000), rng.uniform(0.0, 0.5, 5000))
    random_scores = rng.random(5000)
    assert error_auroc(informative, errors) > 0.95
    assert abs(error_auroc(random_scores, errors) - 0.5) < 0.05
    assert error_aupr(informative, errors) > 0.9
    coverages, risks, aurc_good = risk_coverage(informative, errors)
    assert coverages.shape == risks.shape
    assert risks[0] < 0.05 and abs(risks[-1] - errors.mean()) < 0.01
    _, _, aurc_random = risk_coverage(random_scores, errors)
    assert aurc_good < aurc_random
    mapped = remap_mask(np.array([[0, 4], [2, 9]]), POTSDAM_TO_LOVEDA, 255)
    assert mapped.tolist() == [[2, 255], [0, 255]]
    error, valid = pixel_error_mask(np.array([2, 1]), np.array([2, 255]), 255)
    assert error.tolist() == [False, False] and valid.tolist() == [True, False]
    print("validation smoke check OK")


if __name__ == "__main__":
    _smoke_check()
