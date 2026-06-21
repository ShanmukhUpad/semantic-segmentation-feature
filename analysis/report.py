"""Markdown report assembly for the failure analysis.

This turns the structured analysis results into a single readable markdown file with tables and
embedded figures. The report is meant to be the artifact used to characterize failure modes for
the paper, so every section states what it shows in plain language.
"""
from pathlib import Path


def _fmt(value, places=4):
    """Format a number for a table, showing n a when the value is missing."""
    if value is None:
        return "n a"
    return f"{value:.{places}f}"


def _md_table(headers, rows):
    """Build a github flavored markdown table from headers and a list of row lists."""
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def write_markdown_report(out_dir, results, figures):
    """Write report.md under out_dir and return its path.

    results holds the metrics and analysis dictionaries. figures maps section names to figure
    file names that already live under out_dir, so the report links to them with relative paths.
    """
    out_dir = Path(out_dir)
    lines = []

    lines.append("# Failure Analysis Report")
    lines.append("")
    lines.append(
        f"Model {results['model']} on dataset {results['dataset']} split {results['split']}."
    )
    lines.append(f"Checkpoint path {results['checkpoint']}.")
    lines.append(f"Evaluated samples {results['num_samples']}.")
    lines.append("")

    overall = results["overall"]
    lines.append("## Overall metrics")
    lines.append("")
    lines.append(
        _md_table(
            ["metric", "value"],
            [
                ["mean IoU", _fmt(overall["mean_iou"])],
                ["pixel accuracy", _fmt(overall["pixel_accuracy"])],
                ["mean F1", _fmt(overall["mean_f1"])],
                ["macro precision", _fmt(overall["macro_precision"])],
                ["macro recall", _fmt(overall["macro_recall"])],
            ],
        )
    )
    lines.append("")

    lines.append("## Per class metrics")
    lines.append("")
    per_class_rows = [
        [
            row["class_id"],
            row["class_name"],
            row["support"],
            _fmt(row["iou"]),
            _fmt(row["precision"]),
            _fmt(row["recall"]),
            _fmt(row["f1"]),
        ]
        for row in results["per_class"]
    ]
    lines.append(
        _md_table(
            ["id", "class", "support", "IoU", "precision", "recall", "F1"],
            per_class_rows,
        )
    )
    lines.append("")
    if "confusion_matrix" in figures:
        lines.append(f"![confusion matrix]({figures['confusion_matrix']})")
        lines.append("")

    lines.append("## Class size analysis")
    lines.append("")
    lines.append(
        "Classes are grouped by ground truth pixel support to show whether accuracy falls on "
        "rarer classes."
    )
    lines.append("")
    size_rows = [
        [
            b["bin"],
            ", ".join(b["class_names"]),
            b["total_support"],
            _fmt(b["mean_iou"]),
            _fmt(b["mean_recall"]),
        ]
        for b in results["class_size"]["bins"]
    ]
    lines.append(
        _md_table(
            ["bin", "classes", "total support", "mean IoU", "mean recall"], size_rows
        )
    )
    lines.append("")
    if "class_size" in figures:
        lines.append(f"![class size bins]({figures['class_size']})")
        lines.append("")

    boundary = results["boundary"]
    lines.append("## Boundary error analysis")
    lines.append("")
    lines.append(
        f"Pixels within {boundary['boundary_width']} pixels of a class boundary are compared "
        "with interior pixels. A higher interior score means errors concentrate at boundaries."
    )
    lines.append("")
    lines.append(
        _md_table(
            ["region", "mean IoU", "pixel accuracy", "pixels"],
            [
                [
                    "boundary band",
                    _fmt(boundary["boundary"]["mean_iou"]),
                    _fmt(boundary["boundary"]["pixel_accuracy"]),
                    boundary["boundary"]["pixels"],
                ],
                [
                    "interior",
                    _fmt(boundary["interior"]["mean_iou"]),
                    _fmt(boundary["interior"]["pixel_accuracy"]),
                    boundary["interior"]["pixels"],
                ],
            ],
        )
    )
    lines.append("")
    lines.append(
        f"Interior mean IoU minus boundary mean IoU is "
        f"{_fmt(boundary['interior_minus_boundary_miou'])}."
    )
    lines.append("")

    confidence = results["confidence"]
    lines.append("## Confidence versus correctness")
    lines.append("")
    lines.append(
        "Mean predicted confidence is compared between correct and incorrect pixels. A small "
        "gap means the model stays confident even when it is wrong."
    )
    lines.append("")
    lines.append(
        _md_table(
            ["group", "mean confidence", "pixels"],
            [
                [
                    "correct",
                    _fmt(confidence["mean_confidence_correct"]),
                    confidence["n_correct"],
                ],
                [
                    "incorrect",
                    _fmt(confidence["mean_confidence_incorrect"]),
                    confidence["n_incorrect"],
                ],
            ],
        )
    )
    lines.append("")
    if "confidence" in figures:
        lines.append(f"![confidence histogram]({figures['confidence']})")
        lines.append("")

    lines.append("## Worst performing images")
    lines.append("")
    lines.append(
        "The lowest mean IoU images, each shown as input, ground truth, prediction and an error "
        "overlay."
    )
    lines.append("")
    for item in results["worst_images"]:
        lines.append(
            f"Image index {item['index']} with mean IoU {_fmt(item['miou'])}."
        )
        lines.append("")
        lines.append(f"![error map index {item['index']}]({item['figure']})")
        lines.append("")

    report_path = out_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return report_path
