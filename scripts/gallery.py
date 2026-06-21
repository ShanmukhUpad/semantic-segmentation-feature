"""Build a single HTML gallery for a run directory.

This scans a run directory for metric files and figures and writes one gallery.html that shows
the evaluation and analysis tables, including the test scores, next to every figure and error
map. Open the file in a browser to see the whole run on one page. Pass --embed to inline the
images as base64 so the single file can be shared on its own.

    python scripts/gallery.py --run-dir results/dummy_run
"""
import argparse
import base64
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STYLE = """
body { font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #1b1b1b; background: #fafafa; }
h1 { margin-bottom: 4px; }
h2 { margin-top: 32px; border-bottom: 2px solid #ddd; padding-bottom: 4px; }
h3 { margin-top: 18px; color: #444; }
table { border-collapse: collapse; margin: 10px 0 20px 0; background: #fff; }
th, td { border: 1px solid #ccc; padding: 4px 10px; text-align: right; font-variant-numeric: tabular-nums; }
th { background: #eee; }
td.name, th.name { text-align: left; }
.figgrid { display: flex; flex-wrap: wrap; gap: 16px; }
.card { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 8px; }
.card img { display: block; max-width: 360px; height: auto; }
.card.wide { width: 100%; }
.card.wide img { max-width: 100%; }
.caption { font-size: 12px; color: #666; margin-top: 6px; word-break: break-all; }
.meta { color: #666; font-size: 13px; }
"""


def fmt(value, places=4):
    """Format a number for a cell, showing n a when missing."""
    if value is None:
        return "n a"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def html_table(headers, rows, first_col_left=True):
    """Return an HTML table string from headers and a list of row lists."""
    head_cells = "".join(
        f'<th class="name">{h}</th>' if (first_col_left and i == 0) else f"<th>{h}</th>"
        for i, h in enumerate(headers)
    )
    body = []
    for row in rows:
        cells = "".join(
            f'<td class="name">{c}</td>' if (first_col_left and i == 0) else f"<td>{c}</td>"
            for i, c in enumerate(row)
        )
        body.append(f"<tr>{cells}</tr>")
    return f"<table><tr>{head_cells}</tr>{''.join(body)}</table>"


def render_metrics_file(json_path, run_dir):
    """Render one metrics or analysis json file as an HTML section."""
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    rel_parent = json_path.parent.relative_to(run_dir).as_posix()
    split = data.get("split", "")
    is_analysis = json_path.name == "analysis.json"
    kind = "Failure analysis" if is_analysis else "Evaluation"
    parts = [f"<h2>{kind} ({split})</h2>"]
    parts.append(f'<div class="meta">source {rel_parent}/{json_path.name}</div>')

    overall = data.get("overall", {})
    overall_rows = [
        ["mean IoU", fmt(overall.get("mean_iou"))],
        ["pixel accuracy", fmt(overall.get("pixel_accuracy"))],
        ["mean F1", fmt(overall.get("mean_f1"))],
        ["macro precision", fmt(overall.get("macro_precision"))],
        ["macro recall", fmt(overall.get("macro_recall"))],
    ]
    parts.append("<h3>Overall</h3>")
    parts.append(html_table(["metric", "value"], overall_rows))

    per_class = data.get("per_class", [])
    if per_class:
        rows = [
            [
                row["class_name"],
                row["support"],
                fmt(row["iou"]),
                fmt(row["precision"]),
                fmt(row["recall"]),
                fmt(row["f1"]),
            ]
            for row in per_class
        ]
        parts.append("<h3>Per class</h3>")
        parts.append(
            html_table(["class", "support", "IoU", "precision", "recall", "F1"], rows)
        )

    if "class_size" in data:
        rows = [
            [
                b["bin"],
                ", ".join(b["class_names"]),
                b["total_support"],
                fmt(b["mean_iou"]),
                fmt(b["mean_recall"]),
            ]
            for b in data["class_size"]["bins"]
        ]
        parts.append("<h3>Class size bins</h3>")
        parts.append(
            html_table(["bin", "classes", "support", "mean IoU", "mean recall"], rows)
        )

    if "boundary" in data:
        boundary = data["boundary"]
        rows = [
            [
                "boundary band",
                fmt(boundary["boundary"]["mean_iou"]),
                fmt(boundary["boundary"]["pixel_accuracy"]),
                boundary["boundary"]["pixels"],
            ],
            [
                "interior",
                fmt(boundary["interior"]["mean_iou"]),
                fmt(boundary["interior"]["pixel_accuracy"]),
                boundary["interior"]["pixels"],
            ],
        ]
        parts.append(f"<h3>Boundary versus interior (width {boundary['boundary_width']})</h3>")
        parts.append(html_table(["region", "mean IoU", "pixel accuracy", "pixels"], rows))

    if "confidence" in data:
        conf = data["confidence"]
        rows = [
            ["correct", fmt(conf["mean_confidence_correct"]), conf["n_correct"]],
            ["incorrect", fmt(conf["mean_confidence_incorrect"]), conf["n_incorrect"]],
        ]
        parts.append("<h3>Confidence versus correctness</h3>")
        parts.append(html_table(["group", "mean confidence", "pixels"], rows))

    return "\n".join(parts)


def image_src(png_path, run_dir, embed):
    """Return the img src, a relative path by default or a base64 data uri when embedding."""
    if embed:
        data = base64.b64encode(png_path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{data}"
    return png_path.relative_to(run_dir).as_posix()


def render_figures(run_dir, embed):
    """Render every PNG under the run directory, grouped by folder."""
    pngs = sorted(run_dir.rglob("*.png"))
    if not pngs:
        return "<p>No figures found.</p>"
    groups = {}
    for path in pngs:
        groups.setdefault(path.parent.relative_to(run_dir).as_posix(), []).append(path)

    parts = ["<h2>Figures</h2>"]
    for group in sorted(groups):
        parts.append(f"<h3>{group}</h3>")
        wide = "error_maps" in group
        cards = []
        for path in groups[group]:
            src = image_src(path, run_dir, embed)
            klass = "card wide" if wide else "card"
            cards.append(
                f'<div class="{klass}"><img src="{src}" loading="lazy">'
                f'<div class="caption">{path.name}</div></div>'
            )
        parts.append(f'<div class="figgrid">{"".join(cards)}</div>')
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Build an HTML gallery for a run directory")
    parser.add_argument("--run-dir", required=True, help="a run output directory")
    parser.add_argument(
        "--embed",
        action="store_true",
        help="inline images as base64 so the html stands alone",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist")

    metric_files = sorted(
        list(run_dir.rglob("metrics.json")) + list(run_dir.rglob("analysis.json"))
    )

    sections = [
        "<!doctype html>",
        '<html><head><meta charset="utf-8">',
        f"<title>Run gallery {run_dir.name}</title>",
        f"<style>{STYLE}</style></head><body>",
        f"<h1>Run gallery</h1>",
        f'<div class="meta">{run_dir}</div>',
    ]
    if metric_files:
        for json_path in metric_files:
            sections.append(render_metrics_file(json_path, run_dir))
    else:
        sections.append("<p>No metric files found. Run evaluate or analyze first.</p>")
    sections.append(render_figures(run_dir, args.embed))
    sections.append("</body></html>")

    out_path = run_dir / "gallery.html"
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(sections))
    print(f"Wrote gallery to {out_path}")


if __name__ == "__main__":
    main()
