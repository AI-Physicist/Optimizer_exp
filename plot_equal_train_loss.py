import argparse
import csv
import math
import os


COLORS = {
    "adamw": "#1f77b4",
    "sgd": "#d62728",
    "rmsprop": "#2ca02c",
    "adafactor": "#ff7f0e",
    "rmsprop_no_memory": "#8c564b",
    "rmsprop_pnorm": "#17becf",
}


def parse_float(v):
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return float("nan")
        return x
    except (TypeError, ValueError):
        return float("nan")


def load_matches(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    parsed = []
    for r in rows:
        parsed.append(
            {
                "anchor_id": int(r["anchor_id"]),
                "anchor_train_loss": parse_float(r["anchor_train_loss"]),
                "optimizer": r["optimizer"],
                "matched_step": parse_float(r.get("matched_step")),
                "matched_train_loss": parse_float(r["matched_train_loss"]),
                "abs_train_loss_error": parse_float(r["abs_train_loss_error"]),
                "test_loss": parse_float(r["test_loss"]),
                "test_ece": parse_float(r["test_ece"]),
                "test_acc": parse_float(r["test_acc"]),
                "param_norm_l2": parse_float(r["param_norm_l2"]),
                "generalization_gap": parse_float(r["generalization_gap"]),
            }
        )
    return parsed


def group_by_optimizer(rows):
    grouped = {}
    for r in rows:
        grouped.setdefault(r["optimizer"], []).append(r)
    for opt in grouped:
        grouped[opt].sort(key=lambda x: x["anchor_train_loss"])
    return grouped


def map_to_plot(x, y, x_min, x_max, y_min, y_max, left, top, width, height):
    if x_max == x_min:
        px = left
    else:
        px = left + (x - x_min) / (x_max - x_min) * width
    if y_max == y_min:
        py = top + height / 2.0
    else:
        py = top + height - (y - y_min) / (y_max - y_min) * height
    return px, py


def make_ticks(v_min, v_max, n=5):
    if n <= 1:
        return [v_min]
    if v_min == v_max:
        return [v_min for _ in range(n)]
    step = (v_max - v_min) / (n - 1)
    return [v_min + i * step for i in range(n)]


def polyline(points, color, width=2):
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" points="{pts}" />'


def add_panel(svg, left, top, panel_w, panel_h, title):
    svg.append(
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{panel_w:.1f}" height="{panel_h:.1f}" '
        'fill="#fafafa" stroke="#dddddd"/>'
    )
    svg.append(
        f'<text x="{left + panel_w/2:.1f}" y="{top - 10:.1f}" text-anchor="middle" '
        'font-size="13" font-family="sans-serif">'
        f"{title}</text>"
    )


def add_axis_ticks(svg, left, top, panel_w, panel_h, x_min, x_max, y_min, y_max, y_fmt):
    x_ticks = make_ticks(x_min, x_max, n=5)
    y_ticks = make_ticks(y_min, y_max, n=5)

    for xv in x_ticks:
        x, _ = map_to_plot(xv, y_min, x_min, x_max, y_min, y_max, left, top, panel_w, panel_h)
        y0 = top + panel_h
        svg.append(
            f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y0 + 4:.1f}" stroke="#666" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{x:.1f}" y="{y0 + 17:.1f}" text-anchor="middle" font-size="10" font-family="sans-serif" fill="#555">{xv:.1f}</text>'
        )

    for yv in y_ticks:
        _, y = map_to_plot(x_min, yv, x_min, x_max, y_min, y_max, left, top, panel_w, panel_h)
        svg.append(
            f'<line x1="{left - 4:.1f}" y1="{y:.1f}" x2="{left:.1f}" y2="{y:.1f}" stroke="#666" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{left - 7:.1f}" y="{y + 3:.1f}" text-anchor="end" font-size="10" font-family="sans-serif" fill="#555">{y_fmt(yv)}</text>'
        )


def y_range(values):
    finite = [v for v in values if not math.isnan(v) and not math.isinf(v)]
    if not finite:
        return 0.0, 1.0
    lo = min(finite)
    hi = max(finite)
    if math.isclose(lo, hi):
        pad = 1.0 if hi == 0.0 else abs(hi) * 0.1
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.1
    return lo - pad, hi + pad


def draw_metrics_figure(grouped, out_path):
    width, height = 1400, 880
    margin = 65
    gutter_x = 70
    gutter_y = 85
    panel_w = (width - margin * 2 - gutter_x) / 2
    panel_h = (height - margin * 2 - gutter_y) / 2

    panels = [
        ("test_loss", "Test Loss @ Matched Train-Loss Anchor", lambda v: f"{v:.1f}"),
        ("test_ece", "ECE @ Matched Train-Loss Anchor", lambda v: f"{v:.2f}"),
        ("param_norm_l2", "Param Norm (L2) @ Anchor", lambda v: f"{v:.0f}"),
        ("abs_train_loss_error", "Train-Loss Matching Error (Abs)", lambda v: f"{v:.1f}"),
    ]

    all_rows = [r for rows in grouped.values() for r in rows]
    x_values = [r["anchor_train_loss"] for r in all_rows]
    x_min, x_max = min(x_values), max(x_values)

    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    svg.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')
    svg.append(
        f'<text x="{width/2:.1f}" y="33" text-anchor="middle" font-size="22" font-family="sans-serif">'
        "Equal-Train-Loss Comparison: Generalization and Implicit Regularization</text>"
    )

    for idx, (metric, title, y_fmt) in enumerate(panels):
        row_i = idx // 2
        col_i = idx % 2
        left = margin + col_i * (panel_w + gutter_x)
        top = margin + row_i * (panel_h + gutter_y)

        add_panel(svg, left, top, panel_w, panel_h, title)
        y_vals = [r[metric] for r in all_rows]
        y_min, y_max = y_range(y_vals)
        add_axis_ticks(svg, left, top, panel_w, panel_h, x_min, x_max, y_min, y_max, y_fmt)

        for opt, rows in sorted(grouped.items()):
            color = COLORS.get(opt, "#444444")
            points = []
            for r in rows:
                if math.isnan(r["anchor_train_loss"]) or math.isnan(r[metric]):
                    continue
                px, py = map_to_plot(
                    r["anchor_train_loss"],
                    r[metric],
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    left,
                    top,
                    panel_w,
                    panel_h,
                )
                points.append((px, py))
            if len(points) >= 2:
                svg.append(polyline(points, color, width=2))
            for px, py in points:
                svg.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" fill="{color}" stroke="#ffffff" stroke-width="0.8"/>'
                )

        svg.append(
            f'<text x="{left + panel_w/2:.1f}" y="{top + panel_h + 35:.1f}" text-anchor="middle" '
            'font-size="11" font-family="sans-serif">Anchor Train Loss (higher -> earlier training)</text>'
        )

    legend_x = width - 250
    legend_y = 42
    i = 0
    for opt in sorted(grouped.keys()):
        y = legend_y + i * 21
        color = COLORS.get(opt, "#444444")
        svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        svg.append(
            f'<text x="{legend_x + 30}" y="{y + 4}" font-size="12" font-family="sans-serif">{opt}</text>'
        )
        i += 1

    svg.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg) + "\n")


def draw_final_anchor_bar(grouped, out_path):
    # Use smallest anchor_train_loss, i.e. near converged region.
    all_rows = [r for rows in grouped.values() for r in rows]
    min_anchor = min(r["anchor_train_loss"] for r in all_rows)

    picked = []
    for opt, rows in sorted(grouped.items()):
        row = min(rows, key=lambda x: abs(x["anchor_train_loss"] - min_anchor))
        picked.append(row)

    width, height = 1000, 520
    left, top = 70, 70
    panel_w, panel_h = 860, 360
    bar_w = panel_w / max(1, len(picked)) * 0.62
    finite_test_losses = [r["test_loss"] for r in picked if not math.isnan(r["test_loss"])]
    if not finite_test_losses:
        svg = []
        svg.append(
            '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="220" viewBox="0 0 900 220">'
        )
        svg.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')
        svg.append(
            '<text x="450" y="80" text-anchor="middle" font-size="22" font-family="sans-serif">'
            "Near-Converged Equal-Train-Loss Anchor</text>"
        )
        svg.append(
            '<text x="450" y="130" text-anchor="middle" font-size="15" font-family="sans-serif" fill="#666">'
            "No finite test_loss values found. Re-run logs with evaluation enabled.</text>"
        )
        svg.append("</svg>")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(svg) + "\n")
        return
    max_y = max(finite_test_losses) * 1.15

    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    svg.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')
    svg.append(
        f'<text x="{width/2:.1f}" y="33" text-anchor="middle" font-size="21" font-family="sans-serif">'
        f"Near-Converged Equal-Train-Loss Anchor (train_loss ~= {min_anchor:.4f})</text>"
    )
    svg.append(
        f'<rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" fill="#fafafa" stroke="#dddddd"/>'
    )

    for i in range(6):
        yv = max_y * i / 5
        y = top + panel_h - (yv / max_y) * panel_h
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + panel_w}" y2="{y:.1f}" stroke="#eeeeee" stroke-width="1"/>')
        svg.append(
            f'<text x="{left - 8:.1f}" y="{y + 3:.1f}" text-anchor="end" font-size="10" font-family="sans-serif" fill="#666">{yv:.2f}</text>'
        )

    for idx, r in enumerate(picked):
        x_center = left + (idx + 0.5) * (panel_w / len(picked))
        h = (r["test_loss"] / max_y) * panel_h
        y = top + panel_h - h
        x = x_center - bar_w / 2
        color = COLORS.get(r["optimizer"], "#444444")
        svg.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" opacity="0.88"/>'
        )
        svg.append(
            f'<text x="{x_center:.1f}" y="{top + panel_h + 18:.1f}" text-anchor="middle" font-size="11" font-family="sans-serif">{r["optimizer"]}</text>'
        )
        svg.append(
            f'<text x="{x_center:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="10" font-family="sans-serif">{r["test_loss"]:.3f}</text>'
        )
        svg.append(
            f'<text x="{x_center:.1f}" y="{top + panel_h + 33:.1f}" text-anchor="middle" font-size="9" font-family="sans-serif" fill="#666">ECE {r["test_ece"]:.3f}</text>'
        )
        svg.append(
            f'<text x="{x_center:.1f}" y="{top + panel_h + 46:.1f}" text-anchor="middle" font-size="9" font-family="sans-serif" fill="#666">||w|| {r["param_norm_l2"]:.1f}</text>'
        )

    svg.append(
        f'<text x="{left + panel_w/2:.1f}" y="{height - 16:.1f}" text-anchor="middle" font-size="11" font-family="sans-serif">Bar height: test_loss at near-converged equal-train-loss anchor</text>'
    )
    svg.append("</svg>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matches_csv",
        type=str,
        default="results/real_text/fixed_1000/equal_train_loss_matches.csv",
    )
    parser.add_argument(
        "--out_metrics_svg",
        type=str,
        default="results/real_text/fixed_1000/equal_train_loss_metrics.svg",
    )
    parser.add_argument(
        "--out_final_anchor_svg",
        type=str,
        default="results/real_text/fixed_1000/equal_train_loss_final_anchor.svg",
    )
    args = parser.parse_args()

    rows = load_matches(args.matches_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {args.matches_csv}")
    grouped = group_by_optimizer(rows)

    os.makedirs(os.path.dirname(args.out_metrics_svg), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_final_anchor_svg), exist_ok=True)

    draw_metrics_figure(grouped, args.out_metrics_svg)
    draw_final_anchor_bar(grouped, args.out_final_anchor_svg)

    print(f"saved: {args.out_metrics_svg}")
    print(f"saved: {args.out_final_anchor_svg}")


if __name__ == "__main__":
    main()
