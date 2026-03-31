import argparse
import csv
import glob
import math
import os


COLORS = {
    "adamw": "#1f77b4",
    "sgd": "#d62728",
    "rmsprop": "#2ca02c",
    "rmsprop_no_memory": "#8c564b",
    "rmsprop_pnorm": "#17becf",
    "adafactor": "#ff7f0e",
}

PANELS = [
    ("loss", "Loss", "loss"),
    ("grad_norm_l2", "Grad Norm (L2)", "sci"),
    ("update_ratio", "Update Ratio", "sci"),
    ("grad_update_cos", "Grad-Update Cos", "fixed3"),
    ("hessian_top_eig", "Hessian 2-Norm (power iter)", "sci"),
    ("step_time", "Step Time (s)", "fixed4"),
]

REFERENCE_LINES = {
    "grad_update_cos": [
        {"value": 0.0, "label": "0", "color": "#666666"},
        {"value": -1.0, "label": "-1", "color": "#999999"},
    ],
    "hessian_top_eig": [
        {"value": 0.0, "label": "0", "color": "#666666"},
    ],
}


def parse_float_or_nan(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_dynamics_logs(log_glob):
    data = {}
    for path in sorted(glob.glob(log_glob)):
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        if "hessian_top_eig" not in rows[0]:
            continue
        optimizer = rows[0].get("optimizer", os.path.basename(path).replace(".csv", ""))
        parsed = []
        for r in rows:
            parsed.append(
                {
                    "step": int(r["step"]),
                    "optimizer": optimizer,
                    "loss": parse_float_or_nan(r.get("loss")),
                    "step_time": parse_float_or_nan(r.get("step_time")),
                    "grad_norm_l2": parse_float_or_nan(r.get("grad_norm_l2")),
                    "update_ratio": parse_float_or_nan(r.get("update_ratio")),
                    "grad_update_cos": parse_float_or_nan(r.get("grad_update_cos")),
                    "hessian_top_eig": parse_float_or_nan(r.get("hessian_top_eig")),
                }
            )
        data[optimizer] = parsed
    return data


def map_to_plot(x, y, x_min, x_max, y_min, y_max, left, top, width, height):
    if x_max == x_min:
        px = left
    else:
        px = left + (x - x_min) / (x_max - x_min) * width
    if y_max == y_min:
        py = top + height
    else:
        py = top + height - (y - y_min) / (y_max - y_min) * height
    return px, py


def polyline(points, color, width=2):
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" points="{pts}" />'


def make_ticks(v_min, v_max, n=5):
    if n <= 1:
        return [v_min]
    if v_max == v_min:
        return [v_min for _ in range(n)]
    step = (v_max - v_min) / (n - 1)
    return [v_min + i * step for i in range(n)]


def fmt_tick(v, fmt_kind):
    if fmt_kind == "sci":
        return f"{v:.1e}"
    if fmt_kind == "fixed4":
        return f"{v:.4f}"
    if fmt_kind == "fixed3":
        return f"{v:.3f}"
    return f"{v:.2f}"


def finite_values(rows, key):
    vals = []
    for r in rows:
        v = r[key]
        if math.isfinite(v):
            vals.append(v)
    return vals


def panel_y_range(data, key):
    vals = []
    for rows in data.values():
        vals.extend(finite_values(rows, key))
    for ref in REFERENCE_LINES.get(key, []):
        vals.append(ref["value"])
    if not vals:
        return 0.0, 1.0
    y_min = min(vals)
    y_max = max(vals)
    if y_min == y_max:
        margin = 0.05 * (abs(y_min) + 1.0)
        return y_min - margin, y_max + margin
    margin = 0.08 * (y_max - y_min)
    return y_min - margin, y_max + margin


def draw_panel(svg, data, metric, title, fmt_kind, geom, x_min, x_max):
    left, top, panel_w, panel_h = geom
    y_min, y_max = panel_y_range(data, metric)

    svg.append(
        f'<rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" fill="#fafafa" stroke="#dddddd"/>'
    )
    svg.append(
        f'<text x="{left + panel_w/2:.1f}" y="{top - 10:.1f}" text-anchor="middle" '
        f'font-size="13" font-family="sans-serif">{title}</text>'
    )

    x_ticks = make_ticks(x_min, x_max, n=5)
    y_ticks = make_ticks(y_min, y_max, n=5)

    for xv in x_ticks:
        x, _ = map_to_plot(xv, y_min, x_min, x_max, y_min, y_max, left, top, panel_w, panel_h)
        y0 = top + panel_h
        svg.append(
            f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y0 + 4:.1f}" stroke="#666" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{x:.1f}" y="{y0 + 16:.1f}" text-anchor="middle" font-size="10" '
            f'font-family="sans-serif" fill="#555">{int(round(xv))}</text>'
        )

    for yv in y_ticks:
        _, y = map_to_plot(x_min, yv, x_min, x_max, y_min, y_max, left, top, panel_w, panel_h)
        svg.append(
            f'<line x1="{left - 4:.1f}" y1="{y:.1f}" x2="{left:.1f}" y2="{y:.1f}" stroke="#666" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{left - 8:.1f}" y="{y + 3:.1f}" text-anchor="end" font-size="10" '
            f'font-family="sans-serif" fill="#555">{fmt_tick(yv, fmt_kind)}</text>'
        )

    for ref in REFERENCE_LINES.get(metric, []):
        ref_value = ref["value"]
        _, y = map_to_plot(
            x_min, ref_value, x_min, x_max, y_min, y_max, left, top, panel_w, panel_h
        )
        svg.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + panel_w:.1f}" y2="{y:.1f}" '
            f'stroke="{ref["color"]}" stroke-width="1.5" stroke-dasharray="6,4"/>'
        )
        svg.append(
            f'<text x="{left + panel_w - 6:.1f}" y="{y - 4:.1f}" text-anchor="end" '
            f'font-size="10" font-family="sans-serif" fill="{ref["color"]}">{ref["label"]}</text>'
        )

    for opt, rows in sorted(data.items()):
        color = COLORS.get(opt, "#444444")
        points = []
        for r in rows:
            yv = r[metric]
            if not math.isfinite(yv):
                continue
            px, py = map_to_plot(
                r["step"], yv, x_min, x_max, y_min, y_max, left, top, panel_w, panel_h
            )
            points.append((px, py))
        if points:
            svg.append(polyline(points, color=color, width=2))

    svg.append(
        f'<text x="{left + panel_w/2:.1f}" y="{top + panel_h + 30:.1f}" text-anchor="middle" '
        'font-size="10" font-family="sans-serif" fill="#666">step</text>'
    )


def make_svg(data, output_path):
    width, height = 1400, 920
    margin_x = 90
    margin_y = 70
    gap_x = 45
    gap_y = 70
    cols, rows_n = 2, 3
    panel_w = (width - margin_x * 2 - gap_x) / cols
    panel_h = (height - margin_y * 2 - gap_y * (rows_n - 1)) / rows_n

    all_steps = [r["step"] for rows in data.values() for r in rows]
    x_min, x_max = min(all_steps), max(all_steps)

    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    svg.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff" />')
    svg.append(
        f'<text x="{width/2:.1f}" y="35" text-anchor="middle" font-size="22" '
        'font-family="sans-serif">Optimizer Dynamics Diagnostics</text>'
    )

    for i, (metric, title, fmt_kind) in enumerate(PANELS):
        row = i // cols
        col = i % cols
        left = margin_x + col * (panel_w + gap_x)
        top = margin_y + row * (panel_h + gap_y)
        draw_panel(
            svg=svg,
            data=data,
            metric=metric,
            title=title,
            fmt_kind=fmt_kind,
            geom=(left, top, panel_w, panel_h),
            x_min=x_min,
            x_max=x_max,
        )

    legend_x = width - 260
    legend_y = 40
    for i, opt in enumerate(sorted(data.keys())):
        color = COLORS.get(opt, "#444444")
        y = legend_y + i * 20
        svg.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        svg.append(
            f'<text x="{legend_x + 30}" y="{y + 4}" font-size="12" font-family="sans-serif">{opt}</text>'
        )

    svg.append("</svg>")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_glob", type=str, default="logs/real_text/dynamics_*.csv")
    parser.add_argument(
        "--output",
        type=str,
        default="results/real_text/dynamics/optimizer_dynamics.svg",
    )
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    log_glob = os.path.join(root, args.log_glob)
    output = os.path.join(root, args.output)

    data = load_dynamics_logs(log_glob)
    if not data:
        raise RuntimeError(
            f"No dynamics logs matched pattern: {args.log_glob}. "
            "Expected csv with hessian_top_eig column."
        )

    make_svg(data, output)
    print(f"saved: {output}")
    print(f"optimizers_plotted: {', '.join(sorted(data.keys()))}")


if __name__ == "__main__":
    main()
