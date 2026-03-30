import argparse
import csv
import glob
import math
import os
from statistics import mean


def load_logs(log_dir):
    data = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "*.csv"))):
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        optimizer = rows[0]["optimizer"]
        parsed = []
        for r in rows:
            parsed.append(
                {
                    "step": int(r["step"]),
                    "loss": float(r["loss"]),
                    "lr": float(r["lr"]),
                    "step_time": float(r["step_time"]),
                    "optimizer": r["optimizer"],
                    "seed": int(r["seed"]),
                    "peak_memory_mb": float(r["peak_memory_mb"]),
                }
            )
        data[optimizer] = parsed
    return data


def build_summary(data):
    summary = []
    for opt, rows in data.items():
        final = rows[-1]
        summary.append(
            {
                "optimizer": opt,
                "final_loss": final["loss"],
                "best_loss": min(r["loss"] for r in rows),
                "avg_step_time_s": mean(r["step_time"] for r in rows),
                "peak_memory_mb": max(r["peak_memory_mb"] for r in rows),
                "seed": final["seed"],
                "max_step": final["step"],
            }
        )
    summary.sort(key=lambda x: x["final_loss"])
    return summary


def save_summary_csv(path, summary):
    fieldnames = [
        "optimizer",
        "final_loss",
        "best_loss",
        "avg_step_time_s",
        "peak_memory_mb",
        "seed",
        "max_step",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary:
            writer.writerow(
                {
                    "optimizer": r["optimizer"],
                    "final_loss": f"{r['final_loss']:.6f}",
                    "best_loss": f"{r['best_loss']:.6f}",
                    "avg_step_time_s": f"{r['avg_step_time_s']:.6f}",
                    "peak_memory_mb": f"{r['peak_memory_mb']:.2f}",
                    "seed": r["seed"],
                    "max_step": r["max_step"],
                }
            )


def save_summary_md(path, summary):
    lines = []
    lines.append("# Optimizer Result Summary")
    lines.append("")
    lines.append("| optimizer | final_loss | best_loss | avg_step_time_s | peak_memory_mb | seed | max_step |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in summary:
        lines.append(
            f"| {r['optimizer']} | {r['final_loss']:.6f} | {r['best_loss']:.6f} | "
            f"{r['avg_step_time_s']:.6f} | {r['peak_memory_mb']:.2f} | {r['seed']} | {r['max_step']} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def polyline(points, color, width=2):
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" points="{pts}" />'


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


def make_ticks(v_min, v_max, n=6):
    if n <= 1:
        return [v_min]
    if v_max == v_min:
        return [v_min for _ in range(n)]
    step = (v_max - v_min) / (n - 1)
    return [v_min + i * step for i in range(n)]


def add_axis_ticks(svg, left, top, panel_w, panel_h, x_min, x_max, y_min, y_max, y_fmt):
    x_ticks = make_ticks(x_min, x_max, n=6)
    y_ticks = make_ticks(y_min, y_max, n=6)

    # x-axis ticks and labels
    for xv in x_ticks:
        x, _ = map_to_plot(xv, y_min, x_min, x_max, y_min, y_max, left, top, panel_w, panel_h)
        y0 = top + panel_h
        svg.append(f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y0 + 5:.1f}" stroke="#666" stroke-width="1"/>')
        svg.append(
            f'<text x="{x:.1f}" y="{y0 + 20:.1f}" text-anchor="middle" font-size="11" font-family="sans-serif" fill="#555">{int(round(xv))}</text>'
        )

    # y-axis ticks and labels
    for yv in y_ticks:
        _, y = map_to_plot(x_min, yv, x_min, x_max, y_min, y_max, left, top, panel_w, panel_h)
        svg.append(f'<line x1="{left - 5:.1f}" y1="{y:.1f}" x2="{left:.1f}" y2="{y:.1f}" stroke="#666" stroke-width="1"/>')
        svg.append(
            f'<text x="{left - 10:.1f}" y="{y + 4:.1f}" text-anchor="end" font-size="11" font-family="sans-serif" fill="#555">{y_fmt(yv)}</text>'
        )


def make_svg(data, output_path, loss_theoretical_floor=None):
    colors = {
        "adamw": "#1f77b4",
        "sgd": "#d62728",
        "rmsprop": "#2ca02c",
        "rmsprop_no_memory": "#8c564b",
        "rmsprop_pnorm": "#17becf",
        "adafactor": "#ff7f0e",
    }

    width, height = 1200, 520
    margin = 60
    panel_w = (width - margin * 3) / 2
    panel_h = height - margin * 2

    left1, top1 = margin, margin
    left2, top2 = margin * 2 + panel_w, margin

    all_steps = [r["step"] for rows in data.values() for r in rows]
    all_loss = [r["loss"] for rows in data.values() for r in rows]
    all_time = [r["step_time"] for rows in data.values() for r in rows]

    x_min, x_max = min(all_steps), max(all_steps)
    loss_min, loss_max = min(all_loss), max(all_loss)
    if loss_theoretical_floor is not None:
        loss_min = min(loss_min, loss_theoretical_floor)
        loss_max = max(loss_max, loss_theoretical_floor)
    time_min, time_max = min(all_time), max(all_time)

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff" />')

    svg.append(f'<text x="{width/2:.1f}" y="30" text-anchor="middle" font-size="20" font-family="sans-serif">Optimizer Comparison</text>')

    for left, top, title in [(left1, top1, "Loss vs Step"), (left2, top2, "Step Time vs Step")]:
        svg.append(f'<rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" fill="#fafafa" stroke="#dddddd"/>')
        svg.append(f'<text x="{left + panel_w/2:.1f}" y="{top - 12}" text-anchor="middle" font-size="14" font-family="sans-serif">{title}</text>')

    add_axis_ticks(
        svg,
        left1,
        top1,
        panel_w,
        panel_h,
        x_min,
        x_max,
        loss_min,
        loss_max,
        y_fmt=lambda v: f"{v:.1f}",
    )
    add_axis_ticks(
        svg,
        left2,
        top2,
        panel_w,
        panel_h,
        x_min,
        x_max,
        time_min,
        time_max,
        y_fmt=lambda v: f"{v:.3f}",
    )

    if loss_theoretical_floor is not None:
        y_floor = map_to_plot(
            x_min,
            loss_theoretical_floor,
            x_min,
            x_max,
            loss_min,
            loss_max,
            left1,
            top1,
            panel_w,
            panel_h,
        )[1]
        svg.append(
            f'<line x1="{left1:.1f}" y1="{y_floor:.1f}" x2="{left1 + panel_w:.1f}" y2="{y_floor:.1f}" '
            'stroke="#666666" stroke-width="2" stroke-dasharray="8,6" />'
        )
        svg.append(
            f'<text x="{left1 + 10:.1f}" y="{y_floor - 6:.1f}" font-size="12" font-family="sans-serif" fill="#444444">'
            f'theoretical floor: ln(V)={loss_theoretical_floor:.4f}</text>'
        )

    for opt, rows in sorted(data.items()):
        color = colors.get(opt, "#444444")

        loss_points = []
        time_points = []
        for r in rows:
            px1, py1 = map_to_plot(r["step"], r["loss"], x_min, x_max, loss_min, loss_max, left1, top1, panel_w, panel_h)
            px2, py2 = map_to_plot(r["step"], r["step_time"], x_min, x_max, time_min, time_max, left2, top2, panel_w, panel_h)
            loss_points.append((px1, py1))
            time_points.append((px2, py2))

        svg.append(polyline(loss_points, color=color, width=2))
        svg.append(polyline(time_points, color=color, width=2))

    legend_x = width - 210
    legend_y = 45
    i = 0
    for opt in sorted(data.keys()):
        color = colors.get(opt, "#444444")
        y = legend_y + i * 22
        svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 25}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{legend_x + 32}" y="{y + 4}" font-size="13" font-family="sans-serif">{opt}</text>')
        i += 1

    if loss_theoretical_floor is not None:
        y = legend_y + i * 22
        svg.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 25}" y2="{y}" stroke="#666666" stroke-width="2" stroke-dasharray="8,6"/>'
        )
        svg.append(f'<text x="{legend_x + 32}" y="{y + 4}" font-size="13" font-family="sans-serif">ln(V)</text>')

    svg.append(f'<text x="{left1 + panel_w/2:.1f}" y="{height - 15}" text-anchor="middle" font-size="12" font-family="sans-serif">step</text>')
    svg.append(f'<text x="{left2 + panel_w/2:.1f}" y="{height - 15}" text-anchor="middle" font-size="12" font-family="sans-serif">step</text>')
    svg.append(f'<text x="{left1 - 35}" y="{top1 + panel_h/2:.1f}" text-anchor="middle" font-size="12" font-family="sans-serif" transform="rotate(-90 {left1 - 35},{top1 + panel_h/2:.1f})">loss</text>')
    svg.append(f'<text x="{left2 - 35}" y="{top2 + panel_h/2:.1f}" text-anchor="middle" font-size="12" font-family="sans-serif" transform="rotate(-90 {left2 - 35},{top2 + panel_h/2:.1f})">step_time (s)</text>')

    svg.append("</svg>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=32000)
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(root, "logs", "synthetic", "base")
    out_dir = os.path.join(root, "results", "synthetic", "base")
    os.makedirs(out_dir, exist_ok=True)

    data = load_logs(log_dir)
    if not data:
        raise RuntimeError(f"No log csv found in {log_dir}")

    loss_theoretical_floor = math.log(args.vocab_size)

    summary = build_summary(data)
    save_summary_csv(os.path.join(out_dir, "summary.csv"), summary)
    save_summary_md(os.path.join(out_dir, "summary.md"), summary)
    make_svg(
        data,
        os.path.join(out_dir, "optimizer_comparison.svg"),
        loss_theoretical_floor=loss_theoretical_floor,
    )

    print(f"saved: {os.path.join(out_dir, 'summary.csv')}")
    print(f"saved: {os.path.join(out_dir, 'summary.md')}")
    print(f"saved: {os.path.join(out_dir, 'optimizer_comparison.svg')}")
    print(f"theoretical_loss_floor_lnV: {loss_theoretical_floor:.6f}")


if __name__ == "__main__":
    main()
