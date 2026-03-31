import argparse
import csv
import glob
import math
import os
from statistics import mean, stdev

from summarize_results import make_svg


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def is_finite(x):
    return not math.isnan(x) and not math.isinf(x)


def mean_or_nan(values):
    finite = [v for v in values if is_finite(v)]
    if not finite:
        return float("nan")
    return mean(finite)


def stdev_or_zero(values):
    finite = [v for v in values if is_finite(v)]
    if len(finite) <= 1:
        return 0.0
    return stdev(finite)


def fmt(x, digits=6):
    if not is_finite(x):
        return ""
    return f"{x:.{digits}f}"


def read_log(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for r in rows:
        parsed.append(
            {
                "step": int(r["step"]),
                "loss": parse_float(r["loss"]),
                "step_time": parse_float(r["step_time"]),
                "optimizer": r["optimizer"],
                "seed": int(r["seed"]),
                "peak_memory_mb": parse_float(r["peak_memory_mb"]),
                "test_loss": parse_float(r.get("test_loss")),
                "test_ece": parse_float(r.get("test_ece")),
                "test_acc": parse_float(r.get("test_acc")),
                "param_norm_l2": parse_float(r.get("param_norm_l2")),
            }
        )
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs/real_text/multiseed_1000")
    parser.add_argument("--out_dir", type=str, default="results/real_text/multiseed_1000")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    log_dir = args.log_dir
    out_dir = args.out_dir
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(root, log_dir)
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(root, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    grouped = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "*.csv"))):
        rows = read_log(path)
        if not rows:
            continue
        opt = rows[0]["optimizer"]
        grouped.setdefault(opt, []).append(rows)

    if not grouped:
        raise RuntimeError(f"No logs found in {log_dir}")

    # Build mean curve per optimizer for plotting.
    plot_data = {}
    summary = []
    for opt, runs in grouped.items():
        n_runs = len(runs)
        n_steps = min(len(r) for r in runs)
        mean_rows = []
        final_losses = [r[n_steps - 1]["loss"] for r in runs]
        final_test_losses = [r[n_steps - 1]["test_loss"] for r in runs]
        final_test_eces = [r[n_steps - 1]["test_ece"] for r in runs]
        final_test_accs = [r[n_steps - 1]["test_acc"] for r in runs]
        final_param_norms = [r[n_steps - 1]["param_norm_l2"] for r in runs]
        avg_step_times = [mean(x["step_time"] for x in r[:n_steps]) for r in runs]
        peak_mems = [max(x["peak_memory_mb"] for x in r[:n_steps]) for r in runs]
        seeds = [r[0]["seed"] for r in runs]

        for i in range(n_steps):
            mean_rows.append(
                {
                    "step": runs[0][i]["step"],
                    "loss": mean_or_nan([r[i]["loss"] for r in runs]),
                    "step_time": mean_or_nan([r[i]["step_time"] for r in runs]),
                    "optimizer": opt,
                    "seed": -1,
                    "peak_memory_mb": mean_or_nan([r[i]["peak_memory_mb"] for r in runs]),
                }
            )
        plot_data[opt] = mean_rows

        summary.append(
            {
                "optimizer": opt,
                "num_seeds": n_runs,
                "seeds": ",".join(str(s) for s in sorted(seeds)),
                "final_loss_mean": mean_or_nan(final_losses),
                "final_loss_std": stdev_or_zero(final_losses),
                "final_test_loss_mean": mean_or_nan(final_test_losses),
                "final_test_loss_std": stdev_or_zero(final_test_losses),
                "final_test_ece_mean": mean_or_nan(final_test_eces),
                "final_test_ece_std": stdev_or_zero(final_test_eces),
                "final_test_acc_mean": mean_or_nan(final_test_accs),
                "final_test_acc_std": stdev_or_zero(final_test_accs),
                "final_param_norm_l2_mean": mean_or_nan(final_param_norms),
                "final_param_norm_l2_std": stdev_or_zero(final_param_norms),
                "avg_step_time_mean": mean_or_nan(avg_step_times),
                "peak_memory_mb_max": max(peak_mems),
                "max_step": runs[0][n_steps - 1]["step"],
            }
        )

    summary.sort(key=lambda x: x["final_loss_mean"])

    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "optimizer",
                "num_seeds",
                "seeds",
                "final_loss_mean",
                "final_loss_std",
                "final_test_loss_mean",
                "final_test_loss_std",
                "final_test_ece_mean",
                "final_test_ece_std",
                "final_test_acc_mean",
                "final_test_acc_std",
                "final_param_norm_l2_mean",
                "final_param_norm_l2_std",
                "avg_step_time_mean",
                "peak_memory_mb_max",
                "max_step",
            ],
        )
        w.writeheader()
        for r in summary:
            w.writerow(
                {
                    "optimizer": r["optimizer"],
                    "num_seeds": r["num_seeds"],
                    "seeds": r["seeds"],
                    "final_loss_mean": fmt(r["final_loss_mean"]),
                    "final_loss_std": fmt(r["final_loss_std"]),
                    "final_test_loss_mean": fmt(r["final_test_loss_mean"]),
                    "final_test_loss_std": fmt(r["final_test_loss_std"]),
                    "final_test_ece_mean": fmt(r["final_test_ece_mean"]),
                    "final_test_ece_std": fmt(r["final_test_ece_std"]),
                    "final_test_acc_mean": fmt(r["final_test_acc_mean"]),
                    "final_test_acc_std": fmt(r["final_test_acc_std"]),
                    "final_param_norm_l2_mean": fmt(r["final_param_norm_l2_mean"]),
                    "final_param_norm_l2_std": fmt(r["final_param_norm_l2_std"]),
                    "avg_step_time_mean": fmt(r["avg_step_time_mean"]),
                    "peak_memory_mb_max": f"{r['peak_memory_mb_max']:.2f}",
                    "max_step": r["max_step"],
                }
            )

    md_path = os.path.join(out_dir, "summary.md")
    lines = [
        "# Real Text 1000-Step Multi-Seed Summary",
        "",
        "| optimizer | num_seeds | seeds | final_loss_mean | final_loss_std | final_test_loss_mean | final_test_ece_mean | final_test_acc_mean | final_param_norm_l2_mean | avg_step_time_mean | peak_memory_mb_max | max_step |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['optimizer']} | {r['num_seeds']} | {r['seeds']} | {fmt(r['final_loss_mean'])} | "
            f"{fmt(r['final_loss_std'])} | {fmt(r['final_test_loss_mean'])} | "
            f"{fmt(r['final_test_ece_mean'])} | {fmt(r['final_test_acc_mean'])} | "
            f"{fmt(r['final_param_norm_l2_mean'])} | {fmt(r['avg_step_time_mean'])} | "
            f"{r['peak_memory_mb_max']:.2f} | {r['max_step']} |"
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    fig_path = os.path.join(out_dir, "optimizer_comparison.svg")
    make_svg(plot_data, fig_path, loss_theoretical_floor=None)

    print(f"saved: {csv_path}")
    print(f"saved: {md_path}")
    print(f"saved: {fig_path}")
    print(f"num_optimizers: {len(summary)}")


if __name__ == "__main__":
    main()
