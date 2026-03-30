import csv
import glob
import os
from statistics import mean, stdev

from summarize_results import make_svg


def read_log(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for r in rows:
        parsed.append(
            {
                "step": int(r["step"]),
                "loss": float(r["loss"]),
                "step_time": float(r["step_time"]),
                "optimizer": r["optimizer"],
                "seed": int(r["seed"]),
                "peak_memory_mb": float(r["peak_memory_mb"]),
            }
        )
    return parsed


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(root, "logs", "real_text", "multiseed_1000")
    out_dir = os.path.join(root, "results", "real_text", "multiseed_1000")
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
        avg_step_times = [mean(x["step_time"] for x in r[:n_steps]) for r in runs]
        peak_mems = [max(x["peak_memory_mb"] for x in r[:n_steps]) for r in runs]
        seeds = [r[0]["seed"] for r in runs]

        for i in range(n_steps):
            mean_rows.append(
                {
                    "step": runs[0][i]["step"],
                    "loss": mean(r[i]["loss"] for r in runs),
                    "step_time": mean(r[i]["step_time"] for r in runs),
                    "optimizer": opt,
                    "seed": -1,
                    "peak_memory_mb": mean(r[i]["peak_memory_mb"] for r in runs),
                }
            )
        plot_data[opt] = mean_rows

        summary.append(
            {
                "optimizer": opt,
                "num_seeds": n_runs,
                "seeds": ",".join(str(s) for s in sorted(seeds)),
                "final_loss_mean": mean(final_losses),
                "final_loss_std": stdev(final_losses) if len(final_losses) > 1 else 0.0,
                "avg_step_time_mean": mean(avg_step_times),
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
                    "final_loss_mean": f"{r['final_loss_mean']:.6f}",
                    "final_loss_std": f"{r['final_loss_std']:.6f}",
                    "avg_step_time_mean": f"{r['avg_step_time_mean']:.6f}",
                    "peak_memory_mb_max": f"{r['peak_memory_mb_max']:.2f}",
                    "max_step": r["max_step"],
                }
            )

    md_path = os.path.join(out_dir, "summary.md")
    lines = [
        "# Real Text 1000-Step Multi-Seed Summary",
        "",
        "| optimizer | num_seeds | seeds | final_loss_mean | final_loss_std | avg_step_time_mean | peak_memory_mb_max | max_step |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['optimizer']} | {r['num_seeds']} | {r['seeds']} | {r['final_loss_mean']:.6f} | "
            f"{r['final_loss_std']:.6f} | {r['avg_step_time_mean']:.6f} | {r['peak_memory_mb_max']:.2f} | {r['max_step']} |"
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
