import csv
import glob
import os
from statistics import mean

from summarize_results import make_svg


def load_real_text_logs(log_dir):
    data = {}
    pattern = os.path.join(log_dir, "*_real_text.csv")
    for path in sorted(glob.glob(pattern)):
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


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(root, "logs", "real_text")
    out_dir = os.path.join(root, "results", "real_text")
    os.makedirs(out_dir, exist_ok=True)

    data = load_real_text_logs(log_dir)
    if not data:
        raise RuntimeError(f"No real-text logs found in {log_dir}")

    summary = []
    for opt, rows in data.items():
        summary.append(
            {
                "optimizer": opt,
                "final_loss": rows[-1]["loss"],
                "best_loss": min(r["loss"] for r in rows),
                "avg_step_time_s": mean(r["step_time"] for r in rows),
                "peak_memory_mb": max(r["peak_memory_mb"] for r in rows),
                "seed": rows[-1]["seed"],
                "max_step": rows[-1]["step"],
            }
        )
    summary.sort(key=lambda x: x["final_loss"])

    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "optimizer",
                "final_loss",
                "best_loss",
                "avg_step_time_s",
                "peak_memory_mb",
                "seed",
                "max_step",
            ],
        )
        w.writeheader()
        for r in summary:
            w.writerow(
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

    md_path = os.path.join(out_dir, "summary.md")
    lines = [
        "# Real Text Experiment Summary",
        "",
        "| optimizer | final_loss | best_loss | avg_step_time_s | peak_memory_mb | seed | max_step |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['optimizer']} | {r['final_loss']:.6f} | {r['best_loss']:.6f} | "
            f"{r['avg_step_time_s']:.6f} | {r['peak_memory_mb']:.2f} | {r['seed']} | {r['max_step']} |"
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    fig_path = os.path.join(out_dir, "optimizer_comparison.svg")
    make_svg(data, fig_path, loss_theoretical_floor=None)

    print(f"saved: {csv_path}")
    print(f"saved: {md_path}")
    print(f"saved: {fig_path}")
    print(f"num_optimizers: {len(data)}")


if __name__ == "__main__":
    main()
