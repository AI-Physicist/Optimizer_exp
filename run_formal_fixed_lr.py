import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime


def parse_str_list(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def read_last_row(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty log file: {path}")
    return rows[-1], rows


def run_one(cfg, opt):
    run_name = f"{opt}_lr{cfg.lr:g}"
    log_file = os.path.join(cfg.log_dir, run_name.replace(".", "p") + ".csv")

    cmd = [
        sys.executable,
        "train.py",
        "--optimizer",
        opt,
        "--steps",
        str(cfg.steps),
        "--batch_size",
        str(cfg.batch_size),
        "--seq_len",
        str(cfg.seq_len),
        "--lr",
        str(cfg.lr),
        "--seed",
        str(cfg.seed),
        "--log_file",
        log_file,
    ]

    if opt == "rmsprop_pnorm":
        cmd.extend(["--rmsprop_p", str(cfg.r)])

    print(f"[run] {run_name}")
    subprocess.run(cmd, check=True)

    last, rows = read_last_row(log_file)
    losses = [float(x["loss"]) for x in rows]

    return {
        "optimizer": opt,
        "lr": cfg.lr,
        "r": cfg.r if opt == "rmsprop_pnorm" else "",
        "steps": int(last["step"]),
        "final_loss": float(last["loss"]),
        "best_loss": min(losses),
        "avg_step_time": sum(float(x["step_time"]) for x in rows) / len(rows),
        "peak_memory_mb": max(float(x["peak_memory_mb"]) for x in rows),
        "log_file": log_file,
    }


def write_outputs(out_csv, out_md, results):
    fieldnames = [
        "optimizer",
        "lr",
        "r",
        "steps",
        "final_loss",
        "best_loss",
        "avg_step_time",
        "peak_memory_mb",
        "log_file",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)

    lines = []
    lines.append("# Formal Experiment (fixed lr)")
    lines.append("")
    lines.append(
        "| optimizer | lr | r | final_loss | best_loss | avg_step_time | peak_memory_mb |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        r_value = r["r"] if r["r"] != "" else "-"
        lines.append(
            f"| {r['optimizer']} | {r['lr']:.6g} | {r_value} | {r['final_loss']:.6f} | "
            f"{r['best_loss']:.6f} | {r['avg_step_time']:.6f} | {r['peak_memory_mb']:.2f} |"
        )

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--r", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--optimizers",
        type=str,
        default="adamw,sgd,rmsprop,rmsprop_no_memory,rmsprop_pnorm,adafactor",
    )
    parser.add_argument("--log_dir", type=str, default="logs/synthetic/formal")
    parser.add_argument("--out_dir", type=str, default="results/synthetic/formal")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    opts = parse_str_list(args.optimizers)
    results = []
    for opt in opts:
        results.append(run_one(args, opt))

    results.sort(key=lambda x: x["final_loss"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.out_dir, f"formal_fixed_lr_{ts}.csv")
    out_md = os.path.join(args.out_dir, f"formal_fixed_lr_{ts}.md")
    write_outputs(out_csv, out_md, results)

    print("[done]", out_csv)
    print("[done]", out_md)


if __name__ == "__main__":
    main()
