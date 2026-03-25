import argparse
import csv
import math
import os
import subprocess
import sys
from datetime import datetime


def parse_float_list(value):
    parts = [x.strip() for x in value.split(",") if x.strip()]
    return [float(x) for x in parts]


def safe_tag(v):
    s = f"{v:g}"
    return s.replace("-", "m").replace(".", "p")


def read_last_row(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty log file: {path}")
    return rows[-1], rows


def run_one(cfg, lr, r):
    run_name = f"{cfg.optimizer}_lr{safe_tag(lr)}_r{safe_tag(r)}"
    log_file = os.path.join(cfg.log_dir, run_name + ".csv")

    cmd = [
        sys.executable,
        "train.py",
        "--optimizer",
        cfg.optimizer,
        "--steps",
        str(cfg.steps),
        "--batch_size",
        str(cfg.batch_size),
        "--seq_len",
        str(cfg.seq_len),
        "--lr",
        str(lr),
        "--seed",
        str(cfg.seed),
        "--log_file",
        log_file,
    ]

    if cfg.optimizer == "rmsprop_pnorm":
        cmd.extend(["--rmsprop_p", str(r)])

    print(f"[run] {run_name}")
    subprocess.run(cmd, check=True)

    last, rows = read_last_row(log_file)
    losses = [float(x["loss"]) for x in rows]

    final_loss = float(last["loss"])
    diverged = int(any(math.isnan(x) or math.isinf(x) or x > cfg.diverge_loss for x in losses))

    return {
        "run": run_name,
        "optimizer": cfg.optimizer,
        "lr": lr,
        "r": r,
        "steps": int(last["step"]),
        "final_loss": final_loss,
        "best_loss": min(losses),
        "avg_step_time": sum(float(x["step_time"]) for x in rows) / len(rows),
        "peak_memory_mb": max(float(x["peak_memory_mb"]) for x in rows),
        "diverged": diverged,
        "log_file": log_file,
    }


def write_outputs(out_csv, out_md, results):
    fieldnames = [
        "run",
        "optimizer",
        "lr",
        "r",
        "steps",
        "final_loss",
        "best_loss",
        "avg_step_time",
        "peak_memory_mb",
        "diverged",
        "log_file",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)

    lines = []
    lines.append("# Sweep Result (lr, r)")
    lines.append("")
    lines.append(
        "| run | lr | r | final_loss | best_loss | avg_step_time | peak_memory_mb | diverged |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['run']} | {r['lr']:.6g} | {r['r']:.4g} | {r['final_loss']:.6f} | {r['best_loss']:.6f} | "
            f"{r['avg_step_time']:.6f} | {r['peak_memory_mb']:.2f} | {r['diverged']} |"
        )

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="rmsprop_pnorm")
    parser.add_argument("--lrs", type=str, default="1e-4,3e-4,1e-3")
    parser.add_argument("--rs", type=str, default="1.0,1.5,2.0,3.0")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diverge_loss", type=float, default=1e6)
    parser.add_argument("--log_dir", type=str, default="logs/sweeps")
    parser.add_argument("--out_dir", type=str, default="results/sweeps")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    lrs = parse_float_list(args.lrs)
    rs = parse_float_list(args.rs)

    all_results = []
    for lr in lrs:
        for r in rs:
            result = run_one(args, lr, r)
            all_results.append(result)

    all_results.sort(key=lambda x: (x["diverged"], x["final_loss"]))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.out_dir, f"sweep_lr_r_{ts}.csv")
    out_md = os.path.join(args.out_dir, f"sweep_lr_r_{ts}.md")
    write_outputs(out_csv, out_md, all_results)

    best = all_results[0]
    print("[done]", out_csv)
    print("[done]", out_md)
    print(
        f"[best] lr={best['lr']:.6g} r={best['r']:.4g} final_loss={best['final_loss']:.6f} "
        f"diverged={best['diverged']}"
    )


if __name__ == "__main__":
    main()
