import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime


def parse_int_list(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def read_last_row(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty log file: {path}")
    return rows[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--eval_batches", type=int, default=8)
    parser.add_argument("--ece_bins", type=int, default=15)
    parser.add_argument("--log_dir", type=str, default="logs/real_text/multiseed_1000")
    parser.add_argument("--out_dir", type=str, default="results/real_text/multiseed_1000")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # Fixed from short LR sweep.
    lr_map = {
        "adamw": 1e-3,
        "sgd": 1e-2,
        "rmsprop": 1e-4,
        "adafactor": 1e-2,
    }
    seeds = parse_int_list(args.seeds)

    runs = []
    for opt, lr in lr_map.items():
        for seed in seeds:
            run_name = f"{opt}_lr{lr:g}_seed{seed}".replace(".", "p")
            log_file = os.path.join(args.log_dir, f"{run_name}.csv")
            cmd = [
                sys.executable,
                "train_real_text.py",
                "--optimizer",
                opt,
                "--lr",
                str(lr),
                "--steps",
                str(args.steps),
                "--batch_size",
                str(args.batch_size),
                "--seq_len",
                str(args.seq_len),
                "--seed",
                str(seed),
                "--train_split",
                str(args.train_split),
                "--eval_batches",
                str(args.eval_batches),
                "--ece_bins",
                str(args.ece_bins),
                "--log_file",
                log_file,
            ]
            print(f"[run] optimizer={opt} lr={lr:g} seed={seed}")
            subprocess.run(cmd, check=True)

            last = read_last_row(log_file)
            runs.append(
                {
                    "optimizer": opt,
                    "lr": lr,
                    "seed": seed,
                    "final_loss": float(last["loss"]),
                    "step_time": float(last["step_time"]),
                    "log_file": log_file,
                }
            )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.out_dir, f"runs_{ts}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["optimizer", "lr", "seed", "final_loss", "step_time", "log_file"],
        )
        w.writeheader()
        for r in runs:
            w.writerow(
                {
                    "optimizer": r["optimizer"],
                    "lr": f"{r['lr']:.6g}",
                    "seed": r["seed"],
                    "final_loss": f"{r['final_loss']:.6f}",
                    "step_time": f"{r['step_time']:.6f}",
                    "log_file": r["log_file"],
                }
            )

    subprocess.run(
        [
            sys.executable,
            "summarize_real_text_multiseed.py",
            "--log_dir",
            args.log_dir,
            "--out_dir",
            args.out_dir,
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "summarize_equal_train_loss.py",
            "--log_dir",
            args.log_dir,
            "--out_dir",
            args.out_dir,
            "--pattern",
            "*.csv",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "plot_equal_train_loss.py",
            "--matches_csv",
            os.path.join(args.out_dir, "equal_train_loss_matches.csv"),
            "--out_metrics_svg",
            os.path.join(args.out_dir, "equal_train_loss_metrics.svg"),
            "--out_final_anchor_svg",
            os.path.join(args.out_dir, "equal_train_loss_final_anchor.svg"),
        ],
        check=True,
    )

    print(f"[done] {out_csv}")
    print(f"[done] {os.path.join(args.out_dir, 'summary.csv')}")
    print(f"[done] {os.path.join(args.out_dir, 'summary.md')}")
    print(f"[done] {os.path.join(args.out_dir, 'equal_train_loss_summary.csv')}")
    print(f"[done] {os.path.join(args.out_dir, 'equal_train_loss_summary.md')}")
    print(f"[done] {os.path.join(args.out_dir, 'equal_train_loss_metrics.svg')}")
    print(f"[done] {os.path.join(args.out_dir, 'equal_train_loss_final_anchor.svg')}")


if __name__ == "__main__":
    main()
