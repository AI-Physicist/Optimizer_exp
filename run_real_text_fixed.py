import argparse
import csv
import os
import subprocess
import sys

from summarize_results import build_summary, load_log_paths, make_svg, save_summary_csv, save_summary_md


LR_MAP = {
    "adamw": 1e-3,
    "sgd": 1e-2,
    "rmsprop": 1e-4,
    "adafactor": 1e-2,
}


def safe(v):
    return f"{v:g}".replace(".", "p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="logs/real_text/fixed_1000")
    parser.add_argument("--out_dir", type=str, default="results/real_text/fixed_1000")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    produced_logs = []
    for optimizer, lr in LR_MAP.items():
        run_name = f"{optimizer}_lr{safe(lr)}_{args.steps}"
        log_file = os.path.join(args.log_dir, f"{run_name}.csv")
        produced_logs.append(log_file)
        cmd = [
            sys.executable,
            "train_real_text.py",
            "--optimizer",
            optimizer,
            "--lr",
            str(lr),
            "--steps",
            str(args.steps),
            "--batch_size",
            str(args.batch_size),
            "--seq_len",
            str(args.seq_len),
            "--seed",
            str(args.seed),
            "--log_file",
            log_file,
        ]
        print(f"[run] optimizer={optimizer} lr={lr:g} steps={args.steps}")
        subprocess.run(cmd, check=True)

    data = load_log_paths(produced_logs)
    summary = build_summary(data)
    save_summary_csv(os.path.join(args.out_dir, "summary.csv"), summary)
    save_summary_md(os.path.join(args.out_dir, "summary.md"), summary)
    make_svg(data, os.path.join(args.out_dir, "optimizer_comparison.svg"), loss_theoretical_floor=None)

    runs_csv = os.path.join(args.out_dir, "runs.csv")
    with open(runs_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["optimizer", "lr", "seed", "steps", "log_file"],
        )
        writer.writeheader()
        for optimizer, lr in LR_MAP.items():
            writer.writerow(
                {
                    "optimizer": optimizer,
                    "lr": f"{lr:.6g}",
                    "seed": args.seed,
                    "steps": args.steps,
                    "log_file": os.path.join(
                        args.log_dir, f"{optimizer}_lr{safe(lr)}_{args.steps}.csv"
                    ),
                }
            )

    print(f"saved: {runs_csv}")
    print(f"saved: {os.path.join(args.out_dir, 'summary.csv')}")
    print(f"saved: {os.path.join(args.out_dir, 'summary.md')}")
    print(f"saved: {os.path.join(args.out_dir, 'optimizer_comparison.svg')}")


if __name__ == "__main__":
    main()
