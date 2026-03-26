import csv
import os
import subprocess
import sys
from datetime import datetime


LR_CANDIDATES = {
    "adamw": [1e-4, 3e-4, 1e-3],
    "sgd": [1e-3, 3e-3, 1e-2],
    "rmsprop": [1e-4, 3e-4, 1e-3],
    "adafactor": [1e-3, 3e-3, 1e-2],
}


def safe(v):
    return f"{v:g}".replace(".", "p")


def read_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty log: {path}")
    return rows


def run_one(optimizer, lr, steps, seed, log_dir):
    log_file = os.path.join(log_dir, f"{optimizer}_lr{safe(lr)}.csv")
    cmd = [
        sys.executable,
        "train_real_text.py",
        "--optimizer",
        optimizer,
        "--lr",
        str(lr),
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--log_file",
        log_file,
    ]
    print(f"[run] optimizer={optimizer} lr={lr:g}")
    subprocess.run(cmd, check=True)

    rows = read_rows(log_file)
    final = rows[-1]
    step_time_avg = sum(float(r["step_time"]) for r in rows) / len(rows)

    return {
        "optimizer": optimizer,
        "lr": lr,
        "final_loss": float(final["loss"]),
        "step_time": step_time_avg,
        "log_file": log_file,
    }


def main():
    steps = 300
    seed = 42

    root = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(root, "logs", "real_text_sweep")
    out_dir = os.path.join(root, "results", "real_text_sweep")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []
    for opt, lrs in LR_CANDIDATES.items():
        for lr in lrs:
            all_rows.append(run_one(opt, lr, steps, seed, log_dir))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(out_dir, f"real_text_lr_sweep_{ts}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["optimizer", "lr", "final_loss", "step_time", "log_file"])
        w.writeheader()
        for r in all_rows:
            w.writerow(
                {
                    "optimizer": r["optimizer"],
                    "lr": f"{r['lr']:.6g}",
                    "final_loss": f"{r['final_loss']:.6f}",
                    "step_time": f"{r['step_time']:.6f}",
                    "log_file": r["log_file"],
                }
            )

    print(f"[done] {out_csv}")

    for opt in LR_CANDIDATES.keys():
        cand = [r for r in all_rows if r["optimizer"] == opt]
        best = sorted(cand, key=lambda x: x["final_loss"])[0]
        print(
            f"[recommend] optimizer={opt} lr={best['lr']:.6g} "
            f"final_loss={best['final_loss']:.6f} step_time={best['step_time']:.6f}"
        )


if __name__ == "__main__":
    main()
