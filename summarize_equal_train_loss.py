import argparse
import csv
import glob
import math
import os
from statistics import mean, stdev


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_int(value, default=-1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def is_finite(x):
    return not math.isnan(x) and not math.isinf(x)


def mean_or_nan(values):
    finite = [v for v in values if is_finite(v)]
    if not finite:
        return float("nan")
    return mean(finite)


def std_or_nan(values):
    finite = [v for v in values if is_finite(v)]
    if not finite:
        return float("nan")
    if len(finite) == 1:
        return 0.0
    return stdev(finite)


def load_one_log(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, -1, []

    optimizer = rows[0].get("optimizer", "")
    seed = parse_int(rows[0].get("seed"), default=-1)

    parsed = []
    for r in rows:
        parsed.append(
            {
                "step": int(r["step"]),
                "train_loss": parse_float(r.get("loss")),
                "test_loss": parse_float(r.get("test_loss")),
                "test_ece": parse_float(r.get("test_ece")),
                "test_acc": parse_float(r.get("test_acc")),
                "param_norm_l2": parse_float(r.get("param_norm_l2")),
                "generalization_gap": parse_float(r.get("generalization_gap")),
            }
        )
    return optimizer, seed, parsed


def load_logs(log_dir, pattern):
    grouped = {}
    for path in sorted(glob.glob(os.path.join(log_dir, pattern))):
        optimizer, seed, rows = load_one_log(path)
        if not rows:
            continue
        grouped.setdefault(optimizer, []).append({"seed": seed, "path": path, "rows": rows})
    return grouped


def common_train_loss_range(grouped_runs):
    mins = []
    maxs = []
    for runs in grouped_runs.values():
        train_losses = []
        for run in runs:
            train_losses.extend(r["train_loss"] for r in run["rows"] if is_finite(r["train_loss"]))
        if not train_losses:
            continue
        mins.append(min(train_losses))
        maxs.append(max(train_losses))
    if not mins or not maxs:
        return float("nan"), float("nan")
    low = max(mins)
    high = min(maxs)
    return low, high


def build_anchors(low, high, n_anchors):
    if n_anchors <= 1 or math.isclose(low, high):
        return [(low + high) / 2.0]
    return [high - i * (high - low) / (n_anchors - 1) for i in range(n_anchors)]


def nearest_by_train_loss(rows, anchor):
    valid = [r for r in rows if is_finite(r["train_loss"])]
    if not valid:
        return None
    return min(valid, key=lambda r: abs(r["train_loss"] - anchor))


def fmt(x, digits=6):
    if not is_finite(x):
        return ""
    return f"{x:.{digits}f}"


def aggregate_for_anchor(runs, anchor):
    per_seed = []
    for run in runs:
        hit = nearest_by_train_loss(run["rows"], anchor)
        if hit is None:
            continue
        per_seed.append(
            {
                "seed": run["seed"],
                "matched_step": float(hit["step"]),
                "matched_train_loss": hit["train_loss"],
                "abs_train_loss_error": abs(hit["train_loss"] - anchor),
                "test_loss": hit["test_loss"],
                "test_ece": hit["test_ece"],
                "test_acc": hit["test_acc"],
                "param_norm_l2": hit["param_norm_l2"],
                "generalization_gap": hit["generalization_gap"],
            }
        )

    if not per_seed:
        return None

    seed_ids = sorted(s["seed"] for s in per_seed)
    return {
        "n_seeds": len(per_seed),
        "seed_ids": ",".join(str(s) for s in seed_ids),
        "matched_step": mean_or_nan([s["matched_step"] for s in per_seed]),
        "matched_step_std": std_or_nan([s["matched_step"] for s in per_seed]),
        "matched_train_loss": mean_or_nan([s["matched_train_loss"] for s in per_seed]),
        "matched_train_loss_std": std_or_nan([s["matched_train_loss"] for s in per_seed]),
        "abs_train_loss_error": mean_or_nan([s["abs_train_loss_error"] for s in per_seed]),
        "abs_train_loss_error_std": std_or_nan([s["abs_train_loss_error"] for s in per_seed]),
        "test_loss": mean_or_nan([s["test_loss"] for s in per_seed]),
        "test_loss_std": std_or_nan([s["test_loss"] for s in per_seed]),
        "test_ece": mean_or_nan([s["test_ece"] for s in per_seed]),
        "test_ece_std": std_or_nan([s["test_ece"] for s in per_seed]),
        "test_acc": mean_or_nan([s["test_acc"] for s in per_seed]),
        "test_acc_std": std_or_nan([s["test_acc"] for s in per_seed]),
        "param_norm_l2": mean_or_nan([s["param_norm_l2"] for s in per_seed]),
        "param_norm_l2_std": std_or_nan([s["param_norm_l2"] for s in per_seed]),
        "generalization_gap": mean_or_nan([s["generalization_gap"] for s in per_seed]),
        "generalization_gap_std": std_or_nan([s["generalization_gap"] for s in per_seed]),
    }


def build_matches(grouped_runs, anchors):
    matches = []
    for anchor_idx, anchor in enumerate(anchors, start=1):
        for optimizer, runs in sorted(grouped_runs.items()):
            agg = aggregate_for_anchor(runs, anchor)
            if agg is None:
                continue
            matches.append(
                {
                    "anchor_id": anchor_idx,
                    "anchor_train_loss": anchor,
                    "optimizer": optimizer,
                    **agg,
                }
            )
    return matches


def build_summary(grouped_runs, matches):
    summary = []
    for optimizer in sorted(grouped_runs.keys()):
        rows = [r for r in matches if r["optimizer"] == optimizer]
        if not rows:
            continue
        summary.append(
            {
                "optimizer": optimizer,
                "num_matches": len(rows),
                "num_seeds": min(r["n_seeds"] for r in rows),
                "mean_test_loss": mean_or_nan([r["test_loss"] for r in rows]),
                "mean_test_ece": mean_or_nan([r["test_ece"] for r in rows]),
                "mean_test_acc": mean_or_nan([r["test_acc"] for r in rows]),
                "mean_param_norm_l2": mean_or_nan([r["param_norm_l2"] for r in rows]),
                "mean_generalization_gap": mean_or_nan([r["generalization_gap"] for r in rows]),
                "mean_abs_train_loss_error": mean_or_nan([r["abs_train_loss_error"] for r in rows]),
                "avg_seed_std_test_loss": mean_or_nan([r["test_loss_std"] for r in rows]),
                "avg_seed_std_test_ece": mean_or_nan([r["test_ece_std"] for r in rows]),
                "avg_seed_std_param_norm_l2": mean_or_nan([r["param_norm_l2_std"] for r in rows]),
            }
        )
    summary.sort(key=lambda r: (float("inf") if not is_finite(r["mean_test_loss"]) else r["mean_test_loss"]))
    return summary


def save_matches_csv(path, matches):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "anchor_id",
                "anchor_train_loss",
                "optimizer",
                "n_seeds",
                "seed_ids",
                "matched_step",
                "matched_step_std",
                "matched_train_loss",
                "matched_train_loss_std",
                "abs_train_loss_error",
                "abs_train_loss_error_std",
                "test_loss",
                "test_loss_std",
                "test_ece",
                "test_ece_std",
                "test_acc",
                "test_acc_std",
                "param_norm_l2",
                "param_norm_l2_std",
                "generalization_gap",
                "generalization_gap_std",
            ],
        )
        writer.writeheader()
        for r in matches:
            writer.writerow(
                {
                    "anchor_id": r["anchor_id"],
                    "anchor_train_loss": fmt(r["anchor_train_loss"]),
                    "optimizer": r["optimizer"],
                    "n_seeds": r["n_seeds"],
                    "seed_ids": r["seed_ids"],
                    "matched_step": fmt(r["matched_step"]),
                    "matched_step_std": fmt(r["matched_step_std"]),
                    "matched_train_loss": fmt(r["matched_train_loss"]),
                    "matched_train_loss_std": fmt(r["matched_train_loss_std"]),
                    "abs_train_loss_error": fmt(r["abs_train_loss_error"]),
                    "abs_train_loss_error_std": fmt(r["abs_train_loss_error_std"]),
                    "test_loss": fmt(r["test_loss"]),
                    "test_loss_std": fmt(r["test_loss_std"]),
                    "test_ece": fmt(r["test_ece"]),
                    "test_ece_std": fmt(r["test_ece_std"]),
                    "test_acc": fmt(r["test_acc"]),
                    "test_acc_std": fmt(r["test_acc_std"]),
                    "param_norm_l2": fmt(r["param_norm_l2"]),
                    "param_norm_l2_std": fmt(r["param_norm_l2_std"]),
                    "generalization_gap": fmt(r["generalization_gap"]),
                    "generalization_gap_std": fmt(r["generalization_gap_std"]),
                }
            )


def save_summary_csv(path, summary):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "optimizer",
                "num_matches",
                "num_seeds",
                "mean_test_loss",
                "mean_test_ece",
                "mean_test_acc",
                "mean_param_norm_l2",
                "mean_generalization_gap",
                "mean_abs_train_loss_error",
                "avg_seed_std_test_loss",
                "avg_seed_std_test_ece",
                "avg_seed_std_param_norm_l2",
            ],
        )
        writer.writeheader()
        for r in summary:
            writer.writerow(
                {
                    "optimizer": r["optimizer"],
                    "num_matches": r["num_matches"],
                    "num_seeds": r["num_seeds"],
                    "mean_test_loss": fmt(r["mean_test_loss"]),
                    "mean_test_ece": fmt(r["mean_test_ece"]),
                    "mean_test_acc": fmt(r["mean_test_acc"]),
                    "mean_param_norm_l2": fmt(r["mean_param_norm_l2"]),
                    "mean_generalization_gap": fmt(r["mean_generalization_gap"]),
                    "mean_abs_train_loss_error": fmt(r["mean_abs_train_loss_error"]),
                    "avg_seed_std_test_loss": fmt(r["avg_seed_std_test_loss"]),
                    "avg_seed_std_test_ece": fmt(r["avg_seed_std_test_ece"]),
                    "avg_seed_std_param_norm_l2": fmt(r["avg_seed_std_param_norm_l2"]),
                }
            )


def save_summary_md(path, grouped_runs, low, high, anchors, summary, matches):
    lines = [
        "# Equal-Train-Loss Generalization Summary",
        "",
        f"- compared_optimizers: {len(grouped_runs)}",
        f"- train_loss_overlap: [{low:.6f}, {high:.6f}]",
        f"- num_anchors: {len(anchors)}",
        "",
        "| optimizer | num_matches | num_seeds | mean_test_loss | mean_test_ece | mean_test_acc | mean_param_norm_l2 | mean_gen_gap | mean_abs_train_loss_error | avg_seed_std_test_loss | avg_seed_std_test_ece |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['optimizer']} | {r['num_matches']} | {r['num_seeds']} | {fmt(r['mean_test_loss'])} | "
            f"{fmt(r['mean_test_ece'])} | {fmt(r['mean_test_acc'])} | {fmt(r['mean_param_norm_l2'])} | "
            f"{fmt(r['mean_generalization_gap'])} | {fmt(r['mean_abs_train_loss_error'])} | "
            f"{fmt(r['avg_seed_std_test_loss'])} | {fmt(r['avg_seed_std_test_ece'])} |"
        )

    lines.extend(
        [
            "",
            "## Matched Points",
            "",
            "| anchor_id | anchor_train_loss | optimizer | n_seeds | matched_step | matched_step_std | matched_train_loss | abs_train_loss_error | test_loss | test_loss_std | test_ece | test_ece_std | test_acc | param_norm_l2 |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in sorted(matches, key=lambda x: (x["anchor_id"], x["optimizer"])):
        lines.append(
            f"| {r['anchor_id']} | {fmt(r['anchor_train_loss'])} | {r['optimizer']} | {r['n_seeds']} | "
            f"{fmt(r['matched_step'])} | {fmt(r['matched_step_std'])} | {fmt(r['matched_train_loss'])} | "
            f"{fmt(r['abs_train_loss_error'])} | {fmt(r['test_loss'])} | {fmt(r['test_loss_std'])} | "
            f"{fmt(r['test_ece'])} | {fmt(r['test_ece_std'])} | {fmt(r['test_acc'])} | {fmt(r['param_norm_l2'])} |"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs/real_text/fixed_1000")
    parser.add_argument("--pattern", type=str, default="*.csv")
    parser.add_argument("--out_dir", type=str, default="results/real_text/fixed_1000")
    parser.add_argument("--n_anchors", type=int, default=7)
    args = parser.parse_args()
    args.n_anchors = max(2, args.n_anchors)

    os.makedirs(args.out_dir, exist_ok=True)
    grouped_runs = load_logs(args.log_dir, args.pattern)
    if not grouped_runs:
        raise RuntimeError(f"No logs found in {args.log_dir} with pattern {args.pattern}")

    low, high = common_train_loss_range(grouped_runs)
    if not is_finite(low) or not is_finite(high) or low > high:
        raise RuntimeError(
            "No overlapping train-loss range across optimizers. "
            "Cannot compare at equal train loss."
        )

    anchors = build_anchors(low, high, args.n_anchors)
    matches = build_matches(grouped_runs, anchors)
    if not matches:
        raise RuntimeError("No valid matched points were found.")

    summary = build_summary(grouped_runs, matches)

    match_csv = os.path.join(args.out_dir, "equal_train_loss_matches.csv")
    summary_csv = os.path.join(args.out_dir, "equal_train_loss_summary.csv")
    summary_md = os.path.join(args.out_dir, "equal_train_loss_summary.md")

    save_matches_csv(match_csv, matches)
    save_summary_csv(summary_csv, summary)
    save_summary_md(summary_md, grouped_runs, low, high, anchors, summary, matches)

    print(f"saved: {match_csv}")
    print(f"saved: {summary_csv}")
    print(f"saved: {summary_md}")


if __name__ == "__main__":
    main()
