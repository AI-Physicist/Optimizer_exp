import argparse
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from train import (
    DecoderOnlyTransformer,
    build_optimizer,
    capture_pre_step_state,
    compute_param_norm_l2,
    compute_update_metrics,
    count_parameters,
    estimate_top_hessian_eig,
    get_peak_memory_mb,
    get_trainable_params,
)


def load_corpus(file_list):
    texts = []
    for path in file_list:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            texts.append(f.read())
    merged = "\n\n".join(texts)
    return torch.tensor(list(merged.encode("utf-8")), dtype=torch.long)


def split_train_test(data, train_split, seq_len):
    if not (0.0 < train_split < 1.0):
        raise ValueError("train_split must be in (0, 1).")
    split_idx = int(data.numel() * train_split)
    min_tokens = seq_len + 2
    if split_idx < min_tokens or (data.numel() - split_idx) < min_tokens:
        raise RuntimeError(
            "Corpus is too short for train/test split with current seq_len. "
            f"Need at least {min_tokens} tokens on each side."
        )
    return data[:split_idx], data[split_idx:]


def make_batch_from_starts(data, starts, seq_len, device):
    x = torch.stack([data[s : s + seq_len] for s in starts], dim=0)
    y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts], dim=0)
    return x.to(device), y.to(device)


def sample_batch(data, batch_size, seq_len, device):
    max_start = data.numel() - seq_len - 1
    if max_start <= 0:
        raise RuntimeError("Corpus is too short for the configured seq_len.")

    starts = torch.randint(0, max_start, (batch_size,))
    return make_batch_from_starts(data, starts, seq_len, device)


def build_eval_starts(data, seq_len, total_windows, seed):
    max_start = data.numel() - seq_len - 1
    if max_start <= 0:
        raise RuntimeError("Test split is too short for the configured seq_len.")
    if total_windows <= 0:
        return torch.empty(0, dtype=torch.long)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randint(0, max_start, (total_windows,), generator=gen)


def compute_ece(bin_counts, bin_conf_sum, bin_acc_sum):
    total = float(bin_counts.sum().item())
    if total <= 0.0:
        return float("nan")

    ece = 0.0
    for i in range(bin_counts.numel()):
        count = float(bin_counts[i].item())
        if count <= 0.0:
            continue
        acc = float(bin_acc_sum[i].item()) / count
        conf = float(bin_conf_sum[i].item()) / count
        ece += abs(acc - conf) * (count / total)
    return ece


@torch.no_grad()
def evaluate_model(model, data, batch_size, seq_len, vocab_size, device, eval_starts, ece_bins):
    if eval_starts.numel() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    bin_counts = torch.zeros(ece_bins, dtype=torch.float64)
    bin_conf_sum = torch.zeros(ece_bins, dtype=torch.float64)
    bin_acc_sum = torch.zeros(ece_bins, dtype=torch.float64)

    for start in range(0, eval_starts.numel(), batch_size):
        starts = eval_starts[start : start + batch_size]
        x, y = make_batch_from_starts(data, starts, seq_len, device)
        logits = model(x)

        flat_logits = logits.reshape(-1, vocab_size)
        flat_targets = y.reshape(-1)

        total_loss += F.cross_entropy(flat_logits, flat_targets, reduction="sum").item()
        total_tokens += flat_targets.numel()

        probs = torch.softmax(flat_logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        correct = pred.eq(flat_targets)
        total_correct += int(correct.sum().item())

        bin_idx = torch.clamp((conf * ece_bins).to(torch.long), max=ece_bins - 1).cpu()
        bin_counts.scatter_add_(0, bin_idx, torch.ones_like(bin_idx, dtype=torch.float64))
        bin_conf_sum.scatter_add_(0, bin_idx, conf.cpu().to(torch.float64))
        bin_acc_sum.scatter_add_(0, bin_idx, correct.cpu().to(torch.float64))

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 20.0))
    acc = total_correct / max(1, total_tokens)
    ece = compute_ece(bin_counts, bin_conf_sum, bin_acc_sum)
    return avg_loss, ppl, acc, ece


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_files", type=str, default="data/data.txt")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "sgd", "rmsprop", "rmsprop_no_memory", "rmsprop_pnorm", "adafactor"],
    )
    parser.add_argument("--rmsprop_p", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--curvature_every", type=int, default=50)
    parser.add_argument("--hessian_power_iters", type=int, default=5)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--eval_batches", type=int, default=8)
    parser.add_argument("--ece_bins", type=int, default=15)
    parser.add_argument("--log_file", type=str, default="logs/real_text/real_text.csv")
    args = parser.parse_args()
    args.log_every = max(1, args.log_every)
    args.curvature_every = max(0, args.curvature_every)
    args.hessian_power_iters = max(1, args.hessian_power_iters)
    args.eval_batches = max(0, args.eval_batches)
    args.ece_bins = max(2, args.ece_bins)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_files = [x.strip() for x in args.text_files.split(",") if x.strip()]
    data = load_corpus(text_files)
    train_data, test_data = split_train_test(data, args.train_split, args.seq_len)
    vocab_size = 256
    eval_starts = build_eval_starts(
        test_data,
        seq_len=args.seq_len,
        total_windows=args.eval_batches * args.batch_size,
        seed=args.seed + 12345,
    )

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        max_seq_len=args.seq_len,
        dim=256,
        num_layers=2,
        num_heads=8,
        mlp_ratio=4,
    ).to(device)

    optimizer = build_optimizer(
        args.optimizer, model, args.lr, args.weight_decay, args.rmsprop_p
    )
    trainable_params = get_trainable_params(model)

    total_params = count_parameters(model)
    print(f"model_parameter_count: {total_params} ({total_params / 1e6:.2f}M)")
    print(f"device: {device}")
    print(f"optimizer: {args.optimizer}")
    print(f"batch_size: {args.batch_size}")
    print(f"sequence_length: {args.seq_len}")
    print(f"max_steps: {args.steps}")
    print(f"seed: {args.seed}")
    if args.optimizer == "rmsprop_pnorm":
        print(f"rmsprop_p: {args.rmsprop_p}")
    print(f"corpus_files: {text_files}")
    print(f"corpus_bytes: {data.numel()}")
    print(f"train_bytes: {train_data.numel()} test_bytes: {test_data.numel()}")
    print(f"train_split: {args.train_split:.3f}")
    print(f"eval_windows: {eval_starts.numel()} ece_bins: {args.ece_bins}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    log_dir = os.path.dirname(args.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with open(args.log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "loss",
                "lr",
                "step_time",
                "optimizer",
                "seed",
                "peak_memory_mb",
                "param_norm_l2",
                "grad_norm_l2",
                "update_norm_l2",
                "update_ratio",
                "grad_update_cos",
                "hessian_top_eig",
                "test_loss",
                "test_ppl",
                "test_acc",
                "test_ece",
                "generalization_gap",
            ],
        )
        writer.writeheader()

        model.train()
        running_loss = 0.0
        running_step_time = 0.0

        progress = trange(1, args.steps + 1, desc="train_real_text", leave=True)
        for step in progress:
            t0 = time.perf_counter()
            should_log = (step % args.log_every == 0) or (step == args.steps)
            should_curvature = (
                args.curvature_every > 0
                and ((step % args.curvature_every == 0) or (step == args.steps))
            )

            x, y = sample_batch(train_data, args.batch_size, args.seq_len, device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            param_before = None
            grad_before = None
            grad_norm = float("nan")
            if should_log:
                param_before, grad_before, grad_norm = capture_pre_step_state(trainable_params)

            optimizer.step()

            step_time = time.perf_counter() - t0
            running_step_time += step_time
            running_loss += loss.item()

            if should_log:
                window = args.log_every if step % args.log_every == 0 else (step % args.log_every)
                avg_loss = running_loss / window
                avg_step_time = running_step_time / window
                lr = optimizer.param_groups[0]["lr"]
                peak_mb = get_peak_memory_mb(device)
                param_norm = compute_param_norm_l2(trainable_params)
                update_norm = float("nan")
                grad_update_cos = float("nan")
                if param_before is not None and grad_before is not None:
                    update_norm, grad_update_cos = compute_update_metrics(
                        trainable_params, param_before, grad_before
                    )
                update_ratio = update_norm / (param_norm + 1e-12)

                hessian_top_eig = float("nan")
                if should_curvature:
                    optimizer.zero_grad(set_to_none=True)
                    hessian_top_eig = estimate_top_hessian_eig(
                        model=model,
                        params=trainable_params,
                        x=x,
                        y=y,
                        vocab_size=vocab_size,
                        power_iters=args.hessian_power_iters,
                    )
                    optimizer.zero_grad(set_to_none=True)

                test_loss = float("nan")
                test_ppl = float("nan")
                test_acc = float("nan")
                test_ece = float("nan")
                if args.eval_batches > 0:
                    test_loss, test_ppl, test_acc, test_ece = evaluate_model(
                        model=model,
                        data=test_data,
                        batch_size=args.batch_size,
                        seq_len=args.seq_len,
                        vocab_size=vocab_size,
                        device=device,
                        eval_starts=eval_starts,
                        ece_bins=args.ece_bins,
                    )
                    model.train()

                generalization_gap = test_loss - avg_loss

                print(
                    f"step={step} loss={avg_loss:.4f} lr={lr:.6g} "
                    f"grad_norm={grad_norm:.4e} update_ratio={update_ratio:.4e} "
                    f"test_loss={test_loss:.4f} test_ece={test_ece:.4e} "
                    f"hessian_top={hessian_top_eig:.4e} step_time={avg_step_time:.4f}s"
                )

                writer.writerow(
                    {
                        "step": step,
                        "loss": f"{avg_loss:.6f}",
                        "lr": f"{lr:.8g}",
                        "step_time": f"{avg_step_time:.6f}",
                        "optimizer": args.optimizer,
                        "seed": args.seed,
                        "peak_memory_mb": f"{peak_mb:.4f}",
                        "param_norm_l2": f"{param_norm:.8e}",
                        "grad_norm_l2": f"{grad_norm:.8e}",
                        "update_norm_l2": f"{update_norm:.8e}",
                        "update_ratio": f"{update_ratio:.8e}",
                        "grad_update_cos": f"{grad_update_cos:.8e}",
                        "hessian_top_eig": f"{hessian_top_eig:.8e}",
                        "test_loss": f"{test_loss:.8e}",
                        "test_ppl": f"{test_ppl:.8e}",
                        "test_acc": f"{test_acc:.8e}",
                        "test_ece": f"{test_ece:.8e}",
                        "generalization_gap": f"{generalization_gap:.8e}",
                    }
                )
                f.flush()

                running_loss = 0.0
                running_step_time = 0.0

    print(f"log_file: {args.log_file}")
    print(f"peak_memory_mb: {get_peak_memory_mb(device):.2f}")


if __name__ == "__main__":
    main()
