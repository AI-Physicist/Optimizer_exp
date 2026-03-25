import argparse
import csv
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from tqdm import trange


BASELINE_CONFIG = {
    "steps": 1000,
    "batch_size": 16,
    "seq_len": 128,
    "vocab_size": 32000,
    "lr": 3e-4,
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "seed": 42,
}


class Adafactor(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        beta2=0.999,
        eps1=1e-30,
        eps2=1e-3,
        clip_threshold=1.0,
        weight_decay=0.0,
    ):
        defaults = dict(
            lr=lr,
            beta2=beta2,
            eps1=eps1,
            eps2=eps2,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta2 = group["beta2"]
            eps1 = group["eps1"]
            eps2 = group["eps2"]
            clip_threshold = group["clip_threshold"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    if grad.ndim >= 2:
                        state["exp_avg_sq_row"] = torch.zeros(
                            grad.shape[:-1], dtype=torch.float32, device=grad.device
                        )
                        state["exp_avg_sq_col"] = torch.zeros(
                            grad.shape[:-2] + grad.shape[-1:],
                            dtype=torch.float32,
                            device=grad.device,
                        )
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad, dtype=torch.float32)

                state["step"] += 1
                grad_f32 = grad.float()

                if grad.ndim >= 2:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    grad_sq = grad_f32.pow(2) + eps1
                    exp_avg_sq_row.mul_(beta2).add_((1.0 - beta2) * grad_sq.mean(dim=-1))
                    exp_avg_sq_col.mul_(beta2).add_((1.0 - beta2) * grad_sq.mean(dim=-2))

                    r_factor = (
                        exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)
                    ).rsqrt()
                    c_factor = exp_avg_sq_col.rsqrt()
                    v = torch.mul(r_factor.unsqueeze(-1), c_factor.unsqueeze(-2))
                    update = grad_f32 * v
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).add_((1.0 - beta2) * (grad_f32.pow(2) + eps1))
                    update = grad_f32 / exp_avg_sq.sqrt()

                rms = update.pow(2).mean().sqrt()
                update = update / max(1.0, (rms / clip_threshold).item())
                update = update / (update.pow(2).mean().sqrt() + eps2)

                if weight_decay != 0.0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
                p.data.add_(update.to(p.dtype), alpha=-lr)
        return loss


class RMSpropPNorm(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        alpha=0.99,
        p=2.0,
        eps=1e-8,
        weight_decay=0.0,
    ):
        if p <= 0.0:
            raise ValueError("p must be > 0 for RMSpropPNorm.")
        defaults = dict(lr=lr, alpha=alpha, p=p, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            norm_p = group["p"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("RMSpropPNorm does not support sparse gradients.")

                state = self.state[param]
                if len(state) == 0:
                    state["square_avg"] = torch.zeros_like(grad, dtype=torch.float32)

                grad_f32 = grad.float()
                if weight_decay != 0.0:
                    grad_f32 = grad_f32.add(param.data.float(), alpha=weight_decay)

                square_avg = state["square_avg"]
                square_avg.mul_(alpha).add_((1.0 - alpha) * grad_f32.abs().pow(norm_p))

                denom = (square_avg + eps).pow(1.0 / norm_p)
                update = grad_f32 / denom
                param.data.add_(update.to(param.dtype), alpha=-lr)

        return loss


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seqlen, seqlen, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, dim)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        max_seq_len: int = 128,
        dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = idx.shape
        pos = torch.arange(seqlen, device=idx.device).unsqueeze(0).expand(bsz, seqlen)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


@dataclass
class TrainConfig:
    steps: int = BASELINE_CONFIG["steps"]
    batch_size: int = BASELINE_CONFIG["batch_size"]
    seq_len: int = BASELINE_CONFIG["seq_len"]
    vocab_size: int = BASELINE_CONFIG["vocab_size"]
    lr: float = BASELINE_CONFIG["lr"]
    optimizer: str = BASELINE_CONFIG["optimizer"]
    weight_decay: float = BASELINE_CONFIG["weight_decay"]
    seed: int = BASELINE_CONFIG["seed"]
    log_file: str = "train_log.csv"
    rmsprop_p: float = 2.0


def build_optimizer(
    name: str,
    model: nn.Module,
    lr: float,
    weight_decay: float,
    rmsprop_p: float,
):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    if name == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(), lr=lr, alpha=0.99, weight_decay=weight_decay
        )
    if name == "rmsprop_no_memory":
        return torch.optim.RMSprop(
            model.parameters(), lr=lr, alpha=0.0, weight_decay=weight_decay
        )
    if name == "rmsprop_pnorm":
        return RMSpropPNorm(
            model.parameters(),
            lr=lr,
            alpha=0.99,
            p=rmsprop_p,
            weight_decay=weight_decay,
        )
    if name == "adafactor":
        return Adafactor(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def sample_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.roll(x, shifts=-1, dims=1)
    return x, y


def get_peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=BASELINE_CONFIG["steps"])
    parser.add_argument("--batch_size", type=int, default=BASELINE_CONFIG["batch_size"])
    parser.add_argument("--seq_len", type=int, default=BASELINE_CONFIG["seq_len"])
    parser.add_argument("--vocab_size", type=int, default=BASELINE_CONFIG["vocab_size"])
    parser.add_argument("--lr", type=float, default=BASELINE_CONFIG["lr"])
    parser.add_argument(
        "--optimizer",
        type=str,
        default=BASELINE_CONFIG["optimizer"],
        choices=["adamw", "sgd", "rmsprop", "rmsprop_no_memory", "rmsprop_pnorm", "adafactor"],
    )
    parser.add_argument("--weight_decay", type=float, default=BASELINE_CONFIG["weight_decay"])
    parser.add_argument("--seed", type=int, default=BASELINE_CONFIG["seed"])
    parser.add_argument("--rmsprop_p", type=float, default=2.0)
    parser.add_argument("--log_file", type=str, default="train_log.csv")
    args = parser.parse_args()
    cfg = TrainConfig(**vars(args))

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DecoderOnlyTransformer(
        vocab_size=cfg.vocab_size,
        max_seq_len=cfg.seq_len,
        dim=256,
        num_layers=2,
        num_heads=8,
        mlp_ratio=4,
    ).to(device)

    total_params = count_parameters(model)
    optimizer = build_optimizer(
        cfg.optimizer, model, cfg.lr, cfg.weight_decay, cfg.rmsprop_p
    )

    print(f"model_parameter_count: {total_params} ({total_params / 1e6:.2f}M)")
    print(f"device: {device}")
    print(f"optimizer: {cfg.optimizer}")
    print(f"batch_size: {cfg.batch_size}")
    print(f"sequence_length: {cfg.seq_len}")
    print(f"max_steps: {cfg.steps}")
    print(f"seed: {cfg.seed}")
    if cfg.optimizer == "rmsprop_pnorm":
        print(f"rmsprop_p: {cfg.rmsprop_p}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    log_dir = os.path.dirname(cfg.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with open(cfg.log_file, "w", newline="", encoding="utf-8") as f:
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
            ],
        )
        writer.writeheader()

        model.train()
        running_loss = 0.0
        running_step_time = 0.0

        progress = trange(1, cfg.steps + 1, desc="train", leave=True)
        for step in progress:
            t0 = time.perf_counter()

            x, y = sample_batch(cfg.batch_size, cfg.seq_len, cfg.vocab_size, device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step_time = time.perf_counter() - t0
            running_step_time += step_time
            running_loss += loss.item()

            should_log = (step % 10 == 0) or (step == cfg.steps)
            if should_log:
                window = 10 if step % 10 == 0 else (step % 10)
                avg_loss = running_loss / window
                avg_step_time = running_step_time / window
                lr = optimizer.param_groups[0]["lr"]
                peak_mb = get_peak_memory_mb(device)

                print(
                    f"step={step} loss={avg_loss:.4f} lr={lr:.6g} step_time={avg_step_time:.4f}s"
                )

                writer.writerow(
                    {
                        "step": step,
                        "loss": f"{avg_loss:.6f}",
                        "lr": f"{lr:.8g}",
                        "step_time": f"{avg_step_time:.6f}",
                        "optimizer": cfg.optimizer,
                        "seed": cfg.seed,
                        "peak_memory_mb": f"{peak_mb:.4f}",
                    }
                )
                f.flush()

                running_loss = 0.0
                running_step_time = 0.0

    print(f"log_file: {cfg.log_file}")
    print(f"peak_memory_mb: {get_peak_memory_mb(device):.2f}")


if __name__ == "__main__":
    main()
