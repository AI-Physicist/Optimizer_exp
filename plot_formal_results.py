import argparse
import math
import os

from summarize_results import load_logs, make_svg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--log_dir", type=str, default="logs/synthetic/formal")
    parser.add_argument(
        "--output",
        type=str,
        default="results/synthetic/formal/formal_optimizer_comparison.svg",
    )
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(root, args.log_dir)
    out_path = os.path.join(root, args.output)

    data = load_logs(log_dir)
    if not data:
        raise RuntimeError(f"No logs found in {log_dir}")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    make_svg(data, out_path, loss_theoretical_floor=math.log(args.vocab_size))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
