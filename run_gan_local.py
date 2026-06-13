from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local GAN recommender pipeline.")
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--top_n_items", type=int, default=1000)
    parser.add_argument("--n_users", type=int, default=None)
    parser.add_argument("--state_size", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable

    if not args.skip_preprocess:
        preprocess_cmd = [
            py,
            "preprocess_local.py",
            "--raw_dir",
            args.raw_dir,
            "--top_n_items",
            str(args.top_n_items),
            "--state_size",
            str(args.state_size),
        ]
        if args.n_users is not None:
            preprocess_cmd.extend(["--n_users", str(args.n_users)])
        run(preprocess_cmd)

    run(
        [
            py,
            "train_gan.py",
            "--epochs",
            str(args.epochs),
            "--state_size",
            str(args.state_size),
            "--top_k",
            str(args.top_k),
            "--batch_size",
            str(args.batch_size),
            "--progress_every",
            str(args.progress_every),
            "--device",
            args.device,
        ]
    )
    run(
        [
            py,
            "evaluate_gan.py",
            "--top_k",
            str(args.top_k),
            "--batch_size",
            str(args.batch_size),
            "--device",
            args.device,
        ]
    )
    run([py, "compare_results.py"])


if __name__ == "__main__":
    main()
