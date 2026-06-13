from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from recommender_gan.data import NextItemDataset, load_history
from recommender_gan.metrics import evaluate_generator
from recommender_gan.models import Generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained GAN recommender.")
    parser.add_argument("--data_path", default="data/processed/test_history.pkl")
    parser.add_argument("--checkpoint_path", default="outputs/checkpoints/gan_generator_best.pth")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_path", default="outputs/logs/gan_test_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    generator = Generator(
        action_dim=checkpoint["action_dim"],
        state_size=checkpoint["state_size"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        noise_dim=checkpoint["noise_dim"],
    ).to(device)
    generator.load_state_dict(checkpoint["model_state_dict"])

    history = load_history(args.data_path)
    dataset = NextItemDataset(history, checkpoint["state_size"])
    metrics = evaluate_generator(generator, dataset, args.top_k, args.batch_size, device)

    row = {
        "model": "GAN",
        "model_path": args.checkpoint_path,
        f"hit_rate_at_{args.top_k}": metrics.hit_rate_at_k,
        f"precision_at_{args.top_k}": metrics.precision_at_k,
        f"recall_at_{args.top_k}": metrics.recall_at_k,
        f"ndcg_at_{args.top_k}": metrics.ndcg_at_k,
        f"mrr_at_{args.top_k}": metrics.mrr_at_k,
        "samples": metrics.samples,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output_path, index=False)

    print(pd.DataFrame([row]).to_string(index=False))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
