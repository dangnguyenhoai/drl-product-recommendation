from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from recommender_gan.data import NextItemDataset, infer_action_dim, load_history
from recommender_gan.metrics import evaluate_generator
from recommender_gan.models import Discriminator, Generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GAN recommender locally.")
    parser.add_argument("--train_data_path", default="data/processed/train_history.pkl")
    parser.add_argument("--val_data_path", default="data/processed/val_history.pkl")
    parser.add_argument("--action_dim", type=int, default=None)
    parser.add_argument("--state_size", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--noise_dim", type=int, default=32)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--supervised_weight", type=float, default=1.0)
    parser.add_argument("--adv_weight", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint_path", default="outputs/checkpoints/gan_generator_best.pth")
    parser.add_argument("--log_path", default="outputs/logs/gan_train_log.csv")
    parser.add_argument("--val_result_path", default="outputs/logs/gan_validation_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and device != "cuda":
        print("WARNING: Ban chon --device cuda nhung torch khong thay CUDA, se chay bang CPU.")
    print(f"Using device: {device}")

    train_history = load_history(args.train_data_path)
    val_history = load_history(args.val_data_path)
    action_dim = args.action_dim or infer_action_dim([train_history, val_history])

    train_dataset = NextItemDataset(train_history, args.state_size)
    val_dataset = NextItemDataset(val_history, args.state_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    print(
        f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | "
        f"Batches/epoch: {len(train_loader)} | Batch size: {args.batch_size}"
    )

    generator = Generator(
        action_dim=action_dim,
        state_size=args.state_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        noise_dim=args.noise_dim,
    ).to(device)
    discriminator = Discriminator(
        action_dim=action_dim,
        state_size=args.state_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    best_metrics = None

    with Path(args.log_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "generator_loss",
                "discriminator_loss",
                f"val_hit_rate_at_{args.top_k}",
                f"val_precision_at_{args.top_k}",
                f"val_recall_at_{args.top_k}",
                f"val_ndcg_at_{args.top_k}",
                f"val_mrr_at_{args.top_k}",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            generator.train()
            discriminator.train()
            total_g = 0.0
            total_d = 0.0
            batches = 0

            for batch_idx, (states, real_items) in enumerate(train_loader, start=1):
                states = states.to(device)
                real_items = real_items.to(device)
                batch_size = states.size(0)

                real_labels = torch.ones(batch_size, device=device)
                fake_labels = torch.zeros(batch_size, device=device)

                with torch.no_grad():
                    fake_logits = generator(states)
                    fake_items = torch.multinomial(torch.softmax(fake_logits, dim=1), num_samples=1).squeeze(1)

                opt_d.zero_grad()
                real_score = discriminator(states, real_items)
                fake_score = discriminator(states, fake_items)
                d_loss = bce(real_score, real_labels) + bce(fake_score, fake_labels)
                d_loss.backward()
                opt_d.step()

                opt_g.zero_grad()
                logits = generator(states)
                probs = torch.softmax(logits, dim=1)
                sampled_items = torch.multinomial(probs, num_samples=1).squeeze(1)
                adv_score = discriminator(states, sampled_items)
                supervised_loss = ce(logits, real_items)
                adv_loss = bce(adv_score, real_labels)
                g_loss = args.supervised_weight * supervised_loss + args.adv_weight * adv_loss
                g_loss.backward()
                opt_g.step()

                total_g += float(g_loss.item())
                total_d += float(d_loss.item())
                batches += 1
                if args.progress_every > 0 and batch_idx % args.progress_every == 0:
                    print(
                        f"Epoch {epoch:03d} batch {batch_idx}/{len(train_loader)} | "
                        f"G {total_g / batches:.4f} | D {total_d / batches:.4f}",
                        flush=True,
                    )

            print(f"Epoch {epoch:03d} training done, evaluating validation...", flush=True)
            metrics = evaluate_generator(generator, val_dataset, args.top_k, args.batch_size, device)
            avg_g = total_g / max(batches, 1)
            avg_d = total_d / max(batches, 1)

            writer.writerow(
                {
                    "epoch": epoch,
                    "generator_loss": avg_g,
                    "discriminator_loss": avg_d,
                    f"val_hit_rate_at_{args.top_k}": metrics.hit_rate_at_k,
                    f"val_precision_at_{args.top_k}": metrics.precision_at_k,
                    f"val_recall_at_{args.top_k}": metrics.recall_at_k,
                    f"val_ndcg_at_{args.top_k}": metrics.ndcg_at_k,
                    f"val_mrr_at_{args.top_k}": metrics.mrr_at_k,
                }
            )
            f.flush()

            print(
                f"Epoch {epoch:03d} | G {avg_g:.4f} | D {avg_d:.4f} | "
                f"Val HitRate@{args.top_k} {metrics.hit_rate_at_k:.4f}"
            )

            if best_metrics is None or metrics.hit_rate_at_k > best_metrics.hit_rate_at_k:
                best_metrics = metrics
                torch.save(
                    {
                        "model_state_dict": generator.state_dict(),
                        "action_dim": action_dim,
                        "state_size": args.state_size,
                        "embedding_dim": args.embedding_dim,
                        "hidden_dim": args.hidden_dim,
                        "noise_dim": args.noise_dim,
                        "top_k": args.top_k,
                        "best_val_hit_rate": metrics.hit_rate_at_k,
                    },
                    args.checkpoint_path,
                )

    if best_metrics is None:
        raise RuntimeError("Training khong tao duoc validation metric.")

    result_path = Path(args.val_result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        f"model,model_path,hit_rate_at_{args.top_k},precision_at_{args.top_k},"
        f"recall_at_{args.top_k},ndcg_at_{args.top_k},mrr_at_{args.top_k},samples\n"
        f"GAN,{args.checkpoint_path},{best_metrics.hit_rate_at_k},"
        f"{best_metrics.precision_at_k},{best_metrics.recall_at_k},"
        f"{best_metrics.ndcg_at_k},{best_metrics.mrr_at_k},{best_metrics.samples}\n",
        encoding="utf-8",
    )
    print(f"Best validation HitRate@{args.top_k}: {best_metrics.hit_rate_at_k:.4f}")
    print(f"Saved checkpoint: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
