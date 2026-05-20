from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .environment import InstacartRecommendationEnv
from .models import DQN


def evaluate(args: argparse.Namespace) -> None:
    payload = torch.load(args.episodes, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    episodes = payload["episodes"]
    if args.max_users:
        episodes = episodes[: args.max_users]

    action_size = int(checkpoint["action_size"])
    model = DQN(checkpoint["state_dim"], action_size, checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    env = InstacartRecommendationEnv(episodes, action_size=action_size, top_n=args.top_n)
    rows = []
    total_hits = 0
    total_precision = 0.0
    total_recall = 0.0

    for episode in tqdm(episodes, desc="Evaluating"):
        state = env.reset(episode)
        actions: list[int] = []
        for _ in range(args.top_n):
            with torch.no_grad():
                q_values = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze(0)
            if actions:
                q_values[actions] = -torch.inf
            action = int(q_values.argmax().item())
            actions.append(action)
            state, _, done, _ = env.step(action)
            if done:
                break

        hits = len(set(actions).intersection(episode.targets))
        precision = hits / max(len(actions), 1)
        recall = hits / max(len(episode.targets), 1)
        total_hits += hits
        total_precision += precision
        total_recall += recall
        rows.append(
            {
                "user_id": episode.user_id,
                "target_count": len(episode.targets),
                "hits": hits,
                f"precision_at_{args.top_n}": precision,
                f"recall_at_{args.top_n}": recall,
            }
        )

    n = max(len(episodes), 1)
    summary = {
        "users": len(episodes),
        "top_n": args.top_n,
        "total_hits": total_hits,
        f"hit_rate_at_{args.top_n}": sum(row["hits"] > 0 for row in rows) / n,
        f"precision_at_{args.top_n}": total_precision / n,
        f"recall_at_{args.top_n}": total_recall / n,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "evaluation_by_user.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "evaluation_summary.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate top-N recommendation quality.")
    parser.add_argument("--model", type=Path, default=Path("artifacts/dqn_model.pt"))
    parser.add_argument("--episodes", type=Path, default=Path("artifacts/episodes.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()

