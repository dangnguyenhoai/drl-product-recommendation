from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .environment import InstacartRecommendationEnv
from .models import DQN


def recommend(args: argparse.Namespace) -> None:
    payload = torch.load(args.episodes, weights_only=False)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    episodes = payload["episodes"]
    action_size = int(checkpoint["action_size"])
    episode_by_user = {episode.user_id: episode for episode in episodes}
    if args.user_id not in episode_by_user:
        available = sorted(episode_by_user)[:10]
        raise ValueError(f"user_id {args.user_id} not found in prepared episodes. Example IDs: {available}")

    model = DQN(checkpoint["state_dim"], action_size, checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    env = InstacartRecommendationEnv(episodes, action_size=action_size, top_n=args.top_n)
    state = env.reset(episode_by_user[args.user_id])
    action_ids: list[int] = []
    for _ in range(args.top_n):
        with torch.no_grad():
            q_values = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        for selected in action_ids:
            q_values[selected] = -np.inf
        action = int(np.argmax(q_values))
        action_ids.append(action)
        state, _, done, _ = env.step(action)
        if done:
            break

    product_ids = [checkpoint["top_products"][action] for action in action_ids]
    products = pd.read_csv(args.products)
    names = products.set_index("product_id").loc[product_ids, "product_name"].tolist()
    result = pd.DataFrame({"rank": range(1, len(product_ids) + 1), "product_id": product_ids, "product_name": names})
    print(result.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate top-N recommendations for one user.")
    parser.add_argument("--model", type=Path, default=Path("artifacts/dqn_model.pt"))
    parser.add_argument("--episodes", type=Path, default=Path("artifacts/episodes.pt"))
    parser.add_argument("--products", type=Path, default=Path("data/products.csv"))
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    recommend(parse_args())


if __name__ == "__main__":
    main()

