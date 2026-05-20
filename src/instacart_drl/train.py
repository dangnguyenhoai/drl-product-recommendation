from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from tqdm import trange

from .config import DQNConfig
from .environment import InstacartRecommendationEnv
from .models import DQN
from .replay import ReplayBuffer, Transition
from .utils import seed_everything


def epsilon_by_step(step: int, cfg: DQNConfig) -> float:
    progress = min(step / cfg.epsilon_decay_steps, 1.0)
    return cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * (1.0 - progress)


def available_actions_from_state(state, action_size: int) -> torch.Tensor:
    selected_mask = torch.tensor(state[-action_size:], dtype=torch.bool)
    return torch.where(~selected_mask)[0]


def select_action(policy: DQN, state, epsilon: float, action_size: int, device: torch.device) -> int:
    available = available_actions_from_state(state, action_size)
    if len(available) == 0:
        return 0
    if torch.rand(1).item() < epsilon:
        idx = int(torch.randint(len(available), (1,)).item())
        return int(available[idx].item())
    with torch.no_grad():
        tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = policy(tensor).squeeze(0).cpu()
        q_values[state[-action_size:].astype(bool)] = -torch.inf
        return int(q_values.argmax().item())


def optimize(
    policy: DQN,
    target: DQN,
    replay: ReplayBuffer,
    optimizer: Adam,
    cfg: DQNConfig,
    device: torch.device,
) -> float | None:
    if len(replay) < max(cfg.min_replay, cfg.batch_size):
        return None
    states, actions, rewards, next_states, dones = replay.sample(cfg.batch_size, device)
    q_values = policy(states).gather(1, actions)
    with torch.no_grad():
        next_q_values = target(next_states)
        selected_mask = next_states[:, -policy.net[-1].out_features :].bool()
        next_q_values = next_q_values.masked_fill(selected_mask, -torch.inf)
        next_q = next_q_values.max(dim=1, keepdim=True).values
        next_q = torch.where(torch.isfinite(next_q), next_q, torch.zeros_like(next_q))
        expected = rewards + cfg.gamma * next_q * (1.0 - dones)
    loss = nn.functional.smooth_l1_loss(q_values, expected)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
    optimizer.step()
    return float(loss.item())


def plot_metrics(metrics: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(metrics["epoch"], metrics["avg_reward"], marker="o")
    axes[0].set_title("Average episode reward")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Reward")
    axes[0].grid(alpha=0.3)

    axes[1].plot(metrics["epoch"], metrics["avg_loss"], marker="o", color="tab:red")
    axes[1].set_title("Average DQN loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(args.episodes, weights_only=False)
    episodes = payload["episodes"]
    action_size = int(payload["action_size"])

    env = InstacartRecommendationEnv(episodes, action_size=action_size, top_n=args.top_n, seed=args.seed)
    cfg = DQNConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        min_replay=args.min_replay,
    )
    policy = DQN(env.state_dim, action_size, cfg.hidden_dim).to(device)
    target = DQN(env.state_dim, action_size, cfg.hidden_dim).to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = Adam(policy.parameters(), lr=cfg.learning_rate)
    replay = ReplayBuffer(cfg.replay_size, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    global_step = 0

    for epoch in trange(1, args.epochs + 1, desc="Training"):
        epoch_rewards: list[float] = []
        epoch_losses: list[float] = []
        hits = 0
        total_actions = 0

        for _ in range(args.episodes_per_epoch):
            state = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                epsilon = epsilon_by_step(global_step, cfg)
                action = select_action(policy, state, epsilon, action_size, device)
                next_state, reward, done, info = env.step(action)
                replay.push(Transition(state, action, reward, next_state, done))
                loss = optimize(policy, target, replay, optimizer, cfg, device)
                if loss is not None and math.isfinite(loss):
                    epoch_losses.append(loss)
                state = next_state
                total_reward += reward
                hits += int(info["hit"])
                total_actions += 1
                global_step += 1
                if global_step % cfg.target_update_steps == 0:
                    target.load_state_dict(policy.state_dict())
            epoch_rewards.append(total_reward)

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        hit_rate = hits / max(total_actions, 1)
        row = {
            "epoch": epoch,
            "avg_reward": sum(epoch_rewards) / len(epoch_rewards),
            "avg_loss": avg_loss,
            "hit_rate": hit_rate,
            "epsilon": epsilon_by_step(global_step, cfg),
        }
        metrics.append(row)
        print(row)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_dir / "training_metrics.csv", index=False)
    plot_metrics(metrics_df, output_dir)
    torch.save(
        {
            "model_state": policy.state_dict(),
            "state_dim": env.state_dim,
            "action_size": action_size,
            "hidden_dim": cfg.hidden_dim,
            "top_products": payload["top_products"],
        },
        output_dir / "dqn_model.pt",
    )
    print(f"Saved model and metrics to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN recommender.")
    parser.add_argument("--episodes", type=Path, default=Path("artifacts/episodes.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--episodes-per-epoch", type=int, default=1000)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-replay", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
