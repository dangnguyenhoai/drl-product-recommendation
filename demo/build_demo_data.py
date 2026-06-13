import argparse
import csv
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def to_repo_relative_path(path):
    path = resolve_repo_path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.dqn_model import DQN


STATE_SIZE = 5
TOP_K = 5
HIT_REWARD = 5.0
MISS_PENALTY = -2.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build compact JSON data for the local recommendation demo.",
    )
    parser.add_argument(
        "--train_history_path",
        type=Path,
        default=Path("data") / "processed" / "train_history.pkl",
    )
    parser.add_argument(
        "--eval_history_path",
        type=Path,
        default=Path("data") / "processed" / "test_history.pkl",
    )
    parser.add_argument(
        "--fallback_history_path",
        type=Path,
        default=Path("data") / "processed" / "indexed_history.pkl",
    )
    parser.add_argument(
        "--case_history_path",
        type=Path,
        default=None,
        help=(
            "Optional history pickle used only for interactive demo cases. "
            "Evaluation metrics still come from --eval_history_path and --metrics_csv."
        ),
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("outputs") / "checkpoints" / "dqn_recency5_stable.pth",
    )
    parser.add_argument(
        "--metrics_csv",
        type=Path,
        default=Path("outputs") / "logs" / "final_test_results.csv",
    )
    parser.add_argument(
        "--training_log",
        type=Path,
        default=Path("outputs") / "logs" / "train_dqn_recency5_stable.csv",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data") / "demo" / "demo_data.json",
    )
    parser.add_argument("--recent_boost", type=float, default=None)
    parser.add_argument("--max_cases", type=int, default=90)
    parser.add_argument(
        "--max_case_users",
        type=int,
        default=None,
        help=(
            "Limit users loaded into the interactive case picker. "
            "When --case_history_path is set and this is omitted, 500 users are used."
        ),
    )
    parser.add_argument(
        "--max_windows_per_user",
        type=int,
        default=20,
        help="Maximum state windows sampled per user for the interactive case picker.",
    )
    parser.add_argument("--similar_top_k", type=int, default=8)
    parser.add_argument("--cooccurrence_window", type=int, default=5)
    parser.add_argument("--skip_product_names", action="store_true")
    return parser.parse_args()


def normalize_path_args(args):
    args.train_history_path = resolve_repo_path(args.train_history_path)
    args.eval_history_path = resolve_repo_path(args.eval_history_path)
    args.fallback_history_path = resolve_repo_path(args.fallback_history_path)
    args.model_path = resolve_repo_path(args.model_path)
    args.metrics_csv = resolve_repo_path(args.metrics_csv)
    args.training_log = resolve_repo_path(args.training_log)
    args.output_path = resolve_repo_path(args.output_path)

    if args.case_history_path is not None:
        args.case_history_path = resolve_repo_path(args.case_history_path)


def require_file(path, label):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )


def validate_required_files(args):
    require_file(args.train_history_path, "Train history file")
    require_file(args.eval_history_path, "Evaluation history file")
    require_file(args.model_path, "DQN model checkpoint")
    require_file(args.metrics_csv, "Metrics CSV")
    require_file(args.training_log, "Training log CSV")

    if args.case_history_path is not None:
        require_file(args.case_history_path, "Case history file")


def load_history(path):
    if not path.exists():
        return {}

    with open(path, "rb") as f:
        data = pickle.load(f)

    return {
        str(user_id): [int(item) for item in history]
        for user_id, history in data.items()
    }


def pick_histories(args):
    train_history = load_history(args.train_history_path)
    eval_history = load_history(args.eval_history_path)

    if not eval_history:
        eval_history = load_history(args.fallback_history_path)

    if not train_history:
        train_history = eval_history

    return train_history, eval_history


def limit_history_users(history, max_users):
    if max_users is None or max_users <= 0:
        return history

    eligible = [
        (user_id, user_history)
        for user_id, user_history in history.items()
        if len(user_history) >= STATE_SIZE + TOP_K
    ]
    eligible.sort(key=lambda item: len(item[1]), reverse=True)

    return dict(eligible[:max_users])


def flatten_items(histories):
    for history in histories.values():
        for item in history:
            yield int(item)


def infer_recent_boost(model_path, recent_boost):
    if recent_boost is not None:
        return recent_boost

    name = model_path.name.lower()
    match = re.search(r"boost(\d+(?:\.\d+)?)", name)
    if match:
        return float(match.group(1))

    if "recency5" in name:
        return 5.0

    return 0.0


def load_dqn(model_path, state_size):
    if not model_path.exists():
        return None, {
            "path": to_repo_relative_path(model_path),
            "available": False,
            "reason": "checkpoint not found",
        }

    state_dict = torch.load(model_path, map_location="cpu")
    if "item_embedding.weight" not in state_dict:
        return None, {
            "path": to_repo_relative_path(model_path),
            "available": False,
            "reason": "checkpoint does not contain item_embedding.weight",
        }

    action_dim, embedding_dim = state_dict["item_embedding.weight"].shape
    
    if "value_stream.0.weight" in state_dict:
        hidden_dim = state_dict["value_stream.0.weight"].shape[1]
    elif "q_network.0.bias" in state_dict:
        hidden_dim = state_dict["q_network.0.bias"].shape[0]
    else:
        hidden_dim = 128 # Default fallback

    model = DQN(
        state_dim=state_size,
        action_dim=action_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model, {
        "path": to_repo_relative_path(model_path),
        "available": True,
        "action_dim": int(action_dim),
        "embedding_dim": int(embedding_dim),
        "hidden_dim": int(hidden_dim),
    }


def build_product_lookup(max_items, skip_product_names):
    fallback = {
        idx: {
            "item_id": idx,
            "raw_product_id": None,
            "name": f"Indexed item {idx}",
            "aisle_id": None,
            "department_id": None,
        }
        for idx in range(max_items)
    }

    if skip_product_names:
        return fallback

    data_dir = REPO_ROOT / "data"
    products_path = data_dir / "products.csv"
    order_product_paths = [
        data_dir / "order_products__prior.csv",
        data_dir / "order_products__train.csv",
    ]

    if not products_path.exists() or not all(path.exists() for path in order_product_paths):
        return fallback

    products = pd.read_csv(str(products_path))
    product_meta = {}
    for pid, pname, aid, did in zip(
        products["product_id"],
        products["product_name"],
        products["aisle_id"],
        products["department_id"]
    ):
        product_meta[int(pid)] = {
            "name": str(pname),
            "aisle_id": int(aid),
            "department_id": int(did),
        }

    product_counter = Counter()
    for path in order_product_paths:
        df = pd.read_csv(str(path), usecols=["product_id"])
        counts = df["product_id"].value_counts()
        for pid, count in counts.items():
            product_counter[int(pid)] += int(count)

    lookup = {}
    for idx, (product_id, _) in enumerate(product_counter.most_common(max_items)):
        meta = product_meta.get(int(product_id), {})
        lookup[idx] = {
            "item_id": idx,
            "raw_product_id": int(product_id),
            "name": meta.get("name", f"Product {product_id}"),
            "aisle_id": meta.get("aisle_id"),
            "department_id": meta.get("department_id"),
        }

    for idx in range(max_items):
        lookup.setdefault(idx, fallback[idx])

    return lookup


def item_detail(item_id, lookup, popularity):
    item_id = int(item_id)
    base = lookup.get(
        item_id,
        {
            "item_id": item_id,
            "raw_product_id": None,
            "name": f"Indexed item {item_id}",
            "aisle_id": None,
            "department_id": None,
        },
    )
    detail = dict(base)
    detail["popularity_count"] = int(popularity.get(item_id, 0))
    return detail


def get_valid_actions(histories, action_dim):
    valid = sorted(
        {
            int(item)
            for item in flatten_items(histories)
            if int(item) >= 0 and int(item) < action_dim
        }
    )
    return valid


def q_scores_for_state(model, state, valid_actions, recent_boost):
    if model is None:
        return []

    state_tensor = torch.LongTensor([state])
    with torch.no_grad():
        q_values = model(state_tensor)[0].cpu()

    state_set = set(int(item) for item in state)
    rows = []
    for action in valid_actions:
        raw_q = float(q_values[action].item())
        bonus = float(recent_boost) if action in state_set else 0.0
        rows.append(
            {
                "item_id": int(action),
                "q_value": raw_q,
                "recent_bonus": bonus,
                "score": raw_q + bonus,
            }
        )

    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows


def cosine_similarity_rows(model, valid_actions):
    if model is None:
        return {}

    weights = model.item_embedding.weight.detach().cpu()
    norms = weights.norm(dim=1, keepdim=True).clamp_min(1e-8)
    normalized = weights / norms

    similarities = {}
    for item_id in valid_actions:
        if item_id >= normalized.shape[0]:
            continue
        scores = torch.matmul(normalized, normalized[item_id])
        similarities[item_id] = {
            int(action): float(scores[action].item())
            for action in valid_actions
            if action < scores.shape[0]
        }

    return similarities


def build_cooccurrence(histories, valid_actions, window_size):
    valid_set = set(valid_actions)
    cooccurrence = {item: Counter() for item in valid_actions}

    for history in histories.values():
        filtered = [int(item) for item in history if int(item) in valid_set]
        for idx, item in enumerate(filtered):
            left = max(0, idx - window_size)
            right = min(len(filtered), idx + window_size + 1)
            neighbors = set(filtered[left:idx] + filtered[idx + 1 : right])
            for neighbor in neighbors:
                if neighbor != item:
                    cooccurrence[item][neighbor] += 1

    return cooccurrence


def safe_norm(value, max_value):
    if max_value <= 0:
        return 0.0
    return float(value) / float(max_value)


def build_similar_items(
    valid_actions,
    lookup,
    popularity,
    cooccurrence,
    similarities,
    top_k,
):
    max_popularity = max(popularity.values()) if popularity else 1
    output = {}

    for base_item in valid_actions:
        base_co = cooccurrence.get(base_item, Counter())
        max_co = max(base_co.values()) if base_co else 0
        rows = []

        for candidate in valid_actions:
            if candidate == base_item:
                continue

            embedding_similarity = similarities.get(base_item, {}).get(candidate, 0.0)
            embedding_norm = (embedding_similarity + 1.0) / 2.0
            co_count = int(base_co.get(candidate, 0))
            co_norm = safe_norm(co_count, max_co)
            popularity_norm = safe_norm(popularity.get(candidate, 0), max_popularity)

            score = (
                0.65 * embedding_norm
                + 0.25 * co_norm
                + 0.10 * popularity_norm
            )

            detail = item_detail(candidate, lookup, popularity)
            detail.update(
                {
                    "score": round(score, 6),
                    "embedding_similarity": round(embedding_similarity, 6),
                    "cooccurrence_count": co_count,
                    "popularity_norm": round(popularity_norm, 6),
                }
            )
            rows.append(detail)

        rows.sort(key=lambda row: row["score"], reverse=True)
        output[str(base_item)] = rows[:top_k]

    return output


def build_cases(
    case_history,
    valid_actions,
    lookup,
    popularity,
    model,
    recent_boost,
    max_cases,
    max_windows_per_user,
):
    rows = []
    valid_set = set(valid_actions)

    for user_id, history in case_history.items():
        if len(history) < STATE_SIZE + TOP_K:
            continue

        max_pointer = len(history) - STATE_SIZE - TOP_K
        pointers = sample_pointers(max_pointer, max_windows_per_user)

        for pointer in pointers:
            state = [int(item) for item in history[pointer : pointer + STATE_SIZE]]
            if any(item not in valid_set for item in state):
                continue

            target = [
                int(item)
                for item in history[pointer + STATE_SIZE : pointer + STATE_SIZE + TOP_K]
            ]

            score_rows = q_scores_for_state(model, state, valid_actions, recent_boost)
            if score_rows:
                selected = score_rows[:TOP_K]
            else:
                selected = [
                    {
                        "item_id": int(item),
                        "q_value": 0.0,
                        "recent_bonus": 0.0,
                        "score": float(popularity.get(item, 0)),
                    }
                    for item in valid_actions[:TOP_K]
                ]

            target_set = set(target)
            hits = sum(1 for row in selected if row["item_id"] in target_set)
            reward = hits * HIT_REWARD if hits > 0 else MISS_PENALTY

            recommendations = []
            for rank, row in enumerate(selected, start=1):
                detail = item_detail(row["item_id"], lookup, popularity)
                detail.update(
                    {
                        "rank": rank,
                        "q_value": round(float(row["q_value"]), 6),
                        "recent_bonus": round(float(row["recent_bonus"]), 6),
                        "score": round(float(row["score"]), 6),
                        "is_hit": row["item_id"] in target_set,
                        "valid_action": row["item_id"] in valid_set,
                    }
                )
                recommendations.append(detail)

            case_id = f"user-{user_id}-p{pointer}"
            rows.append(
                {
                    "case_id": case_id,
                    "label": f"User {user_id} | window {pointer} | hits {hits}",
                    "user_id": user_id,
                    "pointer": pointer,
                    "state": [item_detail(item, lookup, popularity) for item in state],
                    "target": [item_detail(item, lookup, popularity) for item in target],
                    "recommendations": recommendations,
                    "hits": int(hits),
                    "reward": float(reward),
                    "hit_rate_at_5": round(hits / TOP_K, 6),
                }
            )

    return select_diverse_cases(rows, max_cases)


def sample_pointers(max_pointer, max_windows_per_user):
    total_windows = max_pointer + 1

    if max_windows_per_user is None or max_windows_per_user <= 0:
        return range(total_windows)

    if total_windows <= max_windows_per_user:
        return range(total_windows)

    if max_windows_per_user == 1:
        return [0]

    return sorted(
        {
            round(idx * max_pointer / (max_windows_per_user - 1))
            for idx in range(max_windows_per_user)
        }
    )


def select_diverse_cases(rows, max_cases):
    rows.sort(
        key=lambda row: (
            row["hits"],
            row["reward"],
            -row["pointer"],
        ),
        reverse=True,
    )

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["user_id"]].append(row)

    user_order = sorted(
        grouped,
        key=lambda user_id: (
            grouped[user_id][0]["hits"],
            grouped[user_id][0]["reward"],
        ),
        reverse=True,
    )

    selected = []
    while len(selected) < max_cases:
        added = False
        for user_id in user_order:
            if grouped[user_id]:
                selected.append(grouped[user_id].pop(0))
                added = True

                if len(selected) >= max_cases:
                    break

        if not added:
            break

    return selected


def read_metrics(metrics_csv):
    if not metrics_csv.exists():
        return {
            "source": None,
            "rows": [],
        }

    rows = []
    with open(metrics_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = dict(row)
            for key in ["average_reward", "hit_rate_at_5", "episodes", "recent_boost"]:
                if parsed.get(key) in (None, ""):
                    parsed[key] = None
                    continue
                try:
                    parsed[key] = float(parsed[key])
                except ValueError:
                    parsed[key] = None
            rows.append(parsed)

    rows.sort(
        key=lambda row: row["hit_rate_at_5"] if row["hit_rate_at_5"] is not None else -1,
        reverse=True,
    )
    return {
        "source": to_repo_relative_path(metrics_csv),
        "rows": rows,
    }


def downsample(rows, max_points=120):
    if len(rows) <= max_points:
        return rows

    step = len(rows) / max_points
    sampled = []
    for idx in range(max_points):
        sampled.append(rows[int(idx * step)])
    return sampled


def read_training_log(training_log):
    if not training_log.exists():
        return {
            "source": None,
            "points": [],
        }

    points = []
    with open(training_log, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                points.append(
                    {
                        "episode": int(float(row["episode"])),
                        "reward": float(row["reward"]),
                        "loss": float(row["loss"]),
                        "epsilon": float(row["epsilon"]),
                    }
                )
            except (KeyError, ValueError):
                continue

    return {
        "source": to_repo_relative_path(training_log),
        "points": downsample(points),
        "last": points[-1] if points else None,
    }


def dataset_stats(name, history):
    lengths = []
    unique_items = set()
    interactions = 0

    for items in history.values():
        lengths.append(len(items))
        interactions += len(items)
        unique_items.update(items)

    if not lengths:
        return {
            "name": name,
            "users": 0,
            "interactions": 0,
            "unique_items": 0,
            "min_history_len": 0,
            "max_history_len": 0,
            "avg_history_len": 0,
        }

    return {
        "name": name,
        "users": len(history),
        "interactions": interactions,
        "unique_items": len(unique_items),
        "min_history_len": min(lengths),
        "max_history_len": max(lengths),
        "avg_history_len": round(sum(lengths) / len(lengths), 2),
    }


def main():
    args = parse_args()
    normalize_path_args(args)
    validate_required_files(args)

    train_history, eval_history = pick_histories(args)
    model, model_meta = load_dqn(args.model_path, STATE_SIZE)

    action_dim = int(model_meta.get("action_dim") or (max(flatten_items(eval_history)) + 1))
    recent_boost = infer_recent_boost(args.model_path, args.recent_boost)

    union_history = dict(train_history)
    union_history.update({f"eval-{user}": history for user, history in eval_history.items()})

    if args.case_history_path is not None:
        case_history = load_history(args.case_history_path)
        case_history_name = args.case_history_path.stem
        case_max_users = args.max_case_users if args.max_case_users is not None else 500
    else:
        case_history = eval_history
        case_history_name = "test_history"
        case_max_users = args.max_case_users if args.max_case_users is not None else 100

    case_history = limit_history_users(case_history, case_max_users)

    valid_actions = get_valid_actions(union_history, action_dim)
    eval_valid_actions = get_valid_actions(eval_history, action_dim)
    if not eval_valid_actions:
        eval_valid_actions = valid_actions

    case_valid_actions = get_valid_actions(case_history, action_dim)
    if not case_valid_actions:
        case_valid_actions = eval_valid_actions

    popularity = Counter(flatten_items(union_history))
    product_lookup = build_product_lookup(
        max_items=action_dim,
        skip_product_names=args.skip_product_names,
    )

    cooccurrence = build_cooccurrence(
        histories=union_history,
        valid_actions=valid_actions,
        window_size=args.cooccurrence_window,
    )
    similarities = cosine_similarity_rows(model, valid_actions)
    similar_items = build_similar_items(
        valid_actions=valid_actions,
        lookup=product_lookup,
        popularity=popularity,
        cooccurrence=cooccurrence,
        similarities=similarities,
        top_k=args.similar_top_k,
    )

    cases = build_cases(
        case_history=case_history,
        valid_actions=case_valid_actions,
        lookup=product_lookup,
        popularity=popularity,
        model=model,
        recent_boost=recent_boost,
        max_cases=args.max_cases,
        max_windows_per_user=args.max_windows_per_user,
    )

    product_options = [
        item_detail(item, product_lookup, popularity)
        for item in valid_actions
    ]
    product_options.sort(key=lambda item: item["name"].lower())

    metrics = read_metrics(args.metrics_csv)
    training = read_training_log(args.training_log)
    full_history = load_history(args.fallback_history_path)
    full_stats = dataset_stats("indexed_history", full_history) if full_history else None
    del full_history

    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "state_size": STATE_SIZE,
            "top_k": TOP_K,
            "hit_reward": HIT_REWARD,
            "miss_penalty": MISS_PENALTY,
            "recent_boost": recent_boost,
            "model": model_meta,
            "notes": [
                "DQN action score = raw Q-value + recent-item bonus.",
                "Product similarity score = 0.65 embedding similarity + 0.25 co-occurrence + 0.10 popularity.",
                "Evaluation metrics are computed on the test split; interactive cases can use another history file for demo breadth.",
            ],
        },
        "dataset": {
            "full": full_stats,
            "train": dataset_stats("train_history", train_history),
            "eval": dataset_stats("test_history", eval_history),
            "case": dataset_stats(case_history_name, case_history),
            "valid_actions": len(valid_actions),
            "eval_valid_actions": len(eval_valid_actions),
            "case_valid_actions": len(case_valid_actions),
        },
        "items": {
            str(item): item_detail(item, product_lookup, popularity)
            for item in valid_actions
        },
        "product_options": product_options,
        "similar_items": similar_items,
        "cases": cases,
        "metrics": metrics,
        "training": training,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    rel_output = to_repo_relative_path(args.output_path)
    print(f"Saved demo data: {rel_output}")
    print(f"Products: {len(product_options)}")
    print(f"Cases: {len(cases)}")
    print(f"Metrics source: {metrics['source']}")


if __name__ == "__main__":
    main()
