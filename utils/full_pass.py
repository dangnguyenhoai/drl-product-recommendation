from math import ceil


FULL_PASS_VALUES = {
    "all",
    "full",
    "full_pass",
    "all_interactions",
}


def uses_full_pass(value):
    return str(value).strip().lower() in FULL_PASS_VALUES


def parse_episode_count(value):
    if isinstance(value, int):
        episode_count = value
    else:
        episode_count = int(str(value).strip())

    if episode_count <= 0:
        raise ValueError("--episodes must be a positive integer or 'all'")

    return episode_count


def iter_full_pass_episode_starts(indexed_history, state_size, top_k, max_steps):
    for user_id in sorted(indexed_history):
        history = indexed_history[user_id]
        max_start = len(history) - state_size - top_k - 1
        if max_start < 0:
            continue

        for pointer in range(0, max_start + 1, max_steps):
            yield user_id, pointer


def count_full_pass(indexed_history, state_size, top_k, max_steps):
    total_windows = 0
    total_episodes = 0

    for history in indexed_history.values():
        max_start = len(history) - state_size - top_k - 1
        if max_start < 0:
            continue

        windows = max_start + 1
        total_windows += windows
        total_episodes += ceil(windows / max_steps)

    return total_episodes, total_windows
