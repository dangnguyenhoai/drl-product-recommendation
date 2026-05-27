import random

import numpy as np


class RecommendationEnv:
    def __init__(
        self,
        user_history,
        state_size=5,
        top_k=5,
        max_steps=10,
        miss_penalty=-2.0,
        hit_reward=5.0,
    ):
        self.user_history = user_history
        self.state_size = state_size
        self.top_k = top_k
        self.max_steps = max_steps
        self.miss_penalty = miss_penalty
        self.hit_reward = hit_reward

        self.users = [
            user
            for user, history in self.user_history.items()
            if len(history) >= self.state_size + self.top_k + 1
        ]

        if len(self.users) == 0:
            raise ValueError(
                "No user has enough interaction history. "
                f"Each user needs at least {self.state_size + self.top_k + 1} items."
            )

        self.current_user = None
        self.current_history = None
        self.current_step = 0
        self.pointer = 0
        self.state = None
        self.recommended_items = set()

    def reset(self):
        self.current_user = random.choice(self.users)
        self.current_history = self.user_history[self.current_user]

        max_start = len(self.current_history) - self.state_size - self.top_k - 1
        self.pointer = random.randint(0, max_start)

        self.current_step = 0
        self.recommended_items = set()

        self.state = self.current_history[
            self.pointer : self.pointer + self.state_size
        ]

        return np.array(self.state, dtype=np.float32)

    def step(self, actions):
        if not isinstance(actions, (list, tuple, np.ndarray)):
            actions = [actions]

        actions = list(actions)

        target_start = self.pointer + self.state_size
        target_end = target_start + self.top_k

        target_items = self.current_history[target_start:target_end]
        target_set = set(target_items)

        unique_actions = []
        duplicate_count = 0

        for item in actions:
            if item in unique_actions:
                duplicate_count += 1
            else:
                unique_actions.append(item)

        hits = 0

        for item in unique_actions:
            if item in target_set and item not in self.recommended_items:
                hits += 1

            self.recommended_items.add(item)

        if hits > 0:
            reward = hits * self.hit_reward
        else:
            reward = self.miss_penalty

        reward -= duplicate_count

        self.pointer += 1
        self.current_step += 1

        done = (
            self.current_step >= self.max_steps
            or self.pointer + self.state_size + self.top_k >= len(self.current_history)
        )

        next_state = self.current_history[
            self.pointer : self.pointer + self.state_size
        ]

        self.state = next_state

        info = {
            "user": self.current_user,
            "target_items": target_items,
            "hits": hits,
            "recommended_items": unique_actions,
        }

        return np.array(next_state, dtype=np.float32), reward, done, info