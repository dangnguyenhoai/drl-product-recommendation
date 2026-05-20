import numpy as np 
import random as rd 

class RecommendationEnv:

    def __init__(self, user_history, state_size=5, max_steps=10):

        self.user_history = user_history
        self.users = list(user_history.keys())
        
        self.state_size = state_size
        self.max_steps = max_steps

        self.current_user = None
        self.current_step = 0

        self.current_history = None 

        self.recommended_items = set()

    def reset(self):

        self.current_user = rd.choice(self.users)

        self.current_history = self.user_history[self.current_user]

        self.current_step = 0

        self.recommended_items = set()

        self.state = self.current_history[:self.state_size]

        return np.array(self.state)
    
    def step(self, action):

        reward = 0

        if (
            action in self.current_history
            and action not in self.recommended_items
        ):
            reward = 5
        else:
            reward = -1

        self.recommended_items.add(action)

        self.current_step += 1

        done = (
            self.current_step >= self.max_steps
        )

        next_state = self.state.copy()

        if reward > 0:

            next_state.pop(0)

            next_state.append(action)

        self.state = next_state

        return (
            np.array(next_state),
            reward,
            done,
            {}
        )