from env.recommendation_env import RecommendationEnv
import pickle
import random

with open(
    "data/processed/indexed_history.pkl",
    "rb"
) as f:

    indexed_history = pickle.load(f)

env = RecommendationEnv(
    indexed_history
)

state = env.reset()

print("Current history:", env.current_history)

print("Initial state:", state)


for step in range(5):

    action = random.randint(0, 100)

    next_state, reward, done, _ = env.step(
        action
    )

    print(f"\nStep {step}")

    print("Action:", action)
    print("Reward:", reward)
    print("Next state:", next_state)
