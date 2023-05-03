import gymnasium as gym
import numpy as np
from model import ActorNet

total_episodes = 100
episode_len = 2000
model_path = "results_3/"
env = gym.make("BipedalWalker-v3", autoreset=True, max_episode_steps=episode_len)
n_states = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]

actor = ActorNet(n_states=n_states, n_actions=n_actions, seed=0)


scores = []
for ep in range(1, total_episodes + 1):
    episode_reward = 0
    state, _ = env.reset()
    for step in range(episode_len):
        action = actor.act(state)
        state, reward, terminated, truncated, _ = env.step(action)
        episode_reward = 0
        if terminated or truncated:
            break
    scores.append(episode_reward)
    print(f'Episode: {ep}, Score: {episode_reward}')

env.close()
print("Score media", np.mean(scores))
