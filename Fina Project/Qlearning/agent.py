import random
import math
from collections import defaultdict

import numpy as np


class Agent:
    def __init__(self, discretize_levels, gamma, alpha, total_exploring_episode):
        self.discretize_levels = discretize_levels
        self.state_space = [(-math.pi, math.pi),
                            (-5, 5),
                            (-5, 5),
                            (-5, 5),
                            (-math.pi, math.pi),
                            (-5, 5),
                            (-math.pi, math.pi),
                            (-5, 5),
                            (0, 5),
                            (-math.pi, math.pi),
                            (-5, 5),
                            (-math.pi, math.pi),
                            (-5, 5),
                            (0, 5),
                            # (0, 1),
                            # (0, 1),
                            # (0, 1),
                            # (0, 1),
                            # (0, 1),
                            # (0, 1),
                            # (0, 1),
                            # (0, 1),
                            # (0, 1),
                            # (0, 1),
                            ]
        # a dictionary that each state contains np.zeros(d_l, d_l, d_l, d_l)
        self.Q = defaultdict(lambda: np.zeros((discretize_levels,
                                               discretize_levels,
                                               discretize_levels,
                                               discretize_levels)))
        for e in self.Q:
            print(e)

        self.gamma = gamma
        self.alpha = alpha
        self.high_score = -300
        self.total_exploring_episode = total_exploring_episode

    def discretize_state(self, state):
        discrete_states = []
        for i in range(len(state)):
            # min-max normalization
            index = int((state[i] - self.state_space[i][0]) / (self.state_space[i][1] - self.state_space[i][0]))
            discrete_states.append(index)
        return tuple(discrete_states)

    def epsilon_policy(self, epsilon, state):

        if random.random() < epsilon:
            action = ()
            for i in range(0, 4):
                action += (random.randint(0, self.discretize_levels - 1),)
        else:
            action = np.unravel_index(np.argmax(self.Q[state]), self.Q[state].shape)

        return action

    def continuous_action(self, next_action):
        return tuple([(a / self.discretize_levels) * 2 - 1 for a in next_action])

    def update_Q(self, state, action, reward, next_state=None):
        current = self.Q[state][action]
        q_next = np.max(self.Q[next_state]) if next_state is not None else 0
        target = reward + (self.gamma * q_next)
        new_value = current + (self.alpha * (target - current))
        return new_value

    def Q_learning(self, env, i, min_eps):
        # print("Episode #: ", i)
        observation, info = env.reset()
        state = self.discretize_state(state=observation[0:14])
        total_reward = 0
        # exploring the first total_exploring_episode episodes
        epsilon = self.total_exploring_episode / i
        if epsilon < min_eps:
            epsilon = min_eps
        while True:
            next_action = self.epsilon_policy(epsilon=epsilon, state=state)
            next_state, reward, terminated, truncated, info = env.step(self.continuous_action(next_action))
            next_state = self.discretize_state(state=next_state[0:14])
            total_reward += reward
            self.Q[state][next_action] = self.update_Q(state=state, action=next_action, reward=reward,
                                                       next_state=next_state)
            state = next_state
            if terminated or truncated:
                break
        if total_reward > self.high_score:
            self.high_score = total_reward

        return total_reward, terminated, truncated, epsilon
