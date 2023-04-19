import itertools
import pickle
import random
from collections import deque
from model import QNet
import gymnasium as gym
import numpy as np
import torch
from torch import nn

GAMMA = 0.99  # discount factor
BATCH_SIZE = 256  # samples from the replay buffer
BUFFER_SIZE = 1000000  # max size of the replay buffer before overriding it
MIN_REPLAY_SIZE = 2048  # min number fo samples inside the replay buffer before computing the gradient or starting the training
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 5000  # epsilon reach from 1 to 0.02 linearly over epsilon_decay steps
TARGET_UPDATE_FREQ = 4  # number of steps we equalled online parameter to targets
LR = 5e-5
TOTAL_EPISODES = 10000
EPISODE_LENGTH = 2000
ACTION_D_LVL = 5  # action space discretization level
SAVE_FREQ = 50
SAVE_DIR = 'trained_models/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    env = gym.make("BipedalWalker-v3", render_mode="human", autoreset=True, max_episode_steps=EPISODE_LENGTH)
    replay_buffer = deque(maxlen=BUFFER_SIZE)

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    online_net = QNet(n_states=n_states, n_actions=n_actions, disct_lvl=3, seed=0).to(device)
    target_net = QNet(n_states=n_states, n_actions=n_actions, disct_lvl=3, seed=0).to(device)

    optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)

    start_episode = 0
    # initialize Replay Buffer
    state, _ = env.reset()

    # save min transitions into replay buffer
    for _ in range(MIN_REPLAY_SIZE):
        action = env.action_space.sample()
        new_state, reward, terminated, truncated, _ = env.step(action)
        transition = (state, action, reward, terminated, new_state)
        replay_buffer.append(transition)
        state = new_state
        if terminated or truncated:
            state, _ = env.reset()

    # rewards earned by the agents in a single episode to track the improvements of the agents as it trains
    reward_buffer = deque(maxlen=TOTAL_EPISODES)
    distance_buffer = deque(maxlen=TOTAL_EPISODES)
    total_reward = 0.0

    scores = []
    mean_scores = []
    last_scores = deque(maxlen=SAVE_FREQ)
    distances = []
    mean_distances = []
    last_distance = deque(maxlen=SAVE_FREQ)

    for ep in range(start_episode + 1, TOTAL_EPISODES + 1):
        epsilon = np.interp(ep, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
        state, info = env.reset()
        episode_reward = 0
        total_distance = 0
        losses = []

        for step in range(EPISODE_LENGTH):
            # select action by epsilon-greedy policy
            rnd_sample = random.random()
            if rnd_sample <= epsilon:
                # action = random.randrange(ACTION_D_LVL ** n_actions)
                action = env.action_space.sample()
            else:
                action = online_net.act(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
                # action = online_net.act(state)


            new_state, reward, terminated, truncated, _ = env.step(action)
            # add the transition into replay buffer
            transition = (state, action, reward, terminated, new_state)
            replay_buffer.append(transition)
            state = new_state
            episode_reward += reward

            # start gradient step
            # get a batch from the replay buffer
            transitions = random.sample(replay_buffer, BATCH_SIZE)

            states = np.asarray([t[0] for t in transitions])
            actions = np.asarray([t[1] for t in transitions])
            rewards = np.asarray([t[2] for t in transitions])
            terminateds = np.asarray([t[3] for t in transitions])
            new_states = np.asarray([t[4] for t in transitions])
            # convert to tensor
            states = torch.as_tensor(states, dtype=torch.float32).to(device)
            actions = torch.as_tensor(actions, dtype=torch.long).to(device)
            rewards = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device)
            terminateds = torch.as_tensor(terminateds, dtype=torch.uint8).unsqueeze(-1).to(device)
            new_states = torch.as_tensor(new_states, dtype=torch.float32).to(device)
            # Computer Targets
            target_q_val = target_net(new_states)
            # max_target_q_values = target_q_values.max(dim=1, keepdim=True)
            target_q_val = rewards + GAMMA * target_q_val * (1 - terminateds)
            # Compute Loss
            q_val = online_net(new_states)
            # action_q_values = torch.gather(input=q_values, dim=-1, index=actions)

            # loss = nn.functional.smooth_l1_loss(action_q_values, targets)
            # loss = nn.functional.smooth_l1_loss(q_val, target_q_val)
            loss = nn.functional.mse_loss(q_val, target_q_val)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if reward != -100:
                total_distance += reward

            if step % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(online_net.state_dict())

            if terminated or truncated:
                distance_buffer.append(total_distance)
                reward_buffer.append(episode_reward)
                break

        print(f'Episode: {ep},\tScore: {episode_reward},\tDistance: {total_distance}, \tEpsilon: {epsilon}, \tAvg '
              f'Score: {np.mean(reward_buffer)}, \tAvg Distance: {np.mean(distance_buffer)}')

        scores.append(total_reward)
        distances.append(total_distance)
        last_scores.append(total_reward)
        last_distance.append(total_distance)
        mean_score = np.mean(last_scores)
        mean_distance = np.mean(last_distance)

        # record rewards dynamically
        file_name = 'records.dat'
        data = [ep, total_reward, total_distance, epsilon]
        with open(file_name, "ab") as f:
            pickle.dump(data, f)

        if mean_score >= 300:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(ep, mean_score))
            torch.save(online_net.state_dict(), SAVE_DIR + '/best/online_ep' + str(ep) + '.pth')
            torch.save(target_net.state_dict(), SAVE_DIR + '/best/target_ep' + str(ep) + '.pth')
            break

        # save model every MEAN_EVERY episodes
        if ep % SAVE_FREQ == 0:
            torch.save(online_net.state_dict(), SAVE_DIR + '/online_ep' + str(ep) + '.pth')
            torch.save(target_net.state_dict(), SAVE_DIR + '/target_ep' + str(ep) + '.pth')
            mean_scores.append(mean_score)
            mean_distances.append(mean_distance)
            FILE = 'records_mean.dat'
            data = [ep, mean_score, mean_distance, epsilon]
            with open(FILE, "ab") as f:
                pickle.dump(data, f)
    env.close()
