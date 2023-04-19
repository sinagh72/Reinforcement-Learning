import itertools
import pickle
import random
from collections import deque
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from model import ActorNet, CriticNet
from noise import OrnsteinUhlenbeckNoise

GAMMA = 0.99  # discount factor
BATCH_SIZE = 256  # samples from the replay buffer
BUFFER_SIZE = 1000000  # max size of the replay buffer before overriding it
MIN_REPLAY_SIZE = 256  # min number fo samples inside the replay buffer before computing the gradient or starting the training
LR = 5e-5
TOTAL_EPISODES = 10000
EPISODE_LENGTH = 2000
SAVE_FREQ = 50
SAVE_DIR = 'trained_models/'
TAU = 0.001
LR_ACTOR = 0.0001  # learning rate of the actor
LR_CRITIC = 0.001  # learning rate of the critic
WEIGHT_DECAY = 0.001  # L2 weight decay
EXP_FACT = 1
EXP_DECAY = 0.999

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def soft_update(online_net, target_net, tau):
    for target_param, online_param in zip(online_net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)


if __name__ == "__main__":
    env = gym.make("BipedalWalker-v3", render_mode="human", autoreset=True, max_episode_steps=EPISODE_LENGTH)
    replay_buffer = deque(maxlen=BUFFER_SIZE)

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor_online_net = ActorNet(n_states=n_states, n_actions=n_actions, seed=0).to(device)
    actor_target_net = ActorNet(n_states=n_states, n_actions=n_actions, seed=0).to(device)
    actor_optimizer = torch.optim.Adam(actor_online_net.parameters(), lr=LR_ACTOR)

    critic_online_net = CriticNet(n_states=n_states, n_actions=n_actions, seed=0).to(device)
    critic_target_net = CriticNet(n_states=n_states, n_actions=n_actions, seed=0).to(device)
    critic_optimizer = torch.optim.Adam(critic_online_net.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

    noise = OrnsteinUhlenbeckNoise(n_actions, mu=0, theta=0.4, sigma=0.2)
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
    actor_losses = []
    critic_losses = []
    for ep in range(TOTAL_EPISODES):
        state, info = env.reset()
        episode_reward = 0
        total_distance = 0
        losses = []
        for step in range(EPISODE_LENGTH):
            # select action by epsilon-greedy policy
            action = actor_online_net.act(
                torch.as_tensor(state, dtype=torch.float32).to(device)) + noise.sample() * EXP_FACT
            EXP_FACT *= EXP_DECAY

            # add noise

            nxt_state, reward, terminated, truncated, _ = env.step(action)
            # add the transition into replay buffer
            transition = (state, action, reward, terminated, nxt_state)
            replay_buffer.append(transition)
            state = nxt_state
            episode_reward += reward

            # start gradient step
            # get a batch from the replay buffer
            transitions = random.sample(replay_buffer, BATCH_SIZE)

            states = np.asarray([t[0] for t in transitions])
            actions = np.asarray([t[1] for t in transitions])
            rewards = np.asarray([t[2] for t in transitions])
            terminateds = np.asarray([t[3] for t in transitions])
            nxt_states = np.asarray([t[4] for t in transitions])
            # convert to tensor
            states = torch.as_tensor(states, dtype=torch.float32).to(device)
            actions = torch.as_tensor(actions, dtype=torch.float32).to(device)
            rewards = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device)
            terminateds = torch.as_tensor(terminateds, dtype=torch.uint8).unsqueeze(-1).to(device)
            nxt_states = torch.as_tensor(nxt_states, dtype=torch.float32).to(device)
            # Computer Targets
            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            nxt_actions = actor_target_net(nxt_states)
            Q_targets_next = critic_target_net(nxt_states, nxt_actions)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (GAMMA * Q_targets_next * (1 - terminateds))
            # Compute critic loss
            Q_expected = critic_online_net(states, actions)
            critic_loss = nn.functional.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = actor_online_net(states)
            actor_loss = -critic_online_net(states, actions_pred).mean()
            # Minimize the loss
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            if reward != -100:
                total_distance += reward

            soft_update(actor_online_net, actor_target_net, tau=TAU)
            soft_update(critic_online_net, critic_target_net, tau=TAU)

            if actor_loss is not None:
                actor_losses.append(actor_loss)
            if critic_loss is not None:
                critic_losses.append(critic_loss)

            if terminated or truncated:
                distance_buffer.append(total_distance)
                reward_buffer.append(episode_reward)
                break
        print(f'Episode: {ep},\tScore: {round(episode_reward, 4)},\tDistance: {round(total_distance, 4)},'
              f'\tActor Loss: '
              f'Score: {torch.mean(torch.stack(actor_losses))}'
              f'\tCritic Loss: {torch.mean(torch.stack(critic_losses))}')

        scores.append(total_reward)
        distances.append(total_distance)
        last_scores.append(total_reward)
        last_distance.append(total_distance)
        mean_score = np.mean(last_scores)
        mean_distance = np.mean(last_distance)

        # record rewards dynamically
        file_name = 'records.dat'
        data = [ep, total_reward, total_distance]
        with open(file_name, "ab") as f:
            pickle.dump(data, f)

        if mean_score >= 300:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(ep, mean_score))
            torch.save(actor_online_net.state_dict(), SAVE_DIR + '/best/actor_online_ep' + str(ep) + '.pth')
            torch.save(actor_target_net.state_dict(), SAVE_DIR + '/best/actor_online_ep' + str(ep) + '.pth')
            torch.save(critic_online_net.state_dict(), SAVE_DIR + '/best/critic_online_ep' + str(ep) + '.pth')
            torch.save(critic_target_net.state_dict(), SAVE_DIR + '/best/critic_target_ep' + str(ep) + '.pth')
            break

        # save model every MEAN_EVERY episodes
        if ep % SAVE_FREQ == 0:
            torch.save(actor_online_net.state_dict(), SAVE_DIR + '/actor_online_ep' + str(ep) + '.pth')
            torch.save(actor_target_net.state_dict(), SAVE_DIR + '/actor_online_ep' + str(ep) + '.pth')
            torch.save(critic_online_net.state_dict(), SAVE_DIR + '/critic_online_ep' + str(ep) + '.pth')
            torch.save(critic_target_net.state_dict(), SAVE_DIR + '/critic_target_ep' + str(ep) + '.pth')
            mean_scores.append(mean_score)
            mean_distances.append(mean_distance)
            FILE = 'record_mean.dat'
            data = [ep, mean_score, mean_distance]
            with open(FILE, "ab") as f:
                pickle.dump(data, f)
    env.close()
