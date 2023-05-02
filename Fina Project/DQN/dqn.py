import os
import pickle
import random
from collections import deque
from dotenv import load_dotenv
from model import QNet
import gymnasium as gym
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


def create_plot(x_axis, y_axis, y_min=-300, y_max=300):
    new_plot = plt.figure()
    x_val, y_val = [], []
    sub_plot = new_plot.add_subplot()
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    x_l = x_axis.replace("#", "")
    plt.title(f"{y_axis} vs. {x_l}")
    plot_line, = sub_plot.plot(x_val, y_val)
    sub_plot.set_xlim([0, total_episodes])
    sub_plot.set_ylim([y_min, y_max])
    return new_plot, plot_line, x_val, y_val


def plot_episode(plot, plt_name, x_val, y_val, total_reward, plot_line, sub_plot_counter):
    x_val.append(sub_plot_counter)
    y_val.append(total_reward)

    plot_line.set_xdata(x_val)
    plot_line.set_ydata(y_val)
    plot.savefig(f"{save_path}/{plt_name}")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # load hyperparameters
    load_dotenv(dotenv_path="./results_1/.env")
    gamma = float(os.getenv('gamma'))  # discount factor
    batch_size = int(os.getenv('batch_size'))  # samples from the replay buffer
    buffer_size = int(os.getenv('buffer_size'))  # max size of the replay buffer before overriding it
    min_replay_size = int(os.getenv('min_replay_size'))
    eps_start = float(os.getenv('eps_start'))
    eps_end = float(os.getenv('eps_end'))
    eps_decay = float(os.getenv('eps_decay'))  # epsilon reach from 1 to 0.02 linearly over epsilon_decay steps
    target_update_freq = int(os.getenv('target_update_freq'))  # number of steps we equalled online parameter to targets
    lr = float(os.getenv('lr'))
    total_episodes = int(os.getenv('total_episodes'))
    episode_len = int(os.getenv('episode_len'))
    save_freq = int(os.getenv('save_freq'))
    save_path = str(os.getenv('save_path'))
    discretize_level = int(os.getenv('discretize_level'))

    episode_score_plt, episode_score_line, x_episode_score, y_episode_score = create_plot("#Episode", "Score")
    episode_avg_loss_plt, episode_avg_loss_line, x_episode_avg_loss, y_episode_avg_loss = create_plot("#Episode",
                                                                                                      "Avg Loss",
                                                                                                      y_min=0,
                                                                                                      y_max=10)
    episode_distance_plt, episode_distance_line, x_episode_distance, y_episode_distance = create_plot("#Episode",
                                                                                                      "distance",
                                                                                                      y_min=-300,
                                                                                                      y_max=350)

    env = gym.make("BipedalWalker-v3", autoreset=True, max_episode_steps=episode_len)
    replay_buffer = deque(maxlen=buffer_size)

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    main_net = QNet(n_states=n_states, n_actions=n_actions, seed=0, discretize_level=discretize_level).to(device)
    target_net = QNet(n_states=n_states, n_actions=n_actions, seed=0, discretize_level=discretize_level).to(device)

    optimizer = torch.optim.Adam(main_net.parameters(), lr=lr)

    start_episode = 0
    # initialize Replay Buffer
    state, _ = env.reset()

    # save min transitions into replay buffer
    for _ in range(min_replay_size):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        transition = (state, action, reward, next_state)
        replay_buffer.append(transition)
        state = next_state
        if terminated or truncated:
            state, _ = env.reset()

    total_rewards = []
    total_distances = []
    total_losses = []
    highest_score = -300
    for ep in range(start_episode + 1, total_episodes + 1):
        # rewards earned by the agents in a single episode to track the improvements of the agents as it trains
        reward_buffer = deque(maxlen=episode_len)
        distance_buffer = deque(maxlen=episode_len)

        epsilon = np.interp(ep, [0, eps_decay], [eps_start, eps_end])
        state, info = env.reset()
        episode_reward = 0
        episode_distance = 0
        losses = []

        for step in range(episode_len):
            # select action by epsilon-greedy policy
            action = main_net.act(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device), epsilon)

            new_state, reward, terminated, truncated, _ = env.step(action)
            # add the transition into replay buffer
            transition = (state, action, reward, new_state)
            replay_buffer.append(transition)
            state = new_state
            episode_reward += reward

            # start gradient step
            # get a batch from the replay buffer
            transitions = random.sample(replay_buffer, batch_size)

            states = np.asarray([t[0] for t in transitions])
            actions = np.asarray([t[1] for t in transitions])
            rewards = np.asarray([t[2] for t in transitions])
            next_states = np.asarray([t[3] for t in transitions])
            # convert to tensor
            states = torch.as_tensor(states, dtype=torch.float32).to(device)
            actions = torch.as_tensor(actions, dtype=torch.long).to(device)
            rewards = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device)
            next_states = torch.as_tensor(next_states, dtype=torch.float32).to(device)
            # Compute Targets
            target_q_val, _ = torch.max(target_net(next_states), dim=1, keepdim=True)
            target_q_val = rewards + gamma * target_q_val
            # Compute Loss
            q_val, _ = torch.max(main_net(next_states), dim=1, keepdim=True)
            loss = nn.functional.smooth_l1_loss(q_val, target_q_val)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
            if reward != -100:
                episode_distance += reward

            if step % target_update_freq == 0:
                target_net.load_state_dict(main_net.state_dict())

            distance_buffer.append(episode_distance)
            reward_buffer.append(episode_reward)

            if terminated or truncated:
                if episode_reward > highest_score:
                    highest_score = episode_reward
                break
        m_episode_reward = np.mean(reward_buffer)
        m_distance = np.mean(distance_buffer)
        m_loss = np.mean(losses)
        print(f'Episode: {ep},\tReward: {episode_reward},\tDistance: {episode_distance}, \tEpsilon: {epsilon}, \tAvg '
              f'Reward: {m_episode_reward}, \tAvg Distance: {m_distance}, \t Avg Loss: {m_loss}')

        total_rewards.append(episode_reward)
        total_distances.append(episode_distance)
        total_losses.append(m_loss)
        plot_episode(episode_score_plt, "episode_score", x_episode_score, y_episode_score, episode_reward,
                     episode_score_line, ep)
        plot_episode(episode_avg_loss_plt, "episode_avg_loss", x_episode_avg_loss, y_episode_avg_loss, m_loss,
                     episode_avg_loss_line, ep)
        plot_episode(episode_distance_plt, "episode_distance", x_episode_distance, y_episode_distance, episode_distance,
                     episode_distance_line, ep)
        # record episode data
        file_name = save_path + '/records.dat'
        data = [ep, episode_reward, episode_distance, epsilon, m_episode_reward, m_distance, np.mean(losses)]
        with open(file_name, "ab") as f:
            pickle.dump(data, f)

        # save model every save_freq episodes
        if ep % save_freq == 0:
            torch.save(main_net.state_dict(), save_path + '/online_ep' + str(ep) + '.pth')
            torch.save(target_net.state_dict(), save_path + '/target_ep' + str(ep) + '.pth')
            file_mean = save_path + '/records_mean.dat'
            data = [ep, np.mean(total_rewards), np.mean(total_distances), epsilon]
            with open(file_mean, "ab") as f:
                pickle.dump(data, f)

        if m_episode_reward >= 300:
            print(f'Solution has been found in episode: {ep}\tAverage Solution Reward: {m_episode_reward}')
            torch.save(main_net.state_dict(), save_path + '/best/online_ep' + str(ep) + '.pth')
            torch.save(target_net.state_dict(), save_path + '/best/target_ep' + str(ep) + '.pth')
            break
    print(save_path, " highest score:", highest_score)
    env.close()
