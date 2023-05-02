import itertools
import os
import pickle
import random
from collections import deque
import gymnasium as gym
import numpy as np
import torch
from dotenv import load_dotenv
from torch import nn
from model import ActorNet, CriticNet
from noise import OrnsteinUhlenbeckNoise
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


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def soft_update(main_net, target_net, tau):
    for name, param in target_net.state_dict().items():
        update_param = (tau * main_net.state_dict()[name] + (1.0 - tau) * param)
        param.copy_(update_param)


if __name__ == "__main__":
    load_dotenv(dotenv_path="./results_4/.env")
    gamma = float(os.getenv('gamma'))  # discount factor
    batch_size = int(os.getenv('batch_size'))  # samples from the replay buffer
    buffer_size = int(os.getenv('buffer_size'))  # max size of the replay buffer before overriding it
    min_replay_size = int(os.getenv('min_replay_size'))
    lr_actor = float(os.getenv('lr_actor'))
    lr_critic = float(os.getenv('lr_critic'))
    tau = float(os.getenv('tau'))
    weight_decay = float(os.getenv('weight_decay'))
    total_episodes = int(os.getenv('total_episodes'))
    episode_len = int(os.getenv('episode_len'))
    save_freq = int(os.getenv('save_freq'))
    save_path = str(os.getenv('save_path'))
    noise_std = float(os.getenv('noise_std'))
    noise_mean = float(os.getenv('noise_mean'))
    noise_scale = float(os.getenv('noise_scale'))

    episode_score_plt, episode_score_line, x_episode_score, y_episode_score = create_plot("#Episode", "Score")
    episode_avg_critic_loss_plt, episode_avg_critic_loss_line, x_episode_avg_critic_loss, y_episode_avg_critic_loss = create_plot(
        "#Episode", "Avg Critic Loss", y_min=0, y_max=20)
    episode_avg_actor_loss_plt, episode_avg_actor_loss_line, x_episode_avg_actor_loss, y_episode_avg_actor_loss = create_plot(
        "#Episode", "Avg Actor Loss", y_min=-20, y_max=20)
    episode_distance_plt, episode_distance_line, x_episode_distance, y_episode_distance = create_plot("#Episode",
                                                                                                      "distance",
                                                                                                      y_min=-300,
                                                                                                      y_max=350)

    env = gym.make("BipedalWalker-v3", autoreset=True, max_episode_steps=episode_len)
    replay_buffer = deque(maxlen=buffer_size)

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor_main_net = ActorNet(n_states=n_states, n_actions=n_actions, seed=0).to(device)
    actor_target_net = ActorNet(n_states=n_states, n_actions=n_actions, seed=0).to(device)
    actor_optimizer = torch.optim.Adam(actor_main_net.parameters(), lr=lr_actor)

    critic_main_net = CriticNet(n_states=n_states, n_actions=n_actions, seed=0).to(device)
    critic_target_net = CriticNet(n_states=n_states, n_actions=n_actions, seed=0).to(device)
    critic_optimizer = torch.optim.Adam(critic_main_net.parameters(), lr=lr_critic, weight_decay=weight_decay)

    noise = OrnsteinUhlenbeckNoise(n_actions, mu=noise_mean, theta=noise_scale, sigma=noise_std)
    # initialize Replay Buffer
    state, _ = env.reset()

    # save min transitions into replay buffer
    for _ in range(min_replay_size):
        action = env.action_space.sample()
        new_state, reward, terminated, truncated, _ = env.step(action)
        transition = (state, action, reward, terminated, new_state)
        replay_buffer.append(transition)
        state = new_state
        if terminated or truncated:
            state, _ = env.reset()

    highest_score = -300
    # rewards earned by the agents in a single episode to track the improvements of the agents as it trains
    reward_buffer = deque(maxlen=total_episodes)
    distance_buffer = deque(maxlen=total_episodes)
    total_reward = 0.0

    scores = []
    mean_scores = []
    last_scores = deque(maxlen=save_freq)
    distances = []
    mean_distances = []
    last_distance = deque(maxlen=save_freq)
    actor_losses = []
    critic_losses = []
    for ep in range(1, total_episodes + 1):
        state, info = env.reset()
        noise.reset()
        episode_reward = 0
        total_distance = 0
        losses = []
        for step in range(episode_len):
            # select action by epsilon-greedy policy
            action = np.clip(actor_main_net.act(
                torch.as_tensor(state, dtype=torch.float32).to(device)) + noise.sample(), a_min=-1, a_max=1)
            # add noise

            nxt_state, reward, terminated, truncated, _ = env.step(action)
            # add the transition into replay buffer
            transition = (state, action, reward, terminated + truncated, nxt_state)
            replay_buffer.append(transition)
            state = nxt_state
            episode_reward += reward

            # start gradient step
            # get a batch from the replay buffer
            transitions = random.sample(replay_buffer, batch_size)

            states = np.asarray([t[0] for t in transitions])
            actions = np.asarray([t[1] for t in transitions])
            rewards = np.asarray([t[2] for t in transitions])
            dones = np.asarray([t[3] for t in transitions])
            nxt_states = np.asarray([t[4] for t in transitions])
            # convert to tensor
            states = torch.as_tensor(states, dtype=torch.float32).to(device)
            actions = torch.as_tensor(actions, dtype=torch.float32).to(device)
            rewards = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device)
            dones = torch.as_tensor(dones, dtype=torch.uint8).unsqueeze(-1).to(device)
            nxt_states = torch.as_tensor(nxt_states, dtype=torch.float32).to(device)
            # ---------------------------- update critic ---------------------------- #
            nxt_actions = actor_target_net(nxt_states)  # batch_size * n_actions
            Q_target_next = critic_target_net(nxt_states, nxt_actions)  # batch_size * 1
            Q_target = rewards + (gamma * Q_target_next * (1 - dones))  # batch_size * 1
            # Compute critic loss
            Q_expected = critic_main_net(states, actions)
            critic_loss = nn.functional.mse_loss(Q_expected, Q_target)
            # Minimize the loss
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = actor_main_net(states)
            actor_loss = -critic_main_net(states, actions_pred).mean()
            # Minimize the loss
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            if reward != -100:
                total_distance += reward

            soft_update(actor_main_net, actor_target_net, tau=tau)
            soft_update(critic_main_net, critic_target_net, tau=tau)

            if actor_loss is not None:
                actor_losses.append(actor_loss)
            if critic_loss is not None:
                critic_losses.append(critic_loss)

            if terminated or truncated:
                distance_buffer.append(total_distance)
                reward_buffer.append(episode_reward)
                if episode_reward > highest_score:
                    highest_score = episode_reward
                break
        print(f'Episode: {ep},\tScore: {round(episode_reward, 4)},\tDistance: {round(total_distance, 4)},'
              f'\tActor Loss: {torch.mean(torch.stack(actor_losses))}'
              f'\tCritic Loss: {torch.mean(torch.stack(critic_losses))}')

        scores.append(total_reward)
        distances.append(total_distance)
        last_scores.append(total_reward)
        last_distance.append(total_distance)
        mean_score = np.mean(last_scores)
        mean_distance = np.mean(last_distance)

        plot_episode(episode_score_plt, "episode_score", x_episode_score, y_episode_score, episode_reward,
                     episode_score_line, ep)
        plot_episode(episode_avg_critic_loss_plt, "episode_avg_critic_loss", x_episode_avg_critic_loss,
                     y_episode_avg_critic_loss, torch.mean(torch.stack(critic_losses)).cpu().detach().numpy(),
                     episode_avg_critic_loss_line, ep)
        plot_episode(episode_avg_actor_loss_plt, "episode_avg_actor_loss", x_episode_avg_actor_loss,
                     y_episode_avg_actor_loss, torch.mean(torch.stack(actor_losses)).cpu().detach().numpy(),
                     episode_avg_actor_loss_line, ep)
        plot_episode(episode_distance_plt, "episode_distance", x_episode_distance, y_episode_distance, total_distance,
                     episode_distance_line, ep)

        if mean_score >= 300:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(ep, mean_score))
            torch.save(actor_main_net.state_dict(), save_path + '/best/actor_online_ep' + str(ep) + '.pth')
            torch.save(actor_target_net.state_dict(), save_path + '/best/actor_online_ep' + str(ep) + '.pth')
            torch.save(critic_main_net.state_dict(), save_path + '/best/critic_online_ep' + str(ep) + '.pth')
            torch.save(critic_target_net.state_dict(), save_path + '/best/critic_target_ep' + str(ep) + '.pth')
            break

        # save model every save_freq episodes
        if ep % save_freq == 0:
            torch.save(actor_main_net.state_dict(), save_path + '/actor_online_ep' + str(ep) + '.pth')
            torch.save(actor_target_net.state_dict(), save_path + '/actor_online_ep' + str(ep) + '.pth')
            torch.save(critic_main_net.state_dict(), save_path + '/critic_online_ep' + str(ep) + '.pth')
            torch.save(critic_target_net.state_dict(), save_path + '/critic_target_ep' + str(ep) + '.pth')

    print(save_path, "highest score:", highest_score)
    env.close()
