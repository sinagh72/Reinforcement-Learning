import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from agent import Agent


def plot_episode(plot, x_val, y_val, total_reward, plot_line, sub_plot_counter):
    x_val.append(sub_plot_counter)
    y_val.append(total_reward)

    plot_line.set_xdata(x_val)
    plot_line.set_ydata(y_val)
    plot.savefig("./plot")


if __name__ == "__main__":
    EPISODES = 10000
    SAVE_FREQ = 1000
    MIN_EPS = 0.1
    agent_plot = plt.figure()
    x_val, y_val = [], []
    sub_plot = agent_plot.add_subplot()
    plt.xlabel("#Episode")
    plt.ylabel("Score")
    plt.title("Scores vs. Episode")
    plot_line, = sub_plot.plot(x_val, y_val)
    sub_plot.set_xlim([0, EPISODES])
    sub_plot.set_ylim([-300, 300])

    env = gym.make("BipedalWalker-v3", max_episode_steps=2000)
    env.reset()

    total_exploring_episode = 10000
    # gamma is the discount factor
    agent = Agent(discretize_levels=20, gamma=0.999, alpha=1e-5, total_exploring_episode=total_exploring_episode)
    for i in range(1, EPISODES + 1):
        total_reward, failed, max_reached, eps = agent.Q_learning(env, i, MIN_EPS)
        end_str = "Termination" if failed else "Truncation"
        print(f"Episode {i} finished with '{end_str}', total reward: {total_reward}, epsilon: {eps}")
        plot_episode(agent_plot, x_val, y_val, total_reward, plot_line, i)
        if i % SAVE_FREQ == 0:
            np.save(f'Q_{i}_episodes', np.array(dict(agent.Q)))

    print(f"All episodes finished!")
    print(f"Highest score: {agent.high_score}")
