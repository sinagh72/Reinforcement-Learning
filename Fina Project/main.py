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

    agent_plot = plt.figure()
    x_val, y_val = [], []
    sub_plot = agent_plot.add_subplot()
    plt.xlabel("Episode #")
    plt.ylabel("Score")
    plt.title("Scores vs Episode")
    plot_line, = sub_plot.plot(x_val, y_val)
    sub_plot.set_xlim([0, EPISODES])
    sub_plot.set_ylim([-220, -80])

    env = gym.make("BipedalWalker-v3",  render_mode="human", autoreset=True, max_episode_steps=1000)
    env.reset()

    total_exploring_episode = 400
    # gamma is the discount factor
    agent = Agent(discretize_levels=100, gamma=0.99, learning_rate=1e-3, total_exploring_episode=total_exploring_episode)
    for i in range(1, EPISODES + 1):
        total_reward, failed, max_reached = agent.Q_learning(env, i)
        end_str = "Termination" if failed else "Truncation"
        print(f"Episode {i} finished with '{end_str}', total reward: {total_reward}")
        plot_episode(agent_plot, x_val, y_val, total_reward, plot_line, i)
        if i % total_exploring_episode == 0:
            np.save(f'Q_{i}_episodes', np.array(dict(agent.Q)))

    print(f"All episodes finished!")
    print(f"Highest score: {agent.high_score}")
