import numpy as np

from agents.PDQN_Agent import PDQNAgent
from agents.DQ_Agent import DQAgent
import gym
import matplotlib.pyplot as plt

if __name__ == '__main__':

    env = gym.make("gym_greenhouse:greenhouse-v0")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    action_size = env.action_space.n
    observation_size = env.observation_space.n
    # agent = DQAgent(observation_size, action_size, seed=0)
    agent = PDQNAgent(observation_size, action_size, seed=0)
    rewards_history = []
    num_episodes = int(5e3)

    for i_episode in range(1, num_episodes):
        observation = env.reset()
        done = False
        while not done:

            action = agent.act(observation)
            next_observation, reward, done, info = env.step(action)
            agent.step(observation, action, reward, next_observation, done)
            observation = next_observation

        if i_episode == 1:
            env.render()

        if i_episode % 100 == 0:
            print(f"Episode: {i_episode}, Average Reward: {np.mean(rewards_history[-100:])}")

        rewards_history.append(np.sum(env.reward_history))

    print("Training Complete")
    env.render()
    y = np.convolve(rewards_history, np.ones(10), 'valid') / 10
    x = np.arange(len(y))
    plt.plot(x, y)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.show()

