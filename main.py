import numpy as np

from agents.PDQN_Agent import Agent
import gym
import matplotlib.pyplot as plt

if __name__ == '__main__':

    env = gym.make("gym_greenhouse:greenhouse-v0")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    action_size = env.action_space.nvec[0]
    observation_size = env.observation_space.n
    agent = Agent(action_size=action_size, observation_size=observation_size, seed=0)
    rewards_history = []
    num_episodes = 1000

    for i_episode in range(num_episodes):
        observation = env.reset()
        done = False
        while not done:

            action = agent.act(observation)
            next_observation, reward, done, info = env.step(action)
            agent.step(observation, action, reward, next_observation, done)
            observation = next_observation

        rewards_history.append(np.sum(env.reward_history))

    print("Training Complete")
    print("Final Episode:")

    env.render()

    x = np.arange(num_episodes)
    y = rewards_history
    plt.plot(x, y)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.show()

