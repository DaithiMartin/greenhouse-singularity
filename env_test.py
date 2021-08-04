import gym

if __name__ == '__main__':
    env = gym.make("gym_greenhouse:greenhouse-v0")
    action = 3
    env.reset()
    for i in range(24):
        env.step(action)

    env.render()