import gymnasium as gym
import numpy as np
from collections import deque

env = gym.make('CartPole-v1', render_mode = "human")
state, info = env.reset()
done = False

while not done:
    env.render()
    action = np.random.choice([0, 1])
    next_state, reward, done, truncated, info = env.step(action)

env.close()
