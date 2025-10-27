import gymnasium as gym
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen = buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)
    
    def get_batch(self):
        data = np.random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        
        return state, action, reward, next_state, done

env = gym.make('CartPole-v1', render_mode = "human")
replay_bufffer = ReplayBuffer(buffer_size=10000, batch_size=32)

for episode in range(10):
    state, info = env.reset()
    done = False


    while not done:
        env.render()
        action = np.random.choice([0, 1])
        next_state, reward, done, truncated, info = env.step(action)
        replay_bufffer.add(state, action, reward, next_state, done)
        state = next_state

env.close()

