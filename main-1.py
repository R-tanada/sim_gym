import gymnasium as gym

env = gym.make('CartPole-v1')

state = env.reset()
# state
# -positon
# -velocity
# -angle positon
# -angular velocity
print(state[0])

action_space = env.action_space
# action
# 0:left
# 1:right
print(action_space)

action = 0
next_state, reward, done, info = env.step(action)[0:4]
# done:finish->True
# 
print(next_state)

