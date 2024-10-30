from stable_baselines3 import DQN
from PO_grid_world import PO_GridWorld

env = PO_GridWorld(partially_observable=False)

model = DQN("MlpPolicy", env, verbose=1,
            learning_rate=0.0001,
            gamma=0.9)
model.learn(total_timesteps=1000000)
model.save("models/dqn_gridworld_normal")