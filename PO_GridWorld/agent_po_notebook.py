from stable_baselines3 import PPO
from PO_grid_world import PO_GridWorld
from notebook_env_wrapper import NotebookEnvWrapper

env = PO_GridWorld(partially_observable=True)
wrapped_env = NotebookEnvWrapper(env)

model = PPO("MlpPolicy", wrapped_env, verbose=1,
            learning_rate=0.0001,
            gamma=0.9)
model.learn(total_timesteps=500000)
model.save("models/ppo_gridworld_notebook")