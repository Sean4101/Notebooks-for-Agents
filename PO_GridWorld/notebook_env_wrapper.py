import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PO_grid_world import PO_GridWorld

class NotebookEnvWrapper(gym.Env):
    def __init__(self, env, notebook_size=8):
        self.env = env

        if type(env.observation_space) != spaces.Discrete:
            raise ValueError("Only Discrete observation spaces are supported for now")
        if type(env.action_space) != spaces.Discrete:
            raise ValueError("Only Discrete action spaces are supported for now")
        
        self.notebook_size = notebook_size
        self.notebook = np.zeros(self.notebook_size, dtype=np.int32)
        
        self.observation_space = spaces.MultiDiscrete([env.observation_space.n] + [2] * self.notebook_size)
        self.action_space = spaces.MultiDiscrete([env.action_space.n, self.notebook_size, 2])

    def reset(self, seed=None, options=None):
        env_obs, info = self.env.reset(seed, options)
        self.notebook = np.zeros(self.notebook_size, dtype=np.int32)
        return np.concatenate([[env_obs], self.notebook]), info

    def step(self, action):
        env_action, notebook_index, notebook_value = action
        env_obs, reward, done, truncated, info = self.env.step(env_action)
        self.notebook[notebook_index] = notebook_value
        return np.concatenate([[env_obs], self.notebook]), reward, done, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed):
        return self.env.seed(seed)

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def __str__(self):
        return str(self.env)

    def __repr__(self):
        return repr(self.env)
    
if __name__ == "__main__":
    env = PO_GridWorld(partially_observable=True)
    wrapped_env = NotebookEnvWrapper(env)
    print(wrapped_env.observation_space)
    print(wrapped_env.action_space)