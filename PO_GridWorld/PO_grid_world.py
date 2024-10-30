import gymnasium as gym
import numpy as np
from gymnasium import spaces

class PO_GridWorld(gym.Env):
    # 0: Empty cell
    # 1: Mountain
    # 2: Lightning
    # 3: Treasure chest
    grid = np.array([[0, 0, 0, 0, 0, 3],
                     [0, 1, 1, 2, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 1, 1, 0],
                     [0, 0, 0, 0, 0, 0]])
    # The agent starts at the bottom left corner

    # Directions correspond to the 4 possible actions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    directions_name = ['up', 'down', 'left', 'right']

    def __init__(self, partially_observable=False):
        super(PO_GridWorld, self).__init__()

        self.agent_pos = (5, 0)
        self.partially_observable = partially_observable

        if not self.partially_observable:
            # The agent can see the precise position of itself (6*6)
            self.observation_space = spaces.Discrete(36)
        else:
            # The agent can only see its position in a larger grid (3*3)
            self.observation_space = spaces.Discrete(9)
        self.action_space = spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        self.agent_pos = (5, 0)
        np.random.seed(seed)
        return self.get_observation(), {}
    
    def step(self, action):
        distribution = self.transition_distribution(self.agent_pos, action)
        rand = np.random.rand()
        cumulative_prob = 0

        reward = self.reward(self.agent_pos)
        done = self.is_terminal(self.agent_pos)

        for state, prob in distribution.items():
            cumulative_prob += prob
            if rand < cumulative_prob:
                self.agent_pos = state
                break

        return self.get_observation(), reward, done, False, {}

    def reward(self, state):
        if self.grid[state[0]][state[1]] == 3: # Treasure chest
            return 1
        elif self.grid[state[0]][state[1]] == 2: # Lightning
            return -1
        else:
            return 0
        
    def get_observation(self):
        if self.partially_observable:
            po_pos = (self.agent_pos[0] // 2, self.agent_pos[1] // 2)
            obs = po_pos[0] * 3 + po_pos[1]
            return obs
        else:
            obs = self.agent_pos[0] * 6 + self.agent_pos[1]
            return obs
        
    def is_terminal(self, state):
        if self.grid[state[0]][state[1]] != 0:
            return True
        return False
    
    def is_valid(self, pos):
        if pos[0] < 0 or pos[0] >= self.grid.shape[0] or pos[1] < 0 or pos[1] >= self.grid.shape[1]:
            return False
        if self.grid[pos[0]][pos[1]] == 1: # Mountain
            return False
        return True
    
    def transition_distribution(self, state, action):
        distribution = {}
        distribution[state] = 0

        # If the agent is at a terminal state (mountain, lightning or treasure chest), the agent stays there
        if self.grid[state[0]][state[1]] != 0:
            distribution[state] = 1
            return distribution
    
        # Calculate the distribution of the next state
        for i, dir in enumerate(self.directions):
            new_pos = (state[0] + dir[0], state[1] + dir[1])
            if i == action: # Desired action
                if self.is_valid(new_pos):
                    distribution[new_pos] = 0.85
                else: # If the next state is not valid, stay in the same cell
                    distribution[state] += 0.85
            else: # Other actions
                if self.is_valid(new_pos):
                    distribution[new_pos] = 0.05
                else: # If the next state is not valid, stay in the same cell
                    distribution[state] += 0.05
        
        return distribution
    
    def render(self):
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.agent_pos == (i, j):
                    print('A', end=' ')
                elif self.grid[i][j] == 0:
                    print('.', end=' ')
                elif self.grid[i][j] == 1:
                    print('M', end=' ')
                elif self.grid[i][j] == 2:
                    print('L', end=' ')
                elif self.grid[i][j] == 3:
                    print('T', end=' ')
            print()

    def close(self):
        pass # Nothing to do

if __name__ == '__main__':
    env = PO_GridWorld()
    while True:
        env.render()
        action = int(input('Enter action: '))
        state, reward, done, _ = env.step(action)
        print('State:', state)
        print('Reward:', reward)
        if done:
            print('Game over')
            break