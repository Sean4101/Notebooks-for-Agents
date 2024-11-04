from PO_grid_world import PO_GridWorld
from notebook_env_wrapper import NotebookEnvWrapper
from stable_baselines3 import PPO

env = PO_GridWorld(partially_observable=True)
env_notebook = NotebookEnvWrapper(env)

model = PPO.load("models_cmp/ppo_gridworld_notebook_0")

while True:
    state, _ = env_notebook.reset()
    print(state)
    G = 0
    while True:
        env_notebook.render()
        print(env_notebook.notebook)
        action, _ = model.predict(state)
        state, reward, done, _, _ = env_notebook.step(action)
        G += reward
        input("Press Enter to continue...")
        if done:
            break
    
    print(f"Episode return: {G}")
    input("Press Enter to continue...")
