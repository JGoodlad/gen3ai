import gym
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.maskable.utils import get_action_masks

class DummyMaskEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(11)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.step_count = 0
        
    def action_masks(self):
        mask = np.zeros(11, dtype=np.int8)
        # Toggle mask based on step
        mask[self.step_count % 11] = 1
        return mask
        
    def step(self, action):
        mask = self.action_masks()
        if mask[action] == 0:
            print(f"ERROR! Sampled {action} but mask is {mask}. (Step {self.step_count})")
        self.step_count += 1
        return np.array([0]), 0, False, {}
        
    def reset(self):
        self.step_count = 0
        return np.array([0])

def make_env():
    return lambda: DummyMaskEnv()

if __name__ == "__main__":
    env = SubprocVecEnv([make_env() for _ in range(2)])
    obs = env.reset()
    for _ in range(5):
        masks = get_action_masks(env)
        # pretend we sample valid action
        actions = [np.argmax(m) for m in masks]
        # BUT what if the environment step count changes?
        env.step(actions)
        
