import numpy as np
import os
from video_env import VideoEnv
from load_trace import load_trace

BITRATES = [300.0, 750.0, 1200.0, 1850.0, 2850.0, 4300.0]

class BOLAAgent:
    def __init__(self, buffer_min=3.0, buffer_max=15.0):
        self.buffer_min = buffer_min
        self.buffer_max = buffer_max

    def select_action(self, state):
        buffer_size = state[3, -1]

        if buffer_size <= self.buffer_min:
            return 0

        if buffer_size >= self.buffer_max:
            return 5

        ratio = (buffer_size - self.buffer_min) / (self.buffer_max - self.buffer_min)
        action = int(ratio * 6)
        return min(max(action, 0), 5)


# ==================== TEST SCRIPT ====================
if __name__ == "__main__":
    trace_folder = 'traces/train/'
    chosen_file = 'trace_weak_0.txt'
    time_arr, bw_arr = load_trace(os.path.join(trace_folder, chosen_file))

    env = VideoEnv(time_arr, bw_arr)
    agent = BOLAAgent(buffer_min=4.0, buffer_max=16.0)

    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    print(f"{'Step':<5} | {'Buffer':<10} | {'Action':<8} | {'Bitrate':<10} | {'Reward':<10}")
    print("-" * 55)

    while not done:
        step_count += 1
        action = agent.select_action(state)
        current_buffer = state[3, -1]

        next_state, reward, done, info = env.step(action)
        total_reward += reward

        print(f"{step_count:<5} | {current_buffer:<10.2f} | {action:<8} | {info['bitrate']:<10} | {reward:<10.2f}")

        state = next_state

    print("-" * 55)
    print(f"BOLA Total Reward: {total_reward:.2f}")