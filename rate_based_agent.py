import numpy as np
import os
import random
from load_trace import load_trace
from video_env import VideoEnv

BITRATES = [300.0, 750.0, 1200.0, 1850.0, 2850.0, 4300.0]

class RateBasedAgent:
    def __init__(self, history_size=5, safety_margin=0.8):
        self.history_size = history_size
        self.safety_margin = safety_margin

    def select_action(self, state):
        throughput_history = state[0, :]
        valid_throughputs = throughput_history[throughput_history > 0]

        if len(valid_throughputs) == 0:
            return 0

        recent_throughputs = valid_throughputs[-self.history_size:]
        recent_throughputs_kbps = recent_throughputs * 1000.0
        harmonic_mean_bw = len(recent_throughputs_kbps) / np.sum(1.0 / recent_throughputs_kbps)
        predicted_bw = harmonic_mean_bw * self.safety_margin

        action = 0
        for i in range(len(BITRATES) - 1, -1, -1):
            if BITRATES[i] <= predicted_bw:
                action = i
                break

        return action


if __name__ == "__main__":
    # Load trace ngẫu nhiên
    trace_folder = 'traces/train/'
    trace_files = [f for f in os.listdir(trace_folder) if f.endswith('.txt')]
    chosen_file = random.choice(trace_files)
    print(f"Using trace: {chosen_file}")
    time_arr, bw_arr = load_trace(os.path.join(trace_folder, chosen_file))

    # Khởi tạo env và agent
    env = VideoEnv(time_arr, bw_arr)
    agent = RateBasedAgent(history_size=5, safety_margin=0.8)

    state = env.reset()
    done = False
    episode_reward = 0.0
    step_count = 0
    rewards_history = []

    print("Bắt đầu chạy test RATE-BASED AGENT...")
    print("-" * 70)

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        episode_reward += reward
        rewards_history.append(reward)
        step_count += 1

        print(
            f"Step {step_count:2d} | "
            f"Action: {action} | "
            f"Bitrate: {info['bitrate']:4.0f} Kbps | "
            f"Delay: {info['delay']:5.2f}s | "
            f"Rebuf: {info['rebuffer']:5.2f}s | "
            f"Reward: {reward:6.2f}"
        )

        state = next_state

    print("-" * 70)
    print(f"[DONE] Kết thúc sau {step_count} chunks")
    print(f"Total Reward   : {episode_reward:.3f}")
    print(f"Average Reward : {episode_reward / step_count:.3f}")
    print(f"Max Reward     : {max(rewards_history):.3f}")
    print(f"Min Reward     : {min(rewards_history):.3f}")