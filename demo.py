import os

from evaluate import load_dqn_agent
from load_trace import load_trace
from video_env import VideoEnv


DEFAULT_TRACE_PATH = "traces/test/trace_test_0.txt"


def run_demo(trace_path: str = DEFAULT_TRACE_PATH, checkpoint_path: str = None):
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Trace not found: {trace_path}")

    agent, resolved_checkpoint = load_dqn_agent(checkpoint_path)
    time_arr, bw_arr = load_trace(trace_path)
    env = VideoEnv(time_arr, bw_arr)
    state = env.reset()
    done = False
    total_reward = 0.0
    step_idx = 0

    print("=" * 72)
    print("DQN Demo Episode")
    print("=" * 72)
    print(f"Checkpoint: {resolved_checkpoint}")
    print(f"Trace: {trace_path}")
    print("-" * 72)

    while not done:
        try:
            action = agent.select_action(state, eval=True)
        except TypeError:
            action = agent.select_action(state, is_training=False)

        next_state, reward, done, info = env.step(action)
        step_idx += 1
        total_reward += reward

        print(
            f"Step {step_idx:2d} | "
            f"Action: {action} | "
            f"Bitrate: {info['bitrate']:7.1f} Kbps | "
            f"Reward: {reward:7.4f} | "
            f"Rebuffer: {info['rebuffer']:6.3f}s | "
            f"Buffer: {info['buffer_size']:6.3f}s"
        )

        state = next_state

    print("-" * 72)
    print(f"Total reward: {total_reward:.4f}")


def main():
    run_demo()


if __name__ == "__main__":
    main()
