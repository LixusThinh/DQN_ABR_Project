import os

import matplotlib.pyplot as plt
import numpy as np


REWARD_HISTORY_PATH = "models/reward_history.npy"
LOSS_HISTORY_PATH = "models/loss_history.npy"
RESULTS_DIR = "results"
MOVING_AVG_WINDOW = 10


def moving_average(data, window):
    if len(data) == 0:
        return np.array([])
    if len(data) < window:
        return np.array([np.mean(data)])

    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_reward_curve(reward_history):
    episodes = np.arange(1, len(reward_history) + 1)
    reward_ma = moving_average(reward_history, MOVING_AVG_WINDOW)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, reward_history, label="Reward per Episode", alpha=0.5)

    if len(reward_history) >= MOVING_AVG_WINDOW:
        ma_episodes = np.arange(MOVING_AVG_WINDOW, len(reward_history) + 1)
    else:
        ma_episodes = np.array([len(reward_history)])

    plt.plot(
        ma_episodes,
        reward_ma,
        label=f"Moving Average (window={MOVING_AVG_WINDOW})",
        linewidth=2,
        color="red",
    )

    plt.title("DQN Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "reward_curve.png"), dpi=300)
    plt.close()


def plot_loss_curve(loss_history):
    episodes = np.arange(1, len(loss_history) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, loss_history, label="Loss per Episode", color="orange")
    plt.title("DQN Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"), dpi=300)
    plt.close()


def print_statistics(reward_history):
    final_window = min(100, len(reward_history))
    final_mean_reward = np.mean(reward_history[-final_window:])

    print(f"Min reward: {np.min(reward_history):.4f}")
    print(f"Max reward: {np.max(reward_history):.4f}")
    print(f"Final {final_window}-episode mean reward: {final_mean_reward:.4f}")


def main():
    if not os.path.exists(REWARD_HISTORY_PATH):
        raise FileNotFoundError(f"Missing file: {REWARD_HISTORY_PATH}")
    if not os.path.exists(LOSS_HISTORY_PATH):
        raise FileNotFoundError(f"Missing file: {LOSS_HISTORY_PATH}")

    reward_history = np.load(REWARD_HISTORY_PATH)
    loss_history = np.load(LOSS_HISTORY_PATH)

    if reward_history.size == 0:
        raise ValueError("Reward history is empty.")
    if loss_history.size == 0:
        raise ValueError("Loss history is empty.")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    plot_reward_curve(reward_history)
    plot_loss_curve(loss_history)
    print_statistics(reward_history)

    print(f"Saved reward plot to: {os.path.join(RESULTS_DIR, 'reward_curve.png')}")
    print(f"Saved loss plot to: {os.path.join(RESULTS_DIR, 'loss_curve.png')}")


if __name__ == "__main__":
    main()
