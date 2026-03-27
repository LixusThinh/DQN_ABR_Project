import os

import matplotlib.pyplot as plt

from evaluate import RESULTS_DIR, evaluate_all_agents


OUTPUT_PATH = os.path.join(RESULTS_DIR, "average_reward_comparison.png")


def plot_average_reward(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    agent_names = list(results["agents"].keys())
    average_rewards = [
        results["agents"][agent_name]["summary"]["average_reward"]
        for agent_name in agent_names
    ]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(agent_names, average_rewards, color=["steelblue", "darkorange", "seagreen"])
    plt.title("Average Reward Comparison")
    plt.xlabel("Agent")
    plt.ylabel("Average Reward")
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, average_rewards):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.close()

    print(f"Saved plot to: {OUTPUT_PATH}")


def main():
    results = evaluate_all_agents()
    plot_average_reward(results)


if __name__ == "__main__":
    main()
