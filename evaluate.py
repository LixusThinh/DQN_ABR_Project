import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from bola_agent import BOLAAgent
from dqn_agent import DQNAgent
from load_trace import load_trace
from rate_based_agent import RateBasedAgent
from video_env import VideoEnv


TEST_TRACES_DIR = "traces/test"
RESULTS_DIR = "results"
DEFAULT_CHECKPOINT_CANDIDATES = [
    "models/checkpoint.pth",
    "checkpoint.pth",
    "models/dqn_model.pth",
    "models/dqn_checkpoint_ep1000.pth",
    "models/dqn_checkpoint_ep500.pth",
]


def find_checkpoint(explicit_path: str = None) -> str:
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"Checkpoint not found: {explicit_path}")
        return explicit_path

    for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "No DQN checkpoint found. Checked: "
        + ", ".join(DEFAULT_CHECKPOINT_CANDIDATES)
    )


def get_test_traces(traces_dir: str = TEST_TRACES_DIR) -> List[str]:
    trace_paths = sorted(glob.glob(os.path.join(traces_dir, "*.txt")))
    if not trace_paths:
        raise FileNotFoundError(f"No trace files found in: {traces_dir}")
    return trace_paths


def select_action_compat(agent, state, is_dqn: bool = False) -> int:
    if is_dqn:
        try:
            return agent.select_action(state, eval=True)
        except TypeError:
            return agent.select_action(state, is_training=False)
    return agent.select_action(state)


def load_dqn_agent(checkpoint_path: str = None) -> Tuple[DQNAgent, str]:
    resolved_path = find_checkpoint(checkpoint_path)
    agent = DQNAgent(state_dim=48, action_dim=6)

    state_dict = torch.load(resolved_path, map_location=agent.device)
    agent.policy_net.load_state_dict(state_dict)
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    agent.policy_net.eval()
    agent.target_net.eval()
    agent.epsilon = 0.0

    return agent, resolved_path


def run_episode(agent, trace_path: str, is_dqn: bool = False) -> Dict[str, float]:
    time_arr, bw_arr = load_trace(trace_path)
    env = VideoEnv(time_arr, bw_arr)
    state = env.reset()
    done = False

    total_reward = 0.0
    total_rebuffer = 0.0
    bitrate_history = []
    smoothness_sum = 0.0
    prev_bitrate = None
    num_steps = 0

    while not done:
        action = select_action_compat(agent, state, is_dqn=is_dqn)
        next_state, reward, done, info = env.step(action)

        bitrate = float(info["bitrate"])
        bitrate_history.append(bitrate)
        total_reward += float(reward)
        total_rebuffer += float(info["rebuffer"])

        if prev_bitrate is not None:
            smoothness_sum += abs(bitrate - prev_bitrate)
        prev_bitrate = bitrate

        num_steps += 1
        state = next_state

    average_bitrate = float(np.mean(bitrate_history)) if bitrate_history else 0.0
    average_reward = total_reward / num_steps if num_steps > 0 else 0.0
    average_smoothness = smoothness_sum / max(len(bitrate_history) - 1, 1)

    return {
        "trace": os.path.basename(trace_path),
        "total_reward": total_reward,
        "average_reward": average_reward,
        "average_bitrate": average_bitrate,
        "rebuffer_time": total_rebuffer,
        "smoothness": average_smoothness,
        "num_steps": num_steps,
    }


def summarize_results(episode_results: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        "num_episodes": len(episode_results),
        "average_reward": float(np.mean([r["average_reward"] for r in episode_results])),
        "average_bitrate": float(np.mean([r["average_bitrate"] for r in episode_results])),
        "rebuffer_time": float(np.mean([r["rebuffer_time"] for r in episode_results])),
        "smoothness": float(np.mean([r["smoothness"] for r in episode_results])),
    }


def evaluate_agent(agent_name: str, agent, trace_paths: List[str], is_dqn: bool = False):
    episode_results = [
        run_episode(agent, trace_path, is_dqn=is_dqn) for trace_path in trace_paths
    ]
    summary = summarize_results(episode_results)
    return {
        "agent": agent_name,
        "summary": summary,
        "episodes": episode_results,
    }


def evaluate_all_agents(checkpoint_path: str = None, traces_dir: str = TEST_TRACES_DIR):
    trace_paths = get_test_traces(traces_dir)
    dqn_agent, resolved_checkpoint = load_dqn_agent(checkpoint_path)

    agents = [
        ("DQN", dqn_agent, True),
        ("BOLA", BOLAAgent(), False),
        ("Rate-Based", RateBasedAgent(), False),
    ]

    all_results = {
        "checkpoint_path": resolved_checkpoint,
        "num_traces": len(trace_paths),
        "trace_files": [os.path.basename(path) for path in trace_paths],
        "agents": {},
    }

    for agent_name, agent, is_dqn in agents:
        all_results["agents"][agent_name] = evaluate_agent(
            agent_name, agent, trace_paths, is_dqn=is_dqn
        )

    return all_results


def print_summary(results: Dict) -> None:
    print("=" * 72)
    print("ABR Evaluation Summary")
    print("=" * 72)
    print(f"DQN checkpoint: {results['checkpoint_path']}")
    print(f"Number of test traces: {results['num_traces']}")
    print("-" * 72)
    print(
        f"{'Agent':<12} | {'Avg Reward':>12} | {'Avg Bitrate':>12} | "
        f"{'Rebuffer':>10} | {'Smoothness':>12}"
    )
    print("-" * 72)

    for agent_name, agent_results in results["agents"].items():
        summary = agent_results["summary"]
        print(
            f"{agent_name:<12} | "
            f"{summary['average_reward']:12.4f} | "
            f"{summary['average_bitrate']:12.2f} | "
            f"{summary['rebuffer_time']:10.4f} | "
            f"{summary['smoothness']:12.2f}"
        )

    print("-" * 72)


def save_results(results: Dict, output_path: str = os.path.join(RESULTS_DIR, "evaluation_results.json")) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    return output_path


def main():
    results = evaluate_all_agents()
    print_summary(results)
    output_path = save_results(results)
    print(f"Saved structured results to: {output_path}")


if __name__ == "__main__":
    main()
