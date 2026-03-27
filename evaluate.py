"""
evaluate.py
-----------
Đánh giá 3 thuật toán ABR: DQN vs Rate-Based vs BOLA
trên tập test traces, tính QoE_log và các metrics liên quan.

Cách dùng:
    python evaluate.py \
        --traces_dir  ./traces/test \
        --model_path  ./checkpoints/dqn_final.pth \
        --output_dir  ./results

Kết quả:
    results/metrics.csv        — raw per-episode metrics
    results/summary.txt        — bảng so sánh
    results/cdf_qoe.png        — CDF plot
    results/bar_qoe.png        — bar chart mean QoE ± std
    results/training_curve.png — training reward curve (nếu có log)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # không cần display
import matplotlib.pyplot as plt
import torch

from video_env      import VideoEnv
from load_trace     import load_trace          # → (time_arr, bw_arr)
from dqn_agent      import DQNAgent
from rate_based_agent import RateBasedAgent
from bola_agent     import BOLAAgent


# ─────────────────────────────────────────────
# CONSTANTS  (kế thừa từ Pensieve / project)
# ─────────────────────────────────────────────
BITRATES     = [300, 750, 1200, 1850, 2850, 4300]   # kbps, 6 levels
R_MIN        = BITRATES[0]                            # 300 kbps
REBUF_PENALTY = 2.66                                  # QoE_log
SMOOTH_PENALTY = 1.0                                  # |Δq|


# ─────────────────────────────────────────────
# QoE helpers
# ─────────────────────────────────────────────
def bitrate_utility(bitrate_kbps: float) -> float:
    """
    q(R) = log(R / R_min)
    Ánh xạ bitrate → perceived quality theo QoE_log (Pensieve Table 1).
    """
    return np.log(bitrate_kbps / R_MIN)


def compute_qoe(bitrates_seq: list,
                rebuf_seq:    list,
                smooth_penalty: float = SMOOTH_PENALTY) -> float:
    """
    Tính tổng QoE_log cho 1 episode theo Equation 6 (Pensieve §5.1):

        QoE = Σ q(Rₙ) - μ·Σ Tₙ - Σ |q(Rₙ₊₁) - q(Rₙ)|

    Args:
        bitrates_seq : list[float]  — bitrate (kbps) của từng chunk
        rebuf_seq    : list[float]  — rebuffer time (s) của từng chunk
        smooth_penalty: float       — hệ số phạt smoothness (default 1.0)

    Returns:
        float — tổng QoE (chưa chia số chunk)
    """
    # Phần 1: Σ q(Rₙ) — tổng utility
    quality_sum = sum(bitrate_utility(r) for r in bitrates_seq)

    # Phần 2: μ·Σ Tₙ — rebuffer penalty
    rebuf_sum = REBUF_PENALTY * sum(rebuf_seq)

    # Phần 3: Σ |q(Rₙ₊₁) - q(Rₙ)| — smoothness penalty
    smoothness = 0.0
    for i in range(len(bitrates_seq) - 1):
        smoothness += abs(
            bitrate_utility(bitrates_seq[i + 1]) -
            bitrate_utility(bitrates_seq[i])
        )
    smoothness *= smooth_penalty

    return quality_sum - rebuf_sum - smoothness


# ─────────────────────────────────────────────
# Core evaluation loop
# ─────────────────────────────────────────────
def run_episode(env: VideoEnv, agent) -> dict:
    """
    Chạy 1 episode đầy đủ với agent bất kỳ.

    env.step(action) → (next_state, reward, done, info)
    Trong đó info được expect chứa:
        info['bitrate']  : float — bitrate đã chọn (kbps)
        info['rebuffer'] : float — rebuffer time (s) ở bước này

    Nếu VideoEnv của mày không trả info như trên thì
    cần điều chỉnh phần parse bên dưới.

    Returns dict với:
        'qoe'              : float  — tổng QoE episode
        'mean_qoe'         : float  — QoE / số chunk
        'mean_bitrate'     : float  — bitrate trung bình (kbps)
        'rebuf_ratio'      : float  — rebuf_total / episode_length (s)
        'switch_count'     : int    — số lần đổi bitrate
        'bitrate_seq'      : list   — chuỗi bitrate qua từng chunk
        'rebuf_seq'        : list   — chuỗi rebuffer time
    """
    state = env.reset()          # shape (6, 8)

    bitrate_seq = []
    rebuf_seq   = []
    delay_seq = []
    buffer_size_seq = []
    episode_length = 0.0         # tổng thời gian video (s)
    done = False

    while not done:
        # ── chọn action ──────────────────────────────────────────
        action = agent.select_action(state)

        # ── bước môi trường ──────────────────────────────────────
        next_state, reward, done, info = env.step(action)

        # ── lấy metrics từ info ──────────────────────────────────
        # Adjust field names nếu VideoEnv của mày dùng key khác
        bitrate_kbps = info.get('bitrate', BITRATES[action])
        rebuf_time   = info.get('rebuffer', 0.0)
        chunk_time   = info.get('chunk_duration', 4.0)  # mặc định 4s/chunk
        delay        = info.get('delay', 0.0)
        buffer_size  = info.get('buffer_size', 0.0)

        bitrate_seq.append(bitrate_kbps)
        rebuf_seq.append(rebuf_time)
        delay_seq.append(delay)
        buffer_size_seq.append(buffer_size)
        episode_length += chunk_time + rebuf_time

        state = next_state

    # ── tính metrics ─────────────────────────────────────────────
    n_chunks    = len(bitrate_seq)
    total_rebuf = sum(rebuf_seq)
    qoe         = compute_qoe(bitrate_seq, rebuf_seq)

    # đếm số lần đổi bitrate
    switches = sum(
        1 for i in range(n_chunks - 1)
        if bitrate_seq[i] != bitrate_seq[i + 1]
    )

    return {
        'qoe'          : qoe,
        'mean_qoe'     : qoe / max(n_chunks, 1),
        'mean_bitrate' : float(np.mean(bitrate_seq)),
        'rebuf_ratio'  : total_rebuf / max(episode_length, 1e-6),
        'mean_delay'   : float(np.mean(delay_seq)),
        'mean_buffer_size': float(np.mean(buffer_size_seq)),
        'switch_count' : switches,
        'bitrate_seq'  : bitrate_seq,
        'rebuf_seq'    : rebuf_seq,
    }


def evaluate_algo(algo_name:  str,
                  agent,
                  traces:     list,
                  verbose:    bool = True) -> pd.DataFrame:
    """
    Chạy agent trên toàn bộ test traces.

    Args:
        algo_name : tên algo (dùng để label)
        agent     : object có .select_action(state)
        traces    : list of (time_arr, bw_arr)
        verbose   : in progress hay không

    Returns:
        DataFrame với 1 row / trace, columns là các metrics
    """
    records = []

    for i, (time_arr, bw_arr) in enumerate(traces):
        env = VideoEnv(time_arr, bw_arr)
        result = run_episode(env, agent)

        records.append({
            'algo'         : algo_name,
            'trace_id'     : i,
            'qoe'          : result['qoe'],
            'mean_qoe'     : result['mean_qoe'],
            'mean_bitrate' : result['mean_bitrate'],
            'rebuf_ratio'  : result['rebuf_ratio'],
            'mean_delay'   : result['mean_delay'],
            'mean_buffer_size': result['mean_buffer_size'],
            'switch_count' : result['switch_count'],
        })

        if verbose:
            print(f"  [{algo_name}] trace {i+1:02d}/{len(traces)}"
                  f"  QoE={result['mean_qoe']:.4f}"
                  f"  bitrate={result['mean_bitrate']:.0f}kbps"
                  f"  rebuf={result['rebuf_ratio']*100:.2f}%"
                  f"  switches={result['switch_count']}")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────
ALGO_COLORS = {
    'DQN'        : '#E63946',   # đỏ — algo chính của mày
    'Rate-Based' : '#457B9D',   # xanh dương
    'BOLA'       : '#2A9D8F',   # xanh lá
}
ALGO_LINESTYLES = {
    'DQN'        : '-',
    'Rate-Based' : '--',
    'BOLA'       : '-.',
}


def plot_cdf(df: pd.DataFrame, output_path: str):
    """
    CDF của mean_qoe per episode cho 3 algo.
    Giống Figure 7/8 trong Pensieve §5.2.
    Đường shift phải = algo tốt hơn.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for algo in df['algo'].unique():
        data = np.sort(df[df['algo'] == algo]['mean_qoe'].values)
        cdf  = np.arange(1, len(data) + 1) / len(data)
        ax.plot(data, cdf,
                label=algo,
                color=ALGO_COLORS.get(algo, 'gray'),
                linestyle=ALGO_LINESTYLES.get(algo, '-'),
                linewidth=2)

    ax.set_xlabel('Mean QoE per Chunk (QoE$_{log}$)', fontsize=13)
    ax.set_ylabel('CDF',                               fontsize=13)
    ax.set_title('CDF of QoE — DQN vs Baselines',     fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[plot] CDF saved → {output_path}")


def plot_bar(df: pd.DataFrame, output_path: str):
    """
    Bar chart: mean ± std của mean_qoe cho 3 algo.
    Dễ đọc, thường dùng để so sánh tổng hợp trong báo cáo.
    """
    algos  = df['algo'].unique()
    means  = [df[df['algo'] == a]['mean_qoe'].mean() for a in algos]
    stds   = [df[df['algo'] == a]['mean_qoe'].std()  for a in algos]
    colors = [ALGO_COLORS.get(a, 'gray') for a in algos]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(algos, means, yerr=stds,
                  color=colors, capsize=6,
                  edgecolor='black', linewidth=0.8, alpha=0.85)

    # ghi số lên đầu bar
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{mean:.4f}',
                ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Mean QoE per Chunk (QoE$_{log}$)', fontsize=13)
    ax.set_title('Average QoE Comparison',             fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[plot] Bar chart saved → {output_path}")


def plot_training_curve(log_path: str, output_path: str,
                        window: int = 50):
    """
    Vẽ training reward curve từ log file.
    Expect log file có dạng: mỗi dòng là 1 episode reward (float).
    Nếu format log của mày khác thì sửa phần đọc file ở đây.

    window: rolling mean window size để smooth
    """
    if not os.path.exists(log_path):
        print(f"[skip] training log không tìm thấy: {log_path}")
        return

    rewards = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # thử parse số trực tiếp
                rewards.append(float(line))
            except ValueError:
                # nếu log có format "episode X reward Y" thì lấy số cuối
                parts = line.split()
                try:
                    rewards.append(float(parts[-1]))
                except (ValueError, IndexError):
                    pass

    if not rewards:
        print("[skip] không parse được reward từ log file")
        return

    rewards  = np.array(rewards)
    episodes = np.arange(1, len(rewards) + 1)

    # rolling mean
    if len(rewards) >= window:
        smoothed = np.array(pd.Series(rewards).rolling(window, min_periods=1).mean())
    else:
        smoothed = rewards

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, rewards,
            alpha=0.25, color='#457B9D', linewidth=0.8, label='Raw')
    ax.plot(episodes, smoothed,
            color='#E63946', linewidth=2,
            label=f'Rolling mean (w={window})')

    ax.set_xlabel('Episode',        fontsize=13)
    ax.set_ylabel('Total Reward',   fontsize=13)
    ax.set_title('DQN Training Curve', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[plot] Training curve saved → {output_path}")


def print_summary(df: pd.DataFrame):
    """
    In bảng tổng kết ra terminal.
    """
    print("\n" + "=" * 65)
    print(f"{'ALGO':<14} {'Mean QoE':>10} {'Avg kbps':>10} "
          f"{'Rebuf%':>9} {'Switches':>10}")
    print("-" * 65)

    for algo in df['algo'].unique():
        sub = df[df['algo'] == algo]
        print(f"{algo:<14}"
              f"{sub['mean_qoe'].mean():>10.4f}"
              f"{sub['mean_bitrate'].mean():>10.1f}"
              f"{sub['rebuf_ratio'].mean()*100:>8.2f}%"
              f"{sub['switch_count'].mean():>10.1f}")

    print("=" * 65 + "\n")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load test traces ──────────────────────────────────────
    print(f"[load] đọc traces từ {args.traces_dir} ...")
    trace_files = sorted([
        f for f in os.listdir(args.traces_dir)
        if not f.startswith('.')
    ])

    if not trace_files:
        raise FileNotFoundError(f"Không tìm thấy trace nào trong {args.traces_dir}")

    traces = []
    for fname in trace_files:
        fpath = os.path.join(args.traces_dir, fname)
        time_arr, bw_arr = load_trace(fpath)
        traces.append((time_arr, bw_arr))

    print(f"[load] Loaded {len(traces)} traces\n")

    # ── 2. Khởi tạo agents ──────────────────────────────────────
    # --- DQN: load model, bắt buộc epsilon = 0 khi eval -----------
    print(f"[init] DQN ← {args.model_path}")
    dqn_agent = DQNAgent()                          # khởi tạo với default config
    dqn_agent.policy_net.load_state_dict(
        torch.load(args.model_path, map_location='cpu')
    )
    dqn_agent.policy_net.eval()   # tắt dropout/batchnorm nếu có
    dqn_agent.epsilon = 0.0       # QUAN TRỌNG: tắt exploration hoàn toàn

    rate_agent = RateBasedAgent()
    bola_agent = BOLAAgent()

    agents = [
        ('DQN',         dqn_agent),
        ('Rate-Based',  rate_agent),
        ('BOLA',        bola_agent),
    ]

    # ── 3. Chạy evaluation ───────────────────────────────────────
    all_dfs = []
    for algo_name, agent in agents:
        print(f"\n[eval] {algo_name} ...")
        df_algo = evaluate_algo(algo_name, agent, traces, verbose=True)
        all_dfs.append(df_algo)

    df = pd.concat(all_dfs, ignore_index=True)

    # ── 4. Lưu raw metrics ───────────────────────────────────────
    csv_path = os.path.join(args.output_dir, 'metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n[save] metrics → {csv_path}")

    # ── 5. In bảng tổng kết ──────────────────────────────────────
    print_summary(df)

    # lưu summary ra file text
    summary_path = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        algos = df['algo'].unique()
        f.write(f"{'ALGO':<14} {'Mean QoE':>10} {'Avg kbps':>10} "
                f"{'Rebuf%':>9} {'Switches':>10}\n")
        f.write("-" * 65 + "\n")
        for algo in algos:
            sub = df[df['algo'] == algo]
            f.write(f"{algo:<14}"
                    f"{sub['mean_qoe'].mean():>10.4f}"
                    f"{sub['mean_bitrate'].mean():>10.1f}"
                    f"{sub['rebuf_ratio'].mean()*100:>8.2f}%"
                    f"{sub['switch_count'].mean():>10.1f}\n")
    print(f"[save] summary → {summary_path}")

    # ── 6. Plots ─────────────────────────────────────────────────
    plot_cdf(df, os.path.join(args.output_dir, 'cdf_qoe.png'))
    plot_bar(df, os.path.join(args.output_dir, 'bar_qoe.png'))

    if args.train_log:
        plot_training_curve(
            args.train_log,
            os.path.join(args.output_dir, 'training_curve.png')
        )

    print("\n[done] Evaluation hoàn tất.")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate DQN ABR vs baselines'
    )
    parser.add_argument('--traces_dir',  required=True,
                        help='Thư mục chứa test traces')
    parser.add_argument('--model_path',  required=True,
                        help='Path tới file .pth của DQN')
    parser.add_argument('--output_dir',  default='./results',
                        help='Thư mục lưu kết quả (default: ./results)')
    parser.add_argument('--train_log',   default=None,
                        help='(Optional) Path tới training reward log file')

    args = parser.parse_args()
    main(args)
