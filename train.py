import os
import glob
import numpy as np
import torch

from load_trace import load_trace
from video_env import VideoEnv
from replay_buffer import ReplayBuffer
from dqn_agent import DQNAgent

# ==========================================
# CẤU HÌNH TRAINING VỚI DATA XỊN
# ==========================================

TRAIN_TRACES_DIR = './traces/train/' # Trỏ về kho 90 file của mày
NUM_EPISODES = 1000       # Nâng lên 2000 tập cho bõ công train data xịn
BATCH_SIZE = 64          
TARGET_UPDATE_FREQ = 100 

def main():
    # 1. Quét lấy toàn bộ danh sách file mạng fcc
    all_traces = glob.glob(os.path.join(TRAIN_TRACES_DIR, 'trace_*'))
    if len(all_traces) == 0:
        print(f"LỖI: Không tìm thấy file trace nào trong {TRAIN_TRACES_DIR}")
        return
        
    print(f"Đã load {len(all_traces)} file traces thực tế (FCC) để train.")

    # 2. Khởi tạo các thành phần cốt lõi
    agent = DQNAgent(state_dim=48, action_dim=6)
    checkpoint_path = "models/dqn_checkpoint_ep1000.pth"
    if os.path.exists(checkpoint_path):
        agent.policy_net.load_state_dict(torch.load(checkpoint_path, map_location=agent.device))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        agent.epsilon = 0.135
        print(f"[RESUME] Loaded checkpoint: {checkpoint_path}")
    replay_buffer = ReplayBuffer(capacity=10000)
    
    total_steps = 0
    reward_history = []
    loss_history = []

    print("="*50)
    print(" BẮT ĐẦU HUẤN LUYỆN DQN AGENT (REAL TRACES)")
    print("="*50)

    # 3. Vòng lặp huấn luyện chính
    for episode in range(1, NUM_EPISODES + 1):
        # Bốc ngẫu nhiên 1 file mạng
        trace_file = np.random.choice(all_traces)
        time_arr, bw_arr = load_trace(trace_file)
        
        # Nếu file rỗng hoặc lỗi, bỏ qua
        if len(time_arr) == 0:
            continue

        env = VideoEnv(time_arr, bw_arr)
        state = env.reset()
        
        done = False
        episode_reward = 0.0
        episode_loss = 0.0
        episode_steps = 0
        
        while not done:
            action = agent.select_action(state, is_training=True)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.train_step(replay_buffer, batch_size=BATCH_SIZE)
            if loss is not None:
                episode_loss += loss
            episode_steps += 1
            
            state = next_state
            total_steps += 1
            
            if total_steps % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()
                
        reward_history.append(episode_reward)
        avg_loss = episode_loss / episode_steps if episode_steps > 0 else 0.0
        loss_history.append(avg_loss)

        if episode % 500 == 0:
            ckpt_path = f"models/dqn_checkpoint_ep{episode}.pth"
            torch.save(agent.policy_net.state_dict(), ckpt_path)
            print(f"[CHECKPOINT] Saved: {ckpt_path}")
            
        agent.update_epsilon()
        agent.update_lr()
        
        # In log ra màn hình mỗi 10 tập
        if episode % 10 == 0:
            avg_loss = episode_loss / episode_steps if episode_steps > 0 else 0.0
            print(f"Episode: {episode:3d}/{NUM_EPISODES} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Avg Loss: {avg_loss:10.6f} | "
                  f"Buffer: {len(replay_buffer)}")

    # 4. Lưu lại "bộ não"
    os.makedirs('models', exist_ok=True) 
    model_path = "models/dqn_model.pth"
    torch.save(agent.policy_net.state_dict(), model_path)
    print("="*50)
    print(f"HUẤN LUYỆN XONG! Đã lưu model tại: {model_path}")
    np.save('models/reward_history.npy', reward_history)
    np.save('models/loss_history.npy', loss_history)
    print("Saved training history.")

if __name__ == "__main__":
    main()