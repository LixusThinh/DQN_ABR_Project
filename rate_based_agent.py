import numpy as np
from video_env import VideoEnv
from load_trace import load_trace

BITRATES = [300.0, 750.0, 1200.0, 1850.0, 2850.0, 4300.0] # kbps

class RateBasedAgent:
    def __init__(self, history_size=5, safety_margin=0.8):
        self.history_size = history_size
        self.safety_margin = safety_margin
        
    def select_action(self, state):
        # 1. Lấy lịch sử băng thông
        throughput_history = state[0, :]
        valid_throughputs = throughput_history[throughput_history > 0]
        
        if len(valid_throughputs) == 0:
            return 0 
            
        recent_throughputs = valid_throughputs[-self.history_size:]
        recent_throughputs_kbps = recent_throughputs * 1000.0 
        
        # 2. Tính Harmonic Mean
        harmonic_mean_bw = len(recent_throughputs_kbps) / np.sum(1.0 / recent_throughputs_kbps)
        
        # 3. Áp dụng hệ số an toàn
        predicted_bw = harmonic_mean_bw * self.safety_margin
        
        # 4. Chọn mức bitrate phù hợp
        action = 0 
        for i in range(len(BITRATES) - 1, -1, -1):
            if BITRATES[i] <= predicted_bw:
                action = i
                break
                
        return action

# ==================== TEST SCRIPT ====================
if __name__ == "__main__":
    # Load data mạng
    time_arr, bw_arr = load_trace('traces/test_trace.txt')
    
    # Khởi tạo môi trường và Agent
    env = VideoEnv(time_arr, bw_arr)
    agent = RateBasedAgent(history_size=5, safety_margin=0.8)
    
    state = env.reset()
    done = False
    
    episode_reward = 0.0
    step_count = 0
    
    print("Bắt đầu chạy test RATE-BASED AGENT...")
    print("-" * 70)
    
    while not done:
        # Thay vì random, giờ ta để Agent tự quyết định dựa vào state
        action = agent.select_action(state)
        
        next_state, reward, done, info = env.step(action)
        
        episode_reward += reward
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
    print(f"[DONE] Kết thúc video sau {step_count} chunks")
    print(f"Total Reward    : {episode_reward:.3f}")
    print(f"Average Reward  : {episode_reward / step_count:.3f}")