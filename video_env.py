import numpy as np
import random
from load_trace import load_trace # Import hàm load data mày đã viết ở file trước

# ==================== CÁC HẰNG SỐ ====================
BITRATES = [300.0, 750.0, 1200.0, 1850.0, 2850.0, 4300.0] # kbps
CHUNK_LENGTH = 4.0  # Mỗi chunk video dài 4 giây
REBUF_PENALTY = 2.66 # Hệ số phạt đứng hình (mu) cho QoE_log
SMOOTH_PENALTY = 1.0 # Phạt thay đổi chất lượng
S_INFO = 6  # Số lượng features trong state 
S_LEN = 8   # Lịch sử 8 chunks gần nhất

class VideoEnv:
    def __init__(self, time_arr, bw_arr):
        """
        Khởi tạo môi trường với dữ liệu mạng truyền vào
        """
        self.time_arr = time_arr
        self.bw_arr = bw_arr
        self.trace_idx = 0 
        self.total_chunks = 48 # Giả sử video có 48 chunks
        
    def reset(self):
        """
        Reset lại môi trường khi bắt đầu episode mới
        """
        self.buffer_size = 0.0 
        self.video_chunk_counter = 0 
        self.last_bitrate_idx = 0 
        
        # Khởi tạo state vector toàn số 0
        self.state = np.zeros((S_INFO, S_LEN))
        
        # Chọn ngẫu nhiên 1 thời điểm bắt đầu trong file mạng
        self.trace_idx = np.random.randint(0, len(self.bw_arr) - 1)
        
        return self.state

    def step(self, action):
        """
        Thực thi 1 bước tải chunk video
        """
        bitrate = BITRATES[action]
        
        # 1. Tính kích thước chunk (Megabits)
        chunk_size_megabits = (bitrate * CHUNK_LENGTH) / 1000.0 
        
        # 2. Tính thời gian tải (delay)
        delay = 0.0
        video_chunk_remain = chunk_size_megabits
        step_time = 1.0 # Giả định data mạng cách nhau 1 giây
        
        while video_chunk_remain > 0:
            current_bw = self.bw_arr[self.trace_idx]
            can_download = current_bw * step_time 
            
            if video_chunk_remain > can_download:
                delay += step_time
                video_chunk_remain -= can_download
                self.trace_idx += 1 
                
                # Loop lại nếu hết mảng data mạng
                if self.trace_idx >= len(self.bw_arr):
                    self.trace_idx = 0 
            else:
                fraction_time = video_chunk_remain / current_bw
                delay += fraction_time
                video_chunk_remain = 0 
        
        # 3. Tính rebuffering và cập nhật buffer
        rebuffer_time = max(delay - self.buffer_size, 0.0)
        self.buffer_size = max(self.buffer_size - delay, 0.0) + CHUNK_LENGTH
        
        # 4. Tính Reward (QoE_log)
        q_current = np.log(bitrate / BITRATES[0])
        q_last = np.log(BITRATES[self.last_bitrate_idx] / BITRATES[0])
        reward = q_current - (REBUF_PENALTY * rebuffer_time) - (SMOOTH_PENALTY * abs(q_current - q_last))
        
        # 5. Cập nhật trạng thái
        self.video_chunk_counter += 1
        done = (self.video_chunk_counter >= self.total_chunks)
        self.last_bitrate_idx = action
        
        # Trượt ma trận sang trái
        self.state = np.roll(self.state, -1, axis=1)
        
        # Cập nhật cột mới nhất
        self.state[0, -1] = chunk_size_megabits / delay # Throughput
        self.state[1, -1] = delay # Download time
        
        if not done:
            next_chunk_sizes = [(b * CHUNK_LENGTH) / 1000.0 for b in BITRATES]
            self.state[2, :6] = next_chunk_sizes
        else:
            self.state[2, :6] = np.zeros(6)
            
        self.state[3, -1] = self.buffer_size # Buffer
        self.state[4, -1] = self.total_chunks - self.video_chunk_counter # Chunks remain
        self.state[5, -1] = bitrate / BITRATES[-1] # Normalized last bitrate
        
        # 6. Đóng gói info
        info = {
            'bitrate': bitrate,
            'rebuffer': rebuffer_time,
            'delay': delay,
            'buffer_size': self.buffer_size
        }
        
        return self.state, reward, done, info

# ==================== TEST SCRIPT ====================
if __name__ == "__main__":
    # Đọc data mạng (đảm bảo file test_trace.txt tồn tại trong thư mục traces)
    time_arr, bw_arr = load_trace('traces/test_trace.txt')
    
    # Khởi tạo môi trường
    env = VideoEnv(time_arr, bw_arr)
    state = env.reset()
    done = False
    
    # Tracking
    episode_reward = 0.0
    step_count = 0
    rewards_history = []
    
    print("Bắt đầu chạy test Random Agent...")
    print("-" * 70)
    
    while not done:
        action = random.randint(0, 5) # Random từ 0 đến 5
        
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
    print(f"[DONE] Kết thúc video sau {step_count} chunks")
    print(f"Total Reward    : {episode_reward:.3f}")
    print(f"Average Reward  : {episode_reward / step_count:.3f}")