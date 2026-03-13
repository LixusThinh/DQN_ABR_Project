import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        # deque (Double-ended queue) là cấu trúc dữ liệu cực ngon của Python.
        # Khi add data vào mà vượt quá maxlen (ví dụ 10000), nó sẽ tự động
        # đá thằng cũ nhất ở đầu ra. Đỡ mất công viết logic mảng vòng (circular buffer).
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        # Lưu 1 bộ kinh nghiệm (transition) vào bộ nhớ.
        # Lưu ý: Do bên video_env.py mày đã dùng self.state.copy() khi return rồi,
        # nên ở đây tao nhét thẳng vào luôn, không sợ dính lỗi tham chiếu (reference bug).
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        # Bốc ngẫu nhiên một vốc (batch) kinh nghiệm để học
        transitions = random.sample(self.buffer, batch_size)
        
        # zip(*transitions) là một trick lộn ngược mảng của Python.
        # Nó tách một list các tuple (s, a, r, s', d) thành 5 list riêng biệt:
        # 1 list toàn state, 1 list toàn action...
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        # Chuyển đổi tất cả sang Numpy array, ép kiểu dữ liệu chặt chẽ luôn
        # để qua bước sau ném vào mạng PyTorch không bị văng lỗi type.
        return (
            np.array(states, dtype=np.float32),      # Mạng nơ-ron chuộng float32
            np.array(actions, dtype=np.int64),       # Index action là số nguyên
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.uint8)          # Done là True/False, xài uint8 cho nhẹ RAM
        )

    def __len__(self):
        # Trả về số lượng kinh nghiệm hiện có trong thùng
        return len(self.buffer)

# ==================== TEST SCRIPT ====================
if __name__ == "__main__":
    rb = ReplayBuffer(capacity=100)
    
    # Tạo 2 state giả (shape 6x8)
    s1 = np.zeros((6, 8))
    s2 = np.ones((6, 8))
    
    # Nhét thử vào buffer
    rb.push(s1, 1, 0.5, s2, False)
    rb.push(s2, 5, -1.2, s1, True)
    
    print(f"Dung lượng buffer hiện tại: {len(rb)}")
    
    # Sample thử 2 cái
    if len(rb) >= 2:
        s_b, a_b, r_b, ns_b, d_b = rb.sample(2)
        print("\n--- Test Sample ---")
        print(f"Shape của states batch: {s_b.shape}")  # Phải ra (2, 6, 8)
        print(f"Actions batch: {a_b}")
        print(f"Rewards batch: {r_b}")
        print(f"Dones batch: {d_b}")