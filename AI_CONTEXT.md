# Project: DQN_ABR_Project (Deep Q-Network for Adaptive Bitrate)

## 🎯 Goal
Tối ưu hóa việc lựa chọn bitrate video bằng thuật toán DQN để đạt chỉ số QoE cao nhất trong các điều kiện mạng thay đổi.

## 🛠 Tech Stack
- **Framework:** PyTorch/TensorFlow (Bạn nên ghi rõ phiên bản).
- **Algorithm:** DQN (Deep Q-Network).
- **Environment:** Custom Gymnasium-based environment (`video_env.py`).

## 📁 Project Structure & Logic
- `train.py`: Entry point để khởi chạy quá trình huấn luyện và cấu hình hyperparameter.
- `dqn_agent.py`: Chứa kiến trúc Neural Network (Input: network throughput, buffer level; Output: bitrate action).
- `video_env.py`: Giả lập môi trường video streaming, tính toán phần thưởng (Reward).
- `replay_buffer.py`: Lưu trữ transition (s, a, r, s') để thực hiện Experience Replay.

## 📊 Core Metrics & Rewards
- **QoE Formula:** $QoE = bitrate - 4.3 \times rebuffer - |bitrate_{n} - bitrate_{n-1}|$
- **Current Status:** Đã huấn luyện 1000 episodes. Đã có checkpoint lưu tại `/models`.

## 🚀 Next Tasks (Cần hỗ trợ)
1. **Evaluation:** Viết script script `eval.py` để chạy model đã train trên tập dữ liệu mạng (trace) mới.
2. **Baselines:** Triển khai thuật toán BOLA (Lyapunov-based) và Rate-based (Heuristic) để làm mốc so sánh.
3. **Visualization:** Vẽ biểu đồ so sánh QoE giữa DQN, BOLA và Rate-based.