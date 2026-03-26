import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# ==========================================
# 1. KIẾN TRÚC MẠNG NEURAL (BỘ NÃO)
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, state_dim=48, action_dim=6):
        super(QNetwork, self).__init__()
        # Mạng truyền thẳng (MLP) đơn giản: 48 -> 128 -> 64 -> 6
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        # x input vào sẽ có shape: (batch_size, 6, 8)
        # BƯỚC QUAN TRỌNG: Đập bẹp ma trận 6x8 thành vector 48
        x = x.view(x.size(0), -1) 
        return self.fc(x)

# ==========================================
# 2. CLASS DQN AGENT (THẰNG ĐIỀU KHIỂN)
# ==========================================
class DQNAgent:
    def __init__(self, state_dim=48, action_dim=6, lr=0.001, gamma=0.99):
        self.action_dim = action_dim
        self.gamma = gamma  
        
        # Thiết lập Epsilon cho Epsilon-Greedy
        self.epsilon = 1.0       
        self.epsilon_min = 0.01  
        self.epsilon_decay = 0.998 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQN] Khởi tạo Agent chạy trên thiết bị: {self.device}")

        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() 
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=200, gamma=0.5
        )
        self.criterion = nn.MSELoss()

    def select_action(self, state, is_training=True):
        if is_training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            
        return torch.argmax(q_values).item()

    def train_step(self, replay_buffer, batch_size=32):
        if len(replay_buffer) < batch_size:
            return None
            
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)     

        q_values = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            max_next_q_values = self.target_net(next_states).gather(1, best_actions)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        loss = self.criterion(q_values, target_q_values)
        
        self.optimizer.zero_grad() 
        loss.backward()            
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()      
        
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_lr(self):
        self.scheduler.step()

# ==================== TEST SCRIPT ====================
if __name__ == "__main__":
    agent = DQNAgent()
    
    # Tạo 1 state ngẫu nhiên 6x8
    dummy_state = np.random.rand(6, 8)
    
    # Test hàm select_action
    action = agent.select_action(dummy_state, is_training=False)
    print(f"Action được chọn bởi mạng Neural chưa train: {action}")
    