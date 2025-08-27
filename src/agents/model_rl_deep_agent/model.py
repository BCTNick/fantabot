import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size = 51, hidden_size_1 = 256, hidden_size_2 = 128, output_size = 1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size_1)
        self.dropout1 = nn.Dropout(0.2)  # 20% dropout after first layer
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.dropout2 = nn.Dropout(0.1)  # 10% dropout after second layer
        self.linear3 = nn.Linear(hidden_size_2, output_size)
        
        # Try to load existing weights
        self.load_best_weights()

    def load_best_weights(self):
        """Load weights from weights/best_weights.pth if it exists, otherwise use random initialization"""
        weights_file = './weights/best_weights.pth'
        if os.path.exists(weights_file):
            self.load_state_dict(torch.load(weights_file))

    def forward(self, x):
        input_tensor = torch.tensor(x, dtype=torch.float32)

        x = F.relu(self.linear1(input_tensor))
        x = self.dropout1(x)  
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)  
        x = F.sigmoid(self.linear3(x))  
        return x

    def save(self, file_name='best_weights.pth'):
        weights_folder_path = './weights'
        if not os.path.exists(weights_folder_path):
            os.makedirs(weights_folder_path)
            print(f"Created weights folder: {weights_folder_path}")

        file_path = os.path.join(weights_folder_path, file_name)
        torch.save(self.state_dict(), file_path)
        print(f"Model weights saved to {file_path}")


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, all_features0, output_probability, reward):
        all_features0 = torch.tensor(all_features0, dtype=torch.float)
        output_probability = torch.tensor(output_probability, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(all_features0.shape) == 1:
            # (1, x)
            all_features0 = torch.unsqueeze(all_features0, 0)
            output_probability = torch.unsqueeze(output_probability, 0)
            reward = torch.unsqueeze(reward, 0)

        # 1: predicted Q values with current state
        pred = self.model(all_features0)

        # 2: Create target - use actual reward as target Q-value
        target = pred.clone()
        for idx in range(len(all_features0)):
            # Simple approach: Q(state, action) = actual_reward
            target[idx][torch.argmax(output_probability[idx]).item()] = reward[idx]
    
        # 3: Train to match actual rewards
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()