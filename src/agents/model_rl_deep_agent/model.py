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
        self.load_weights()

    def load_weights(self, custom_weights=None):
        if custom_weights:
            # Custom weights provided - build path from base weights directory
            base_weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
            
            # Always treat custom_weights as a path relative to base_weights_dir
            weights_file = os.path.join(base_weights_dir, custom_weights)
                
            if os.path.exists(weights_file):
                self.load_state_dict(torch.load(weights_file))

        else:
            # Default behavior - load best_weights.pth directly from weights folder
            weights_file = os.path.join(os.path.dirname(__file__), 'weights', 'best_weights.pth')
            if os.path.exists(weights_file):
                self.load_state_dict(torch.load(weights_file))


    def forward(self, x):
        # Efficient tensor conversion - avoid unnecessary copying if already a tensor
        if isinstance(x, torch.Tensor):
            input_tensor = x.detach().clone().to(torch.float32)
        else:
            input_tensor = torch.tensor(x, dtype=torch.float32)

        x = F.relu(self.linear1(input_tensor))
        x = self.dropout1(x)  
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)  
        x = F.sigmoid(self.linear3(x))  
        return x

    def save(self, file_path='best_weights.pth'):
        # If file_path is absolute, use it directly; if relative, use with ./weights/
        if os.path.isabs(file_path):
            # Absolute path - use as is
            save_path = file_path
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        else:
            # Relative path - use with ./weights/
            weights_folder_path = './weights'
            if not os.path.exists(weights_folder_path):
                os.makedirs(weights_folder_path)
                print(f"Created weights folder: {weights_folder_path}")
            save_path = os.path.join(weights_folder_path, file_path)

        torch.save(self.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")


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