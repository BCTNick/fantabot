import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class PolicyValueNet(nn.Module):
    def __init__(self, input_size = 51, hidden_size_1 = 256, hidden_size_2 = 128):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size_1)
        self.dropout1 = nn.Dropout(0.2)  # 20% dropout after first layer
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.dropout2 = nn.Dropout(0.1)  # 10% dropout after second layer
        self.policy = nn.Linear(hidden_size_2, 2)  # Two possible actions
        self.value = nn.Linear(hidden_size_2, 1)   # Value estimate

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


    def forward(self, input_tensor):

        x = F.relu(self.linear1(input_tensor))
        x = self.dropout1(x)  
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)  
        policy = F.softmax(self.policy(x), dim=-1)
        #TODO: Check if sigmoid is appropriate for value output and if it should be linked with the final reward 
        value = F.sigmoid(self.value(x))
        return policy, value

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
