import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
from datetime import datetime
import os

# Set up logging
if not os.path.exists('logs'):
    os.makedirs('logs')

log_filename = f"logs/policy_value_net_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# --- Environment ---
# Sequence length = 5, actions = {0, 1}
# Reward = +1 if sum(actions) == 3 else 0

EPISODE_LENGTH = 5

def environment(actions):
    reward = 1.0 if sum(actions) == 3 else 0.0
    return reward

# --- Neural Network: policy and value combined ---
class PolicyValueNet(nn.Module):
    def __init__(self, state_dim=EPISODE_LENGTH, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc_policy = nn.Linear(hidden, 2)  # Two possible actions
        self.fc_value = nn.Linear(hidden, 1)   # Value estimate

    def forward(self, state):
        x = F.relu(self.fc1(state))
        policy_logits = self.fc_policy(x)
        value = self.fc_value(x)
        return F.softmax(policy_logits, dim=-1), value

# --- Training setup ---
net = PolicyValueNet()
optimizer = optim.Adam(net.parameters(), lr=0.01)
gamma = 1.0  # no discounting

# --- Training loop ---
for episode in range(1000):
    state = torch.zeros(EPISODE_LENGTH)  # initially empty sequence
    log_probs = []
    actions = []
    values = []

    # Generate one episode
    for t in range(EPISODE_LENGTH):
        probs, value = net(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        
        log_probs.append(m.log_prob(action))
        values.append(value)
        actions.append(action.item())
        
        # Create new state tensor instead of modifying in-place
        new_state = state.clone()
        new_state[t] = action.item()
        state = new_state

    # Get final reward
    reward = environment(actions)
    R = torch.tensor([reward], dtype=torch.float32)

    # --- Compute losses ---
    policy_loss = 0
    value_loss = 0

    for log_prob, value in zip(log_probs, values):
        advantage = R - value.detach()
        policy_loss += -log_prob * advantage
        value_loss += F.mse_loss(value, R)

    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # --- Logging ---
    if episode % 100 == 0:
        logger.info(f"\nEpisode {episode}")
        logger.info("-" * 40)
        logger.info(f"Actions taken: {actions}")
        logger.info(f"Final reward: {reward:.1f}")
        logger.info(f"Total loss: {loss.item():.4f}")
        logger.info(f"Policy loss: {policy_loss.item():.4f}")
        logger.info(f"Value loss: {value_loss.item():.4f}")
        
        # Log network outputs for each step
        logger.info("\nNetwork outputs for each step:")
        for t in range(EPISODE_LENGTH):
            # Create a padded state tensor
            current_state = torch.zeros(EPISODE_LENGTH)
            current_state[:t+1] = state[:t+1].clone()
            probs, value = net(current_state)
            logger.info(f"Step {t+1}:")
            logger.info(f"  Action probabilities: {probs.detach().numpy()}")
            logger.info(f"  Value estimate: {value.item():.4f}")
            logger.info(f"  Current state: {current_state.tolist()}")
