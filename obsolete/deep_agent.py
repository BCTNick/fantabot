"""
DeepAgent - Neural Network Based Bidding Agent with Reinforcement Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

# Import using full module path that works with sys.path setup
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.agent_class import Agent
from models import Player, Squad, Slots
from obsolete.reward_function import compute_bidding_reward, compute_final_auction_reward

def loss_function(decision: str, outcome: dict, player: Player, final_price: int, agent_state: dict) -> float:
    """
    Calculate loss based on bidding decision and its outcome
    
    Args:
        decision: "offer_+1" or "no_offer"
        outcome: {"won": bool, "final_price": int, "winner": str}
        player: The player that was being bid on
        final_price: Final price the player sold for
        agent_state: {"credits_before": int, "credits_after": int, "squad_before": list, "squad_after": list}
    
    Returns:
        loss: Float representing the loss for this decision
    """
    
    if decision == "offer_+1":
        # Agent decided to bid
        if outcome["won"]:
            # Won the player - evaluate if it was a good purchase
            value_gained = player.evaluation
            price_paid = final_price
            
            # Value-based loss: negative if good value, positive if overpaid
            value_ratio = price_paid / max(value_gained, 1)  # Avoid division by zero
            
            if value_ratio <= 1.0:
                # Good deal: loss decreases with better value
                loss = -np.log(2.0 - value_ratio)  # Negative loss (reward)
            else:
                # Overpaid: loss increases exponentially with overpayment
                loss = (value_ratio - 1.0) ** 2
                
            # Penalty for spending too much relative to remaining budget
            credits_spent_ratio = price_paid / max(agent_state["credits_before"], 1)
            if credits_spent_ratio > 0.8:  # Spending >80% of remaining credits
                loss += credits_spent_ratio * 2.0
                
        else:
            # Didn't win - small penalty for unsuccessful bid
            loss = 0.1
            
        return loss
    
    if decision == "no_offer":
        # Agent decided not to bid
        if outcome["won"] and outcome["winner"] != "UNSOLD":
            # Someone else won the player - evaluate if we missed a good opportunity
            final_price = outcome["final_price"]
            player_value = player.evaluation
            
            # Check if it was a good deal we missed
            value_ratio = final_price / max(player_value, 1)
            
            if value_ratio <= 0.8:  # Player sold for <80% of evaluation (good deal)
                # We missed a bargain - penalty proportional to value missed
                lost_value = player_value - final_price
                affordable = final_price <= agent_state["credits_before"]
                
                if affordable:
                    # We could afford it but didn't bid - higher penalty
                    loss = lost_value / 100.0  # Normalize by typical evaluation scale
                else:
                    # We couldn't afford it anyway - small penalty
                    loss = 0.05
            else:
                # Player sold for fair/high price - good decision not to bid
                loss = -0.05  # Small reward for avoiding overpayment
        else:
            # Player went unsold - neutral outcome
            loss = 0.0
            
        return loss


class BiddingNetwork(nn.Module):
    """Neural network for auction bidding decisions with comprehensive feature input"""
    
    def __init__(self, input_size=39, hidden_size=128):  # Updated to 39 features, larger hidden layer
        super(BiddingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, 16)
        self.fc5 = nn.Linear(16, 1)  # Output: probability of bidding
        self.dropout = nn.Dropout(0.3)  # Increased dropout for regularization
        
    def forward(self, x):
        # Handle both single samples and batches
        single_sample = False
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension for single samples
            single_sample = True
            
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))  # Output probability between 0 and 1
        
        # Return scalar for single samples
        if single_sample:
            return x.squeeze(0)
        return x


class DeepAgent(Agent):
    """Neural network-based auction agent with learning capability"""
    
    def __init__(self, agent_id: str, learning_rate: float = 0.001):
        super().__init__(agent_id)
        self.model = BiddingNetwork()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.experiences = []  # Store experiences for training
        self.all_agents = []  # Store reference to all agents for reward calculation
        
    def initialize(self, players: List[Player], slots: Slots, initial_credits: int, num_participants: int):
        """Initialize the agent for auction"""
        super().initialize(players, slots, initial_credits, num_participants)
        self.slots = slots  # Store slots for feature extraction
        
    def set_auction_context(self, all_agents: List[Agent]):
        """Set reference to all agents for reward calculation"""
        self.all_agents = all_agents
        
    def extract_features(self, player: Player, current_price: int, auction_state: dict) -> torch.Tensor:
        """Extract comprehensive features for neural network input including auction context"""
        
        # === PLAYER FEATURES (6 features) ===
        player_eval = player.evaluation / 100.0  # Normalize
        player_role_encoding = {
            "GK": [1, 0, 0, 0],
            "DEF": [0, 1, 0, 0], 
            "MID": [0, 0, 1, 0],
            "ATT": [0, 0, 0, 1]
        }.get(player.role, [0, 0, 0, 0])
        player_rank_in_role = self._get_player_rank_in_role(player, auction_state.get('player_list', []))
        
        # === AUCTION STATE FEATURES (3 features) ===
        price_normalized = current_price / 200.0  # Normalize typical price range
        credits_remaining = self.current_credits / 1000.0  # Normalize initial credits
        budget_ratio = current_price / max(self.current_credits, 1)
        
        # === MY SQUAD STATE FEATURES (8 features) ===
        current_players_by_role = {
            "GK": len(self.squad.gk),
            "DEF": len(self.squad.def_),
            "MID": len(self.squad.mid),
            "ATT": len(self.squad.att)
        }
        
        # Slots needed by role
        if hasattr(self, 'slots') and self.slots:
            slots_by_role = {
                "GK": self.slots.gk,
                "DEF": self.slots.def_,
                "MID": self.slots.mid,
                "ATT": self.slots.att
            }
            role_needed = 1.0 if current_players_by_role[player.role] < slots_by_role[player.role] else 0.0
            squad_completeness = len(self._squad) / (self.slots.gk + self.slots.def_ + self.slots.mid + self.slots.att)
        else:
            role_needed = 1.0
            squad_completeness = len(self._squad) / 25.0
        
        # My squad quality metrics
        my_squad_value = self.squad.objective(standardized=True) / 100.0  # Normalize
        my_bestxi_value = self.squad.objective(bestxi=True, standardized=True) / 100.0  # Normalize
        
        # Squad role distribution (4 features: GK, DEF, MID, ATT counts normalized)
        my_squad_roles = [
            current_players_by_role["GK"] / 3.0,  # Normalize by typical max
            current_players_by_role["DEF"] / 8.0,
            current_players_by_role["MID"] / 8.0,
            current_players_by_role["ATT"] / 6.0
        ]
        
        # === REMAINING PLAYERS FEATURES (8 features) ===
        remaining_players = auction_state.get('player_list', [])
        remaining_features = self._extract_remaining_players_features(remaining_players)
        
        # === COMPETITORS STATE FEATURES (12 features) ===
        competitors_features = self._extract_competitors_features()
        
        # === AUCTION PROGRESS FEATURES (2 features) ===
        auction_progress = self._calculate_auction_progress(remaining_players)
        time_pressure = min(1.0, (1000 - self.current_credits) / 800.0)  # Spending pressure
        
        # Combine all features
        features = [
            # Player features (6)
            player_eval,
            *player_role_encoding,  # 4 features
            player_rank_in_role,  # 1 feature
            
            # Auction state (3)
            price_normalized,
            credits_remaining,
            budget_ratio,
            
            # My squad state (8)
            role_needed,
            squad_completeness,
            my_squad_value,
            my_bestxi_value,
            *my_squad_roles,  # 4 features
            
            # Remaining players (8)
            *remaining_features,  # 8 features
            
            # Competitors (12)
            *competitors_features,  # 12 features
            
            # Auction progress (2)
            auction_progress,
            time_pressure
        ]  # Total: 39 features
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _get_player_rank_in_role(self, player: Player, all_players: List[Player]) -> float:
        """Get the rank of the player within their role (0-1, where 1 is best)"""
        same_role_players = [p for p in all_players if p.role == player.role]
        if not same_role_players:
            return 0.5  # Default if no comparison possible
        
        # Sort by evaluation descending
        same_role_players.sort(key=lambda p: p.evaluation, reverse=True)
        try:
            rank = same_role_players.index(player)
            return 1.0 - (rank / len(same_role_players))  # Convert to 0-1 where 1 is best
        except ValueError:
            return 0.5  # Player not found
    
    def _extract_remaining_players_features(self, remaining_players: List[Player]) -> List[float]:
        """Extract features about remaining players in auction (8 features)"""
        if not remaining_players:
            return [0.0] * 8
        
        # Count remaining players by role
        remaining_by_role = {"GK": 0, "DEF": 0, "MID": 0, "ATT": 0}
        total_remaining_value = 0.0
        
        for player in remaining_players:
            remaining_by_role[player.role] += 1
            total_remaining_value += player.evaluation
        
        # Calculate features
        total_remaining = len(remaining_players)
        avg_remaining_value = total_remaining_value / max(total_remaining, 1) / 100.0  # Normalize
        
        # Best remaining player by role
        best_remaining = {"GK": 0.0, "DEF": 0.0, "MID": 0.0, "ATT": 0.0}
        for player in remaining_players:
            best_remaining[player.role] = max(best_remaining[player.role], player.evaluation / 100.0)
        
        return [
            total_remaining / 300.0,  # Normalize by typical auction size
            avg_remaining_value,
            remaining_by_role["GK"] / 10.0,  # Normalize counts
            remaining_by_role["DEF"] / 50.0,
            remaining_by_role["MID"] / 50.0,
            remaining_by_role["ATT"] / 30.0,
            best_remaining["GK"],
            max(best_remaining["DEF"], best_remaining["MID"], best_remaining["ATT"])  # Best outfield
        ]
    
    def _extract_competitors_features(self) -> List[float]:
        """Extract features about competitor squads (12 features)"""
        if not self.all_agents:
            return [0.0] * 12
        
        competitors = [agent for agent in self.all_agents if agent.agent_id != self.agent_id]
        if not competitors:
            return [0.0] * 12
        
        # Competitor metrics
        competitor_values = []
        competitor_bestxi_values = []
        competitor_credits = []
        competitor_squad_sizes = []
        
        for comp in competitors:
            try:
                competitor_values.append(comp.squad.objective(standardized=True))
                competitor_bestxi_values.append(comp.squad.objective(bestxi=True, standardized=True))
                competitor_credits.append(comp.current_credits)
                competitor_squad_sizes.append(len(comp._squad))
            except:
                # Handle any attribute errors gracefully
                competitor_values.append(0.0)
                competitor_bestxi_values.append(0.0)
                competitor_credits.append(1000.0)
                competitor_squad_sizes.append(0)
        
        # Calculate comparative metrics
        avg_competitor_value = sum(competitor_values) / max(len(competitor_values), 1) / 100.0
        best_competitor_value = max(competitor_values) / 100.0 if competitor_values else 0.0
        avg_competitor_bestxi = sum(competitor_bestxi_values) / max(len(competitor_bestxi_values), 1) / 100.0
        best_competitor_bestxi = max(competitor_bestxi_values) / 100.0 if competitor_bestxi_values else 0.0
        avg_competitor_credits = sum(competitor_credits) / max(len(competitor_credits), 1) / 1000.0
        min_competitor_credits = min(competitor_credits) / 1000.0 if competitor_credits else 1.0
        avg_competitor_squad_size = sum(competitor_squad_sizes) / max(len(competitor_squad_sizes), 1) / 25.0
        
        # Relative position features
        my_value = self.squad.objective(standardized=True) / 100.0
        my_bestxi = self.squad.objective(bestxi=True, standardized=True) / 100.0
        my_credits = self.current_credits / 1000.0
        my_squad_size = len(self._squad) / 25.0
        
        value_rank = sum(1 for val in competitor_values if my_value * 100 > val) / max(len(competitor_values), 1)
        credits_rank = sum(1 for cred in competitor_credits if my_credits * 1000 > cred) / max(len(competitor_credits), 1)
        
        return [
            avg_competitor_value,
            best_competitor_value,
            avg_competitor_bestxi,
            best_competitor_bestxi,
            avg_competitor_credits,
            min_competitor_credits,
            avg_competitor_squad_size,
            value_rank,  # My ranking in squad value (0-1)
            credits_rank,  # My ranking in remaining credits (0-1)
            my_value - avg_competitor_value,  # Value gap
            my_bestxi - avg_competitor_bestxi,  # Best XI gap
            my_credits - avg_competitor_credits  # Credits gap
        ]
    
    def _calculate_auction_progress(self, remaining_players: List[Player]) -> float:
        """Calculate how far through the auction we are (0-1)"""
        if not hasattr(self, '_initial_player_count'):
            # Estimate initial players based on remaining + current squad sizes
            total_current_squads = sum(len(agent._squad) for agent in self.all_agents) if self.all_agents else len(self._squad)
            self._initial_player_count = len(remaining_players) + total_current_squads
        
        if self._initial_player_count == 0:
            return 1.0
        
        return 1.0 - (len(remaining_players) / self._initial_player_count)
        
    def make_offer_decision(self, current_player: Player, current_price: int, highest_bidder: str, player_list: List[Player], agents) -> str:
        """Make bidding decision using neural network and compute immediate reward"""
        # Create auction state dictionary
        auction_state = {
            'highest_bidder': highest_bidder,
            'player_list': player_list
        }
        
        features = self.extract_features(current_player, current_price, auction_state)
        
        # Get decision from network
        decision_prob = self.get_decision_probability(features)
        decision = "offer_+1" if decision_prob > 0.5 else "no_offer"
        
        # Store experience for later training with reward calculation context
        self.store_experience_with_reward(
            features=features, 
            decision=decision, 
            decision_prob=decision_prob,
            player=current_player,
            current_price=current_price,
            highest_bidder=highest_bidder,
            remaining_players=player_list
        )
        
        return decision
    
    def store_experience_simple(self, features: torch.Tensor, action: str, decision_prob: float):
        """Store experience for training (simplified version)"""
        self.experiences.append({
            'features': features.tolist(),
            'action': 1 if action == "offer_+1" else 0,
            'decision_prob': decision_prob
        })
    
    def store_experience_with_reward(self, features: torch.Tensor, decision: str, decision_prob: float,
                                   player: Player, current_price: int, highest_bidder: str, 
                                   remaining_players: List[Player]):
        """Store experience with immediate reward calculation using new reward function"""
        
        # Calculate immediate reward using the new reward function
        if self.all_agents:
            reward = compute_bidding_reward(
                agent=self,
                agents=self.all_agents,
                remaining_players=remaining_players,
                decision=decision,
                player=player,
                current_price=current_price,
                highest_bidder=highest_bidder
            )
        else:
            # Fallback to simple reward if no auction context
            reward = 0.0
        
        self.experiences.append({
            'features': features.tolist(),
            'action': 1 if decision == "offer_+1" else 0,
            'decision_prob': decision_prob,
            'reward': reward,
            'target': decision_prob + reward * 0.1  # Adjust target based on reward
        })
    
    def loss_function(self, offer_prob: float, actual_decision: int, 
                     final_evaluation: float, credits_spent: int) -> float:
        """
        Calculate loss for a bidding decision
        
        Args:
            offer_prob: Probability of offering (0-1)
            actual_decision: 1 if offered, 0 if not
            final_evaluation: Final squad evaluation
            credits_spent: Total credits spent in auction
            
        Returns:
            Loss value
        """
        
        # Base prediction loss
        prediction_loss = (offer_prob - actual_decision) ** 2
        
        # Performance-based adjustment
        if final_evaluation > 50:  # Good performance
            performance_multiplier = 0.8  # Reduce loss
        elif final_evaluation > 30:  # Average performance
            performance_multiplier = 1.0  # Normal loss
        else:  # Poor performance
            performance_multiplier = 1.2  # Increase loss
        
        # Budget efficiency adjustment
        if credits_spent < 800:  # Under-spending
            budget_multiplier = 1.1
        elif credits_spent > 950:  # Over-spending
            budget_multiplier = 1.1
        else:  # Good budget usage
            budget_multiplier = 0.9
        
        total_loss = prediction_loss * performance_multiplier * budget_multiplier
        
        return total_loss
        
    def compute_final_reward(self) -> float:
        """Compute final reward at the end of the auction"""
        if self.all_agents:
            return compute_final_auction_reward(self, self.all_agents)
        return 0.0
        
    def get_decision_probability(self, features: torch.Tensor) -> float:
        """Get probability of bidding from the network"""
        self.model.eval()  # Set to evaluation mode to disable dropout
        with torch.no_grad():
            return self.model(features).item()
    
    def make_decision(self, features: torch.Tensor, threshold: float = 0.5) -> str:
        """Make bidding decision based on network output"""
        prob = self.get_decision_probability(features)
        return "offer_+1" if prob > threshold else "no_offer"
    
    def store_experience(self, features: torch.Tensor, decision: str, outcome: dict, 
                        player: Player, final_price: int, agent_state: dict):
        """Store an experience for later training"""
        loss = loss_function(decision, outcome, player, final_price, agent_state)
        self.experiences.append({
            "features": features,
            "decision": decision, 
            "loss": loss,
            "target": 1.0 if decision == "offer_+1" else 0.0
        })
    
    def train_on_experiences(self, batch_size: int = 32):
        """Train the network on stored experiences using rewards"""
        if len(self.experiences) < batch_size:
            return
            
        # Sample random batch
        batch_indices = np.random.choice(len(self.experiences), batch_size, replace=False)
        batch = [self.experiences[i] for i in batch_indices]
        
        # Prepare batch tensors - handle both old and new experience formats
        features_batch = []
        targets_batch = []
        rewards_batch = []
        
        for exp in batch:
            features_batch.append(torch.tensor(exp["features"], dtype=torch.float32))
            
            # Handle both old and new experience formats
            if "reward" in exp:
                # New format with rewards
                targets_batch.append(exp["target"])
                rewards_batch.append(exp["reward"])
            else:
                # Old format fallback
                targets_batch.append(exp.get("target", exp["action"]))
                rewards_batch.append(0.0)
        
        features_batch = torch.stack(features_batch)
        targets_batch = torch.tensor(targets_batch, dtype=torch.float32)
        rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32)
        
        # Forward pass
        predictions = self.model(features_batch).squeeze()
        
        # Calculate loss - use reward-weighted MSE
        mse_loss = F.mse_loss(predictions, targets_batch, reduction='none')
        
        # Weight loss by reward magnitude (higher rewards = more important to learn from)
        reward_weights = 1.0 + torch.abs(rewards_batch) * 0.5
        weighted_loss = (mse_loss * reward_weights).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()
        
        # Clear old experiences periodically
        if len(self.experiences) > 1000:
            self.experiences = self.experiences[-500:]  # Keep recent experiences