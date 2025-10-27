# Import NN stuff
from collections import deque
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import random 
import copy

# Handle both direct execution and package import
try:
    from .model_rl_deep_agent.model import Linear_QNet, QTrainer
except ImportError:
    from model_rl_deep_agent.model import Linear_QNet, QTrainer

# Data 
import numpy as np
from typing import List, Tuple

# Import using full module path that works with sys.path setup
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Handle both direct execution and package import for agent and model imports
try:
    from agents.agent_class import Agent
    from models import Player, Squad, Slots
except ImportError:
    # When running directly, adjust the path
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from src.agents.agent_class import Agent
    from src.models import Player, Squad, Slots


# Get logger
logger = logging.getLogger(__name__)


class PolicyValueAgent(Agent):
    def __init__(self, agent_id, mode: str = "inference", weights: str = None):
        super().__init__(agent_id)
        self.mode = mode
        self.agent_id = agent_id
        self.weights = weights

        # Concerning the model
        self.epsilon = 0  
        self.gamma = 0.9 
        
        # Calculate input size:
        # 3 base features (auction_progress, player_eval, price)
        # + 12 highest bidder one-hot encoding 
        # + 3 self features (credits, bestxi_eval, squad_eval)
        # + 11 other agents * 3 features each = 33
        # Total: 3 + 12 + 3 + 33 = 51
        
        self.model = Linear_QNet(input_size = 51, hidden_size_1 = 256, hidden_size_2 = 128, output_size = 1)
        self.model.load_weights(custom_weights=self.weights)
        self.states_store = []
        self.policies_store = []
        self.values_store = []
        self.actions_store = []
        self.last_player = None
        self.num_decisions = 0

    def initialize(self, players: List[Player], slots: Slots, initial_credits: int, num_participants: int):
        """Initialize the agent for auction"""
        super().initialize(players, slots, initial_credits, num_participants)
        self.players = players
        self.slots = slots  
        self.current_credits = initial_credits
        self.initial_credits = initial_credits
        self.num_participants = num_participants
        
        # calculate the sum of the s.evaluation of the best [slot * num_participants] players
        best_players_count = self.slots.get_numbers("total") * self.num_participants
        best_players_overall = sorted(players, key=lambda p: p.evaluation, reverse=True)[:best_players_count]
        self.sum_best_overall = sum(p.standardized_evaluation for p in best_players_overall)

    # TODO : understand if it works: Print it at the end of every player in the single_auction setup
    def get_auction_progress(self, all_agents: List[Agent]) -> float:

        squad_evaluations = 0
        for agent in all_agents:
            squad_evaluations += agent.squad.objective(bestxi=False, standardized=True)

        auction_progress = squad_evaluations / self.sum_best_overall 

        return auction_progress

    def get_features(self, current_player: Player, current_price: int, highest_bidder: str, other_agents: List[Agent]) -> List[float]:
        
        #TODO: it will never happen that I have to decide to do the offer when I am the highest bidder
        # Create one-hot encoding for highest bidder (12 positions: self + 11 others)
        highest_bidder_encoding = [0.0] * 12
        if highest_bidder == self.agent_id:
            highest_bidder_encoding[0] = 1.0  # Self is position 0
        elif highest_bidder:
            # Find the position of the highest bidder in other_agents
            for i, agent in enumerate(other_agents):
                if agent.agent_id == highest_bidder:
                    highest_bidder_encoding[i + 1] = 1.0  # Other agents start at position 1
                    break
        
        # Build base features (current auction state)
        base_features = [
            self.auction_progress,
            current_player.standardized_evaluation,
            current_price / self.initial_credits,  
        ]
        
        # Add highest bidder one-hot encoding
        base_features.extend(highest_bidder_encoding)
        
        # Add self features
        base_features.extend([
            self.current_credits / self.initial_credits,
            self.squad.objective(bestxi=True, standardized=True),
            self.squad.objective(bestxi=False, standardized=True),
        ])
        
        # Build other agents features with padding
        other_agent_features = []
        for i in range(11):  # 11 other agents max
            if i < len(other_agents):
                agent = other_agents[i]
                other_agent_features.extend([
                    agent.current_credits / agent.initial_credits,
                    agent.squad.objective(bestxi=True, standardized=True),
                    agent.squad.objective(bestxi=False, standardized=True),
                ])
            else:
                # Padding with zeros for non-existent agents
                other_agent_features.extend([0.0, 0.0, 0.0])
        
        # Combine all features and convert to tensor
        all_features = base_features + other_agent_features
        state = torch.tensor(all_features, dtype=torch.float32)

        return state

    def make_offer_decision(self, current_player: Player, current_price: int, highest_bidder: str, player_list: List[Player], other_agents: List[Agent]) -> str:
        
        # Get the auction progress (current player not included)
        self.auction_progress = self.get_auction_progress(other_agents + [self])

        state = self.get_features(current_player, current_price, highest_bidder, other_agents)

        # Get model output and extract the probability value
        with torch.no_grad():  # Don't track gradients during inference
            probs, value = self.model.forward(state)
            m = torch.distributions.Categorical(probs)        
        action = "offer_+1" if m.sample() > 0.5 else "no_offer"

        # Store decision data if in training mode
        if self.mode == "training":

            #TODO: if its the same player as last decision, overwrite the last decision of the store with the new decision
            if current_player == self.last_player:
                #TODO: they all are lists of tensors but i dont know if its right
                self.states_store[-1] = state
                self.policies_store[-1] = probs[0]
                self.values_store[-1] = value
                self.actions_store[-1] = m

            else:
                #TODO: they all are lists of tensors but i dont know if its right
                self.states_store.append(state)
                self.policies_store.append(probs[0]) 
                self.values_store.append(value) 
                self.actions_store.append(m)  
                self.num_decisions += 1
                #TODO: the reward has to be finalized at the end of the auction in a separate environment (train.py)

            self.last_player = current_player
        return action
