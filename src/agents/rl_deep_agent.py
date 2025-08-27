# Import NN stuff
from collections import deque
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import random 
import copy
from .model_rl_deep_agent.model import Linear_QNet, QTrainer

# Data 
import numpy as np
from typing import List, Tuple

# Import using full module path that works with sys.path setup
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from src.agents.training_rl_deep_agent import train 
from agents.agent_class import Agent
from models import Player, Squad, Slots


# Get logger
logger = logging.getLogger(__name__)


class RLDeepAgent(Agent):
    def __init__(self, agent_id, mode: str = "inference"):
        super().__init__(agent_id)
        self.mode = mode
        self.agent_id = agent_id

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
        self.model.load_best_weights()
        self.trainer = QTrainer(model=self.model, lr=0.1, gamma=self.gamma)
        self.all_features0_store = []
        self.output_probability_store = []
        self.reward_store = []

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

    def get_auction_progress(self, all_agents: List[Agent]) -> float:

        # TODO : understand if the to, t1 makes sense in this function
        # Get the sum of the players evaluation of every squads' agent 
        squad_evaluations = 0
        for agent in all_agents:
            squad_evaluations += agent.squad.objective(bestxi=False, standardized=True)

        auction_progress = squad_evaluations / self.sum_best_overall 

        return auction_progress

    def get_score(self, agent: Agent, auction_progress) -> float:

        # Get the score of my agent
        weighted_eval = agent.squad.objective(bestxi=True, standardized=True) * 0.9 + agent.squad.objective(bestxi=False, standardized=True) * 0.1
        credit_ratio = agent.current_credits / agent.initial_credits
        remaining_slots = 1 - (agent.squad.get_remaining_slots(agent.slots, "total") / agent.slots.get_numbers("total"))

        # Dynamic weights based on auction stage
        w_e = 0.5 + 0.7 * auction_progress  # team value matters more when good players are finishing
        w_c = 0.5 - 0.5 * auction_progress  # credits matter less when good players are finishing
        w_s = 0.1 - 0.1 * auction_progress  # slots matter less when good players are finishing
        
        # Normalize weights just in case floating point errors occur
        total_w = w_c + w_e + w_s
        score = (weighted_eval * w_e + credit_ratio * w_c + remaining_slots * w_s) / total_w
        return score

    def get_scores(self, action: str, current_player: Player, current_price: int, highest_bidder: str, updated_player_list: List[Player], other_agents: List[Agent], mode: chr = "t0") -> List[float]:
        
        if mode == "t0": # TODO you can do this just when the agent is doing the first bid for the current_player

            # Get my score
            my_score = self.get_score(self, self.auction_progress)

            scores = [my_score]  # Self always at index 0

            # Get the score of other agents 
            for agent in other_agents:
                score = self.get_score(agent, self.auction_progress)
                scores.append(score)

        elif mode == "t1":  

            # Create temporary copies for simulation without affecting real auction
            temp_other_agents = copy.deepcopy(other_agents)
        
            # Create a copy of self 
            temp_self = copy.copy(self)  # Shallow copy first
            temp_self._squad = copy.deepcopy(self._squad)  # Deep copy the squad
            temp_self.current_credits = self.current_credits  # Copy credits
            temp_self.initial_credits = self.initial_credits  # Copy initial credits

            temp_player = copy.deepcopy(current_player)
            temp_updated_players_list = copy.deepcopy(updated_player_list)

            # Simulate the auction outcome on temporary copies
            if action == "offer_+1": 
                highest_bidder = self.agent_id
                current_price = current_price + 1

            # Simulate the player being sold at the highest bidder
            if highest_bidder:
                temp_player.fantasy_team = highest_bidder
                temp_player.final_cost = current_price

                # Update the player in temp_updated_players_list
                for i, player in enumerate(temp_updated_players_list):
                    if player.name == temp_player.name and player.role == temp_player.role:
                        temp_updated_players_list[i].fantasy_team = highest_bidder
                        temp_updated_players_list[i].final_cost = current_price
                        break

                # Update the winner's squad and credits
                winner_agent = next(agent for agent in temp_other_agents + [temp_self] if agent.agent_id == highest_bidder)
                winner_agent._squad.append(temp_player)
                winner_agent.current_credits -= current_price

            else:
                temp_player.fantasy_team = "UNSOLD"
                temp_player.final_cost = 0
                
                # Update the player in temp_updated_players_list
                for i, player in enumerate(temp_updated_players_list):
                    if player.name == temp_player.name and player.role == temp_player.role:
                        temp_updated_players_list[i].fantasy_team = "UNSOLD"
                        temp_updated_players_list[i].final_cost = 0
                        break
            
            # Get the auction progress (current player included for simulation)
            auction_progress = self.get_auction_progress(temp_other_agents + [temp_self])

            # Get scores using temporary copies
            my_score = self.get_score(temp_self, auction_progress)

            scores = [my_score]  

            # Get the score of other agents using temporary copies
            for agent in temp_other_agents:
                score = self.get_score(agent, auction_progress)
                scores.append(score)

        return scores

    def get_features(self, current_player: Player, current_price: int, highest_bidder: str, other_agents: List[Agent]) -> List[float]:
    
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
        
        # Combine all features
        all_features = base_features + other_agent_features

        return all_features

    def get_reward(self, scores_t0, scores_t1, method="zscore") -> float:

        # Pre defined parameters
        tau, beta, eps = 1.0, 1e-3, 1e-8
        s0, s1 = np.array(scores_t0), np.array(scores_t1)

        # --- 1. Robust z-score improvement ---
        if method.lower() == "zscore":
            def z(scores, i):
                med = np.median(scores)
                mad = np.median(np.abs(scores - med))
                return (scores[i] - med) / (1.4826 * mad + eps)
            return z(s1, 0) - z(s0, 0)

        # --- 2. Bradley–Terry likelihood improvement ---
        elif method.lower() == "bradleyterry":
            def L(scores, i):
                diffs = (scores[i] - np.delete(scores, i)) / tau
                return np.mean(-np.log1p(np.exp(-diffs)))  # log-sigmoid
            return L(s1, 0) - L(s0, 0)

        # --- 3. Delta-vs-cohort ---
        elif method.lower() == "delta":
            da = s1[0] - s0[0]
            others = [j for j in range(len(s1)) if j != 0]
            d_others = s1[others] - s0[others]

            med = np.median(d_others)
            mad = np.median(np.abs(d_others - med)) + eps
            da_norm = (da - med) / (1.4826 * mad)
            others_norm = (d_others - med) / (1.4826 * mad)
            denom = np.abs(np.mean(others_norm)) + beta
            return (da_norm - np.mean(others_norm)) / denom
        
            # --- 4. Purely positional (percentile rank improvement) ---
        elif method.lower() == "positional":
            def percentile(scores, i):
                order = scores.argsort()
                ranks = np.empty_like(order)
                ranks[order] = np.arange(len(scores))
                return ranks[i] / max(len(scores) - 1, 1)
            return percentile(s1, 0) - percentile(s0, 0)

    def make_offer_decision(self, current_player: Player, current_price: int, highest_bidder: str, player_list: List[Player], other_agents: List[Agent]) -> str:
        
        # Get the auction progress (current player not included)
        self.auction_progress = self.get_auction_progress(other_agents + [self])

        all_features0 = self.get_features(current_player, current_price, highest_bidder, other_agents)

        # Get model output and extract the probability value
        with torch.no_grad():  # Don't track gradients during inference
            probability_tensor = self.model.forward(all_features0)
            probability = float(probability_tensor.item())  # Extract scalar value
        
        action = "offer_+1" if probability > 0.5 else "no_offer"

        # update the model if on training 
        if self.mode == "training":
            try:
                # Store everything
                self.all_features0_store.append(all_features0)
                self.output_probability_store.append(probability)  # Now storing the scalar value
                # Get and store the reward
                scores_t0 = self.get_scores(action, current_player, current_price, highest_bidder, player_list, other_agents, mode = "t0")
                scores_t1 = self.get_scores(action, current_player, current_price, highest_bidder, player_list, other_agents, mode = "t1")

                reward = self.get_reward(scores_t0, scores_t1, "zscore")
                # logger.info(f"  Reward RLDEEPAGENT (zscore): {reward}")

                # reward = self.get_reward(scores_t0, scores_t1, "bradleyterry")
                # logger.info(f"  Reward RLDEEPAGENT (bradleyterry): {reward}")

                # reward = self.get_reward(scores_t0, scores_t1, "delta")
                # logger.info(f"  Reward RLDEEPAGENT (delta): {reward}")

                # reward = self.get_reward(scores_t0, scores_t1, "positional")
                # logger.info(f"  Reward RLDEEPAGENT (positional): {reward}")

                self.reward_store.append(reward)

            except Exception as e:
                logger.error(f"Error in training mode: {e}")

        return action
