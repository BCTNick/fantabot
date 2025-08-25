# Import NN stuff
from collections import deque
from venv import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import random 
import copy
from model_rl_deep_agent.model import Linear_QNet, QTrainer

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

batch_size = 64
lr = 0.001


class RLDeepAgent(Agent):
    def __init__(self, agent_id, mode: str = "inference"):
        super().__init__(mode)
        self.mode = mode
        self.agent_id = agent_id

        # Concerning the model
        self.n_auctions = 0
        self.epsilon = 0  
        self.gamma = 0.9 
        self.memory = deque(maxlen=100000)
        
        # Calculate input size:
        # 3 base features (auction_progress, player_eval, price)
        # + 12 highest bidder one-hot encoding 
        # + 3 self features (credits, bestxi_eval, squad_eval)
        # + 11 other agents * 3 features each = 33
        # Total: 3 + 12 + 3 + 33 = 51
        self.input_size = 51
        
        self.model = Linear_QNet(input_size=self.input_size, hidden_size=256, output_size=1)
        self.trainer = QTrainer(model=self.model, lr=lr, gamma=self.gamma)

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

    def get_auction_progress(self, other_agents: List[Agent],updated_player_list: List[Player]) -> float:

        # TODO : understand if the to, t1 makes sense in this function
        # Get the sum of the players evaluation of every squads' agent 
        squad_evaluations = 0
        for agent in other_agents + [self]:
            squad_evaluations += agent.squad.objective(bestxi=False, standardized=True)

        auction_progress = squad_evaluations / self.sum_best_overall 

        return auction_progress

    def get_score(self, agent: Agent, agents: List[Agent], updated_player_list: List[Player]) -> float:
        
        auction_progress = self.get_auction_progress(agents, updated_player_list) # TODO for optimization this could go in get state before the get scores and be passed as an argument

        # Get the score of my agent
        weighted_eval = agent.squad.objective(bestxi=True, standardized=True) * 0.9 + agent.squad.objective(bestxi=False, standardized=True) * 0.1
        credit_ratio = agent.current_credits / agent.initial_credits
        remaining_slots = agent.squad.get_remaining_slots(agent.slots, "total") / agent.slots.get_numbers("total")
        # remaining_players_value = agent.get_remaining_players_value(updated_player_list) #Maybe its useless

        # Dynamic weights based on auction stage
        w_e = 0.2 + 0.7 * auction_progress  # team value matters more when good players are finishing
        w_c = 0.5 - 0.5 * auction_progress  # credits matter less when good players are finishing
        w_s = 0.1 - 0.1 * auction_progress  # slots matter less when good players are finishing
        
        # Normalize weights just in case floating point errors occur
        total_w = w_c + w_e + w_s
        score = (weighted_eval * w_e + credit_ratio * w_c + remaining_slots * w_s) / total_w
        return score

    def get_state(self, action: str, current_player: Player, current_price: int, highest_bidder: str, updated_player_list: List[Player], other_agents: List[Agent], mode: chr = "t0") -> float:
        
        if mode == "t0":

            # Get my score
            my_score = self.get_score(self, other_agents, updated_player_list)

            scores = []
            # Get the score of other agents 
            for agent in other_agents:
                score = self.get_score(agent, other_agents, updated_player_list)
                scores.append(score)

            scores.append(my_score)

            state = (my_score - np.mean(scores)) / np.std(scores) if np.std(scores) != 0 else 0

        elif mode == "t1":  
            # Create temporary copies for simulation without affecting real auction
            
            # Deep copy agents to avoid modifying the original ones
            temp_other_agents = copy.deepcopy(other_agents)
            temp_self = copy.deepcopy(self)
            temp_player = copy.deepcopy(current_player)
            temp_updated_players_list = copy.deepcopy(updated_player_list)

            # Simulate the auction outcome on temporary copies
            if action == "offer+1": 
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

            # Get scores using temporary copies
            my_score = self.get_score(temp_self, temp_other_agents, temp_updated_players_list)

            scores = []
            # Get the score of other agents using temporary copies
            for agent in temp_other_agents:
                score = self.get_score(agent, temp_other_agents, temp_updated_players_list)
                scores.append(score)

            scores.append(my_score)

            if np.std(scores) != 0: 
                state = (my_score - np.mean(scores)) / np.std(scores) 
            else:
                state = 0


        return state

    def nn(self, current_player: Player, current_price: int, highest_bidder: str, updated_player_list: List[Player], other_agents: List[Agent]) -> str:
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
        # If no highest bidder or bidder not found, all remain 0
        
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
        
        input_tensor = torch.tensor(all_features, dtype=torch.float32)
        
        probability = self.model.forward(input_tensor)
        return "offer_+1" if probability > 0.5 else "no_offer"

    def get_reward(self, state_t0: np.ndarray, state_t1: np.ndarray, action: str) -> float:
        return state_t1 / state_t0 - 1

    def make_offer_decision(self, current_player: Player, current_price: int, highest_bidder: str, player_list: List[Player], other_agents: List[Agent]) -> str:

        # Get the auction progress (current player not included)
        self.auction_progress = self.get_auction_progress(other_agents, player_list)

        # Get the action from the model
        action = self.nn(current_player, current_price, highest_bidder, player_list, other_agents)

        # update the model if on training 
        if self.mode == "training":
            # get the state of the agent compared to other partecipants before the auction of the current player 
            state_t0 = self.get_state(action, current_player, current_price, highest_bidder, player_list, other_agents, mode = "t0")

            # get the state of the agent after the auction
            state_offer = self.get_state("offer+1", current_player, current_price, highest_bidder, player_list, other_agents, mode = "t1")
            state_no_bid = self.get_state("no_offer", current_player, current_price, highest_bidder, player_list, other_agents, mode = "t1")

            # Compare future states to determine action
            if state_offer > state_no_bid:
                action = "offer_+1"
            
            else:
                action = "no_offer"


            # reward = self.get_reward(state_t0, state_offer, action)

            logger.info(f"  State0: {state_t0}")
            logger.info(f"  State1 +1: {state_offer}")
            logger.info(f"  State1 no_bid: {state_no_bid}")

            # logger.info(f"  Reward RLDEEPAGENT: {reward}")


            # TODO:  Store everything to send at the training function at the end of the auction

        return action
