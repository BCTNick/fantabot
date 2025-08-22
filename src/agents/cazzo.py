# Import NN stuff
import torch
import torch.nn as nn
import torch.nn.functional as F

# Data 
import numpy as np
from typing import List, Tuple

# Import using full module path that works with sys.path setup
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents.training_rl_deep_agent import train 
from agents.agent_class import Agent
from models import Player, Squad, Slots

class RLDeepAgent(Agent):
    def __init__(self, agent_id, mode: str = "inference"):
        super().__init__(mode)
        self.mode = mode
        self.agent_id = agent_id

    def initialize(self, players: List[Player], slots: Slots, initial_credits: int, num_participants: int):
        """Initialize the agent for auction"""
        super().initialize(players, slots, initial_credits, num_participants)
        self.players = players
        self.slots = slots  
        self.current_credits = initial_credits
        self.initial_credits = initial_credits
        self.num_participants = num_participants

    def get_auction_progress(self, agents: List[Agent],updated_player_list: List[Player]) -> float:

        # Get the sum of the evaluation of the best players as if they were all allocated in every team
        best_players = sorted([p for p in updated_player_list],
                                  key=lambda p: p.standardized_evaluation, reverse=True)[:self.slots.get_numbers("total") * self.num_participants]

        best_evaluations = sum(p.standardized_evaluation for p in best_players)

        # Get the sum of the players evaluation of every squads' agent
        squad_evaluations = 0
        for agent in agents:
            squad_evaluations += agent.squad.objective(bestxi=False, standardized=True)

        auction_progress = squad_evaluations / best_evaluations 

        return auction_progress

    def get_remaining_players_value(self, updated_player_list: List[Player]) -> float:
        
        # Calculate how many best players we need to consider
        best_players_count = self.slots.get_numbers("total") * self.num_participants
        
        # Get the best N players overall (regardless of availability)
        best_players_overall = sorted(updated_player_list, 
                                    key=lambda p: p.evaluation, reverse=True)[:best_players_count]
        sum_best_overall = sum(p.evaluation for p in best_players_overall)
        
        # Get available players (fantasy_team is None) and take the best N of them
        available_players = [p for p in updated_player_list if p.fantasy_team is None]
        best_available_players = sorted(available_players,
                                      key=lambda p: p.evaluation, reverse=True)[:best_players_count]
        sum_best_available = sum(p.evaluation for p in best_available_players)
        
        # Return ratio (avoid division by zero)
        if sum_best_overall == 0:
            return 1.0
        
        return sum_best_available / sum_best_overall

    def get_state(self, agents: List[Agent], updated_player_list: List[Player]) -> float:

        auction_progress = self.get_auction_progress(agents, updated_player_list)

        # Get the score of my agent
        weighted_eval = self.squad.objective(bestxi=True, standardized=True) * 0.9 + self.squad.objective(bestxi=False, standardized=True) * 0.1
        credit_ratio = self.current_credits / self.initial_credits
        remaining_slots = self.squad.get_remaining_slots(self.slots, "total") / self.slots.get_numbers("total")
        remaining_players_value = self.get_remaining_players_value(updated_player_list)
        
        # Get the score of other agents 
        scores_matrix = []
        pass

    def nn():
        pass

    def get_reward(self, state_t0: np.ndarray, state_t1: np.ndarray, action: str) -> float:
        pass


         
    def make_offer_decision(self, current_player: Player, current_price: int, highest_bidder: str, player_list: List[Player], other_agents: List[Agent]) -> str:

        # Use the RL model to predict the action
        action = self.nn(current_player, current_price, highest_bidder, player_list, other_agents)

        # update the model if on training 
        if self.mode == "training":

            train(self) # TODO

        return action
