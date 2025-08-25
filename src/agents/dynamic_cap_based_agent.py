"""
Dynamic Cap Based agent class
"""

from src.agents.agent_class import Agent
from src.models import Player, Squad, Slots
from typing import List

class DynamicCapAgent(Agent):
    def __init__(self, agent_id: str, name: str = None, cap_strategy: str = "bestxi_based", bestxi_budget: float = 0.95):
        super().__init__(agent_id, name)
        self.cap_strategy = cap_strategy
        self.bestxi_budget = bestxi_budget
        self.listone = {}  # Will be filled during initialize
        self.players_caps = {}  # Dictionary to store player caps
    
    def initialize(self, listone: List[Player], slots: Slots, initial_credits: int, num_participants: int):
        """Initialize agent with auction environment"""
        super().initialize(listone, slots, initial_credits, num_participants)
        # Store references for use in make_offer_decision
        self.num_participants = num_participants  # Explicitly store this
        self.initial_credits = initial_credits
        self.slots = slots

    
    def make_offer_decision(self, current_player, current_price, highest_bidder, player_list, other_agents: List[Agent]) -> str:
        """Make bidding decision based on cap calculation"""

        num_participants = self.num_participants
        slots = self.slots
        # Filter to only include players who haven't been bought yet
        listone = [p for p in player_list if p.fantasy_team is None]

        if self.cap_strategy == "bestxi_based":

            # Calculate average budget per team for the best players 
            avg_budget_per_team = self.current_credits * self.bestxi_budget

            # Filter best players by role based on actual slot requirements:
            best_gk = sorted([p for p in listone if p.role == "GK"], 
                           key=lambda p: p.evaluation, reverse=True)[:1 * num_participants]
            best_def = sorted([p for p in listone if p.role == "DEF"], 
                            key=lambda p: p.evaluation, reverse=True)[:3 * num_participants]
            best_mid = sorted([p for p in listone if p.role == "MID"], 
                            key=lambda p: p.evaluation, reverse=True)[:3 * num_participants]
            best_att = sorted([p for p in listone if p.role == "ATT"], 
                            key=lambda p: p.evaluation, reverse=True)[:3 * num_participants]
            
            # Combine all best players
            best_players = best_gk + best_def + best_mid + best_att

            # Calculate total evaluation points available
            total_evaluation = sum(p.evaluation for p in best_players)

            # Calculate value per evaluation point
            value_per_point = (num_participants * avg_budget_per_team) / total_evaluation

            # For ONLY the best players, compute cap and store in dictionary
            for player in best_players:
                cap = player.evaluation * value_per_point
                self.players_caps[player.name] = cap


            # Calculate average budget per team for the acceptable players
            avg_budget_per_team = self.current_credits * (1 - self.bestxi_budget)

            # Filter acceptable players by role based on actual slot requirements:
            acceptable_gk = sorted([p for p in listone if p.role == "GK"], 
                           key=lambda p: p.evaluation, reverse=True)[1 * num_participants : slots.gk * num_participants]
            acceptable_def = sorted([p for p in listone if p.role == "DEF"], 
                            key=lambda p: p.evaluation, reverse=True)[3 * num_participants :slots.def_ * num_participants]
            acceptable_mid = sorted([p for p in listone if p.role == "MID"], 
                            key=lambda p: p.evaluation, reverse=True)[3 * num_participants :slots.mid * num_participants]
            acceptable_att = sorted([p for p in listone if p.role == "ATT"], 
                            key=lambda p: p.evaluation, reverse=True)[3 * num_participants :slots.att * num_participants]

            # Combine all acceptable players
            acceptable_players = acceptable_gk + acceptable_def + acceptable_mid + acceptable_att 

            # Calculate total evaluation points available
            total_evaluation = sum(p.evaluation for p in acceptable_players)

            # Calculate value per evaluation point
            value_per_point = (num_participants * avg_budget_per_team) / (total_evaluation + 0.0001)  # Avoid division by zero

            # For ONLY the average players, compute cap and store in dictionary
            for player in acceptable_players:
                cap = player.evaluation * value_per_point
                self.players_caps[player.name] = cap
 
        if self.cap_strategy == "tier_based":

            # Calculate average budget per team for the first-tier players
            avg_budget_per_team = self.current_credits * 0.6

            # Filter best players by role based on actual slot requirements:
            best_gk = sorted([p for p in listone if p.role == "GK"], 
                           key=lambda p: p.evaluation, reverse=True)[:1 * num_participants]
            best_def = sorted([p for p in listone if p.role == "DEF"], 
                            key=lambda p: p.evaluation, reverse=True)[:1 * num_participants]
            best_mid = sorted([p for p in listone if p.role == "MID"], 
                            key=lambda p: p.evaluation, reverse=True)[:1 * num_participants]
            best_att = sorted([p for p in listone if p.role == "ATT"], 
                            key=lambda p: p.evaluation, reverse=True)[:1 * num_participants]
            
            # Combine all best players
            best_players = best_gk + best_def + best_mid + best_att

            # Calculate total evaluation points available
            total_evaluation = sum(p.evaluation for p in best_players)

            # Calculate value per evaluation point
            value_per_point = (num_participants * avg_budget_per_team) / total_evaluation

            # For ONLY the best players, compute cap and store in dictionary
            for player in best_players:
                cap = player.evaluation * value_per_point
                self.players_caps[player.name] = cap

            # Calculate average budget per team for the second-tier players
            avg_budget_per_team = self.current_credits * 0.25

            # Filter second-tier players by role (excluding goalkeepers):
            second_def = sorted([p for p in listone if p.role == "DEF"], 
                            key=lambda p: p.evaluation, reverse=True)[1 * num_participants:2 * num_participants]
            second_mid = sorted([p for p in listone if p.role == "MID"], 
                            key=lambda p: p.evaluation, reverse=True)[1 * num_participants:2 * num_participants]
            second_att = sorted([p for p in listone if p.role == "ATT"], 
                            key=lambda p: p.evaluation, reverse=True)[1 * num_participants:2 * num_participants]
            
            # Combine all second-tier players
            second_players = second_def + second_mid + second_att

            # Calculate total evaluation points available for second tier
            total_evaluation = sum(p.evaluation for p in second_players)

            # Calculate value per evaluation point for second tier
            value_per_point = (num_participants * avg_budget_per_team) / (total_evaluation + 0.0001)

            # For second-tier players, compute cap and store in dictionary
            for player in second_players:
                cap = player.evaluation * value_per_point
                self.players_caps[player.name] = cap

            # Calculate average budget per team for the third-tier players
            avg_budget_per_team = self.current_credits * 0.1

            # Filter third-tier players by role (excluding goalkeepers):
            third_def = sorted([p for p in listone if p.role == "DEF"], 
                            key=lambda p: p.evaluation, reverse=True)[2 * num_participants:3 * num_participants]
            third_mid = sorted([p for p in listone if p.role == "MID"], 
                            key=lambda p: p.evaluation, reverse=True)[2 * num_participants:3 * num_participants]
            third_att = sorted([p for p in listone if p.role == "ATT"], 
                            key=lambda p: p.evaluation, reverse=True)[2 * num_participants:3 * num_participants]
            
            # Combine all third-tier players
            third_players = third_def + third_mid + third_att

            # Calculate total evaluation points available for third tier
            total_evaluation = sum(p.evaluation for p in third_players)

            # Calculate value per evaluation point for third tier
            value_per_point = (num_participants * avg_budget_per_team) / (total_evaluation + 0.0001)


            # For third-tier players, compute cap and store in dictionary
            for player in third_players:
                cap = player.evaluation * value_per_point
                self.players_caps[player.name] = cap

            
            # Calculate average budget per team for the acceptable players
            avg_budget_per_team = self.current_credits * 0.05

            # Filter acceptable players by role based on actual slot requirements:
            acceptable_gk = sorted([p for p in listone if p.role == "GK"], 
                           key=lambda p: p.evaluation, reverse=True)[1 * num_participants : slots.gk * num_participants]
            acceptable_def = sorted([p for p in listone if p.role == "DEF"], 
                            key=lambda p: p.evaluation, reverse=True)[3 * num_participants :slots.def_ * num_participants]
            acceptable_mid = sorted([p for p in listone if p.role == "MID"], 
                            key=lambda p: p.evaluation, reverse=True)[3 * num_participants :slots.mid * num_participants]
            acceptable_att = sorted([p for p in listone if p.role == "ATT"], 
                            key=lambda p: p.evaluation, reverse=True)[3 * num_participants :slots.att * num_participants]

            # Combine all acceptable players
            acceptable_players = acceptable_gk + acceptable_def + acceptable_mid + acceptable_att 

            # Calculate total evaluation points available
            total_evaluation = sum(p.evaluation for p in acceptable_players)

            # Calculate value per evaluation point
            value_per_point = (num_participants * avg_budget_per_team) / (total_evaluation + 0.0001)

            # For ONLY the average players, compute cap and store in dictionary
            for player in acceptable_players:
                cap = player.evaluation * value_per_point
                self.players_caps[player.name] = cap



        if current_player.name not in self.players_caps:
            return "no_offer"
        
        max_bid = self.players_caps[current_player.name]
    
        # Don't bid if current price exceeds our calculated maximum
        if current_price >= max_bid:
            return "no_offer"
        else:
            return "offer_+1"