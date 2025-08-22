"""
Agent Class and random agent
"""

import random
from typing import List
from src.models import Player, Slots, Squad

class Agent:
    def __init__(self, agent_id: str, initial_credits: int = 1000, slots: Slots = None, _squad: List[Player] = None):
        self.agent_id = agent_id
        self.current_credits = initial_credits
        self.initial_credits = initial_credits
        self.slots = slots if slots is not None else Slots()
        self._squad: List[Player] = _squad if _squad is not None else []

    @property
    def squad(self):
        return Squad(self._squad)
    
    def initialize(self, listone: List[Player], slots: Slots, initial_credits: int, num_participants: int):
        """Initialize agent with auction environment - to be overridden by subclasses if needed"""
        self.current_credits = initial_credits
        self.initial_credits = initial_credits
        self.num_participants = num_participants
        self.slots = slots
        self.listone = listone

    def make_offer_decision(self, current_player, current_price, highest_bidder, player_list, other_agents) -> str:
        raise NotImplementedError("This method should be implemented by subclasses.")

class RandomAgent(Agent):
    def make_offer_decision(self, current_player, current_price, highest_bidder, player_list, other_agents) -> str:
        return "offer_+1" if random.random() < 0.5 else "no_offer"

