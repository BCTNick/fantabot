from src.agents.agent_class import Agent
from src.models import Player, Squad, Slots
from typing import List

class HumanAgent(Agent):
    def initialize(self, listone: List[Player], slots: Slots, initial_credits: int, num_participants: int):
        """Initialize agent with auction environment"""
        super().initialize(listone, slots, initial_credits, num_participants)

