from src.agents.agent_class import Agent
from src.models import Player, Squad, Slots
from typing import List

class HumanAgent(Agent):
    def __init__(self, agent_id: str, name: str = None):
        super().__init__(agent_id, name)
        
    def initialize(self, listone: List[Player], slots: Slots, initial_credits: int, num_participants: int):
        """Initialize agent with auction environment"""
        super().initialize(listone, slots, initial_credits, num_participants)

    def make_offer_decision(self, current_player, current_price, highest_bidder, player_list, other_agents) -> str:
        pass
