"""
Validation utilities for auction system
"""

from typing import List, Dict, Tuple, Optional
from src.models import Player, Slots
from src.agents.agent_class import Agent


class AuctionValidator:
    """Validation utilities for auction operations"""
    
    @staticmethod
    def validate_agent_config(agents_config: List[Dict]) -> Tuple[bool, str]:
        """Validate agent configuration"""
        if len(agents_config) < 2:
            return False, "Servono almeno 2 agenti per l'asta"
        
        # Check for duplicate names
        names = [agent["name"] for agent in agents_config]
        if len(names) != len(set(names)):
            return False, "Nomi degli agenti devono essere unici"
        
        # Check for empty names
        if any(not agent["name"].strip() for agent in agents_config):
            return False, "Tutti gli agenti devono avere un nome"
        
        return True, ""
    
    @staticmethod
    def validate_auction_settings(initial_credits: int, slots: Slots) -> Tuple[bool, str]:
        """Validate auction settings"""
        if initial_credits <= 0:
            return False, "I crediti iniziali devono essere maggiori di 0"
        
        if any(val <= 0 for val in [slots.gk, slots.def_, slots.mid, slots.att]):
            return False, "Tutti i slot devono essere maggiori di 0"
        
        total_slots = slots.gk + slots.def_ + slots.mid + slots.att
        if initial_credits < total_slots:
            return False, "Crediti insufficienti per acquistare tutti i giocatori necessari"
        
        return True, ""
    
    @staticmethod
    def validate_bid(agent: Agent, offer_price: int, current_price: int, 
                    player_role: str, slots: Dict[str, int]) -> Tuple[bool, str]:
        """Validate a bid offer"""
        # Check if offer is higher than current price
        if offer_price <= current_price:
            return False, f"L'offerta deve essere maggiore di {current_price}"
        
        # Check if agent has enough credits
        if offer_price > agent.current_credits:
            return False, f"Crediti insufficienti (disponibili: {agent.current_credits})"
        
        # Check if agent has a free slot for this role
        role_map = {
            "GK": len(agent.squad.gk), 
            "DEF": len(agent.squad.def_), 
            "MID": len(agent.squad.mid), 
            "ATT": len(agent.squad.att)
        }
        
        if role_map[player_role] >= slots[player_role]:
            return False, f"Nessun slot libero per {player_role}"
        
        # Check if agent can afford remaining players
        credits_after_purchase = agent.current_credits - offer_price
        total_slots = sum(slots.values())
        total_players = len(agent.squad)
        remaining_slots = total_slots - (total_players + 1)
        
        if credits_after_purchase < remaining_slots:
            return False, f"Crediti insufficienti per completare la squadra"
        
        return True, ""
    
    @staticmethod
    def validate_player_data(players: List[Player]) -> Tuple[bool, str]:
        """Validate player data"""
        if not players:
            return False, "Nessun giocatore caricato"
        
        # Check for required fields
        for player in players:
            if not player.name or not player.name.strip():
                return False, "Tutti i giocatori devono avere un nome"
            
            if player.role not in ["GK", "DEF", "MID", "ATT"]:
                return False, f"Ruolo non valido per {player.name}: {player.role}"
            
            if player.evaluation <= 0:
                return False, f"Valutazione non valida per {player.name}: {player.evaluation}"
        
        return True, ""
    
    @staticmethod
    def get_max_bid_for_agent(agent: Agent, current_price: int, 
                             player_role: str, slots: Dict[str, int]) -> int:
        """Calculate maximum valid bid for an agent"""
        # Calculate remaining slots after this purchase
        total_slots = sum(slots.values())
        total_players = len(agent.squad)
        remaining_slots = total_slots - (total_players + 1)
        
        # Maximum bid is current credits minus minimum needed for remaining slots
        max_bid = agent.current_credits - remaining_slots
        
        # Must be at least current_price + 1
        return max(current_price + 1, max_bid)
    
    @staticmethod
    def can_agent_participate(agent: Agent, current_price: int, 
                            player_role: str, slots: Dict[str, int]) -> bool:
        """Check if agent can participate in bidding"""
        is_valid, _ = AuctionValidator.validate_bid(
            agent, current_price + 1, current_price, player_role, slots
        )
        return is_valid
