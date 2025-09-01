"""
Core auction manager with improved logic
"""

import random
import threading
import queue
from typing import List, Dict, Optional, Callable
from src.models import Player, Slots
from src.agents.agent_class import Agent
from src.agents.human_agent import HumanAgent
from src.utils.logging_handler import AuctionLogger
from src.utils.validators import AuctionValidator


class EnhancedAuction:
    """Enhanced auction system with better management and UI integration"""
    
    def __init__(self, slots: Slots, agents: List[Agent], players: List[Player], 
                 initial_credits: int = 1000):
        self.slots = slots  # Keep as Slots object
        self.slots_dict = slots.to_dict()  # Also keep as dict for compatibility
        self.initial_credits = initial_credits
        self.agents = agents
        self.players = players
        self.current_player = None
        self.current_price = 0
        self.highest_bidder = None
        self.running = False
        
        # Callbacks for UI updates
        self.on_player_start: Optional[Callable] = None
        self.on_bid_made: Optional[Callable] = None
        self.on_player_sold: Optional[Callable] = None
        self.on_human_input_needed: Optional[Callable] = None
        
        # Human decision queue
        self.human_decision_queue = queue.Queue()
        
        # Initialize all agents
        for agent in self.agents:
            agent.initialize(players, slots, initial_credits, len(agents))
    
    def set_callbacks(self, on_player_start=None, on_bid_made=None, 
                     on_player_sold=None, on_human_input_needed=None):
        """Set callback functions for UI updates"""
        self.on_player_start = on_player_start
        self.on_bid_made = on_bid_made
        self.on_player_sold = on_player_sold
        self.on_human_input_needed = on_human_input_needed
    
    def can_participate_in_bid(self, agent: Agent, offer_price: int, 
                              position: str, slots: Dict[str, int]) -> bool:
        """Check if agent can participate in bidding"""
        return AuctionValidator.can_agent_participate(agent, offer_price - 1, position, self.slots_dict)
    
    def run_full_auction(self, auction_type: str = "chiamata", per_ruolo: bool = True):
        """Run complete auction"""
        self.running = True
        
        # Log auction start
        AuctionLogger.log_auction_start(
            self.slots, self.initial_credits, len(self.agents), len(self.players)
        )
        
        try:
            if auction_type == "chiamata":
                self._run_chiamata_auction(per_ruolo)
            else:
                self._run_classic_auction(per_ruolo)
        except Exception as e:
            AuctionLogger.log_error(f"Errore durante l'asta: {str(e)}", e)
        finally:
            self.running = False
            AuctionLogger.log_auction_end()
    
    def _run_chiamata_auction(self, per_ruolo: bool):
        """Run chiamata-style auction"""
        available_players = self.players.copy()
        
        if per_ruolo:
            roles = ['GK', 'DEF', 'MID', 'ATT']
            for role in roles:
                if not self.running:
                    break
                
                AuctionLogger.log_role_start(role)
                role_players = [p for p in available_players if p.role == role]
                random.shuffle(role_players)
                
                for player in role_players:
                    if not self.running:
                        break
                    self._auction_single_player(player)
        else:
            random.shuffle(available_players)
            for player in available_players:
                if not self.running:
                    break
                self._auction_single_player(player)
    
    def _run_classic_auction(self, per_ruolo: bool):
        """Run classic auction (to be implemented based on original logic)"""
        # For now, fallback to chiamata style
        self._run_chiamata_auction(per_ruolo)
    
    def _auction_single_player(self, player: Player):
        """Auction a single player with enhanced logic"""
        if not self.running:
            return
        
        # Initialize auction state
        self.current_player = player
        self.current_price = 1
        self.highest_bidder = None
        
        # Notify UI
        if self.on_player_start:
            self.on_player_start(player)
        
        AuctionLogger.log_player_auction_start(player)
        
        # Run bidding rounds
        rounds_without_bids = 0
        max_rounds_without_bids = 3
        
        while rounds_without_bids < max_rounds_without_bids and self.running:
            AuctionLogger.log_bid_round(self.current_price)
            
            # Get automatic agent bids
            automatic_offers = self._get_automatic_bids()
            
            # Process automatic bids
            if automatic_offers:
                highest_offer = max(automatic_offers, key=lambda x: x[1])
                agent, offer_price = highest_offer
                self.current_price = offer_price
                self.highest_bidder = agent
                AuctionLogger.log_highest_bid(agent.agent_id, offer_price)
                rounds_without_bids = 0
            
            # Handle human agents
            human_agents = [agent for agent in self.agents if isinstance(agent, HumanAgent)]
            if human_agents:
                human_decision = self._handle_human_bidding(human_agents)
                
                if human_decision["agent_id"] != "nessuno":
                    human_agent = next(agent for agent in human_agents 
                                     if agent.agent_id == human_decision["agent_id"])
                    human_offer_price = human_decision["amount"]
                    
                    # Validate human bid
                    is_valid, error_msg = AuctionValidator.validate_bid(
                        human_agent, human_offer_price, self.current_price - 1,
                        player.role, self.slots_dict
                    )
                    
                    if is_valid:
                        self.current_price = human_offer_price
                        self.highest_bidder = human_agent
                        AuctionLogger.log_bid(human_agent.agent_id, human_offer_price)
                        rounds_without_bids = 0
                    else:
                        AuctionLogger.log_error(f"Offerta di {human_agent.agent_id} non valida: {error_msg}")
                        rounds_without_bids += 1
                else:
                    AuctionLogger.log_no_bid("Agenti umani")
                    if not automatic_offers:
                        rounds_without_bids += 1
            else:
                if not automatic_offers:
                    rounds_without_bids += 1
        
        # Finalize sale
        self._finalize_player_sale(player)
    
    def _get_automatic_bids(self) -> List[tuple]:
        """Get bids from automatic agents"""
        offers = []
        
        for agent in self.agents:
            if isinstance(agent, HumanAgent):
                continue
            
            if self.can_participate_in_bid(agent, self.current_price + 1, 
                                         self.current_player.role, self.slots_dict):
                
                decision = agent.make_offer_decision(
                    self.current_player, self.current_price, self.highest_bidder,
                    self.players, self.agents
                )
                
                if decision != "no_offer":
                    if decision == "offer_+1":
                        offer_price = self.current_price + 1
                    else:
                        try:
                            offer_price = int(decision.split("_")[1])
                        except:
                            offer_price = self.current_price + 1
                    
                    offers.append((agent, offer_price))
                    AuctionLogger.log_bid(agent.agent_id, offer_price)
                else:
                    AuctionLogger.log_no_bid(agent.agent_id)
        
        return offers
    
    def _handle_human_bidding(self, human_agents: List[Agent]) -> Dict:
        """Handle human agent bidding"""
        if not self.on_human_input_needed:
            # No UI callback, default to no bid
            return {"agent_id": "nessuno", "amount": 0}
        
        # Request human input via callback
        return self.on_human_input_needed(
            self.current_player, human_agents, self.current_price, self.highest_bidder
        )
    
    def _finalize_player_sale(self, player: Player):
        """Finalize player sale"""
        if self.highest_bidder:
            # Update player
            player.fantasy_team = self.highest_bidder.agent_id
            player.final_cost = self.current_price
            
            # Update agent
            self.highest_bidder.current_credits -= self.current_price
            self.highest_bidder._squad.append(player)
            
            # Log and notify
            AuctionLogger.log_player_sold(
                player.name, self.highest_bidder.agent_id, 
                self.current_price, self.highest_bidder.current_credits
            )
            
            if self.on_player_sold:
                self.on_player_sold(player, self.highest_bidder.agent_id, self.current_price)
        else:
            player.fantasy_team = "UNSOLD"
            player.final_cost = 0
            AuctionLogger.log_player_unsold(player.name)
    
    def stop_auction(self):
        """Stop the auction"""
        self.running = False
    
    def get_auction_summary(self) -> Dict:
        """Get current auction summary"""
        summary = {
            "total_players": len(self.players),
            "sold_players": len([p for p in self.players if p.fantasy_team]),
            "unsold_players": len([p for p in self.players if not p.fantasy_team]),
            "agents": []
        }
        
        for agent in self.agents:
            agent_summary = {
                "name": agent.agent_id,
                "credits_remaining": agent.current_credits,
                "players_count": len(agent.squad),
                "squad_value": agent.squad.objective(standardized=False),
                "best_xi_value": agent.squad.objective(bestxi=True, standardized=False),
                "squad_by_role": {
                    "GK": len(agent.squad.gk),
                    "DEF": len(agent.squad.def_),
                    "MID": len(agent.squad.mid),
                    "ATT": len(agent.squad.att)
                }
            }
            summary["agents"].append(agent_summary)
        
        return summary
