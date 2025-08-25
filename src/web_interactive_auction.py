"""
Web Interactive Auction system - Non-blocking auction for web interfaces
"""

import random
import logging
from typing import List, Dict, Optional, Callable
from threading import Event, Thread
import time
from enum import Enum

from .agents.human_agent import HumanAgent
from .models import Player, Slots
from .agents.agent_class import Agent

logger = logging.getLogger(__name__)


class AuctionState(Enum):
    CREATED = "created"
    RUNNING = "running"
    WAITING_FOR_HUMAN = "waiting_for_human"
    COMPLETED = "completed"
    ERROR = "error"


class WebInteractiveAuction:
    """Non-blocking auction system for web interfaces"""
    
    def __init__(self, slots: Slots, agents: List[Agent], players: List[Player], initial_credits: int = 1000):
        # Rules 
        self.slots = slots.to_dict()
        self.initial_credits = initial_credits

        # Agents and players
        self.agents = agents
        self.players = players
        self.human_agents = [agent for agent in agents if isinstance(agent, HumanAgent)]

        # Auction state
        self.state = AuctionState.CREATED
        self.current_player: Optional[Player] = None
        self.current_price = 0
        self.highest_bidder = None
        
        # Web interaction state
        self.waiting_for_human_bid = False
        self.human_bid_event = Event()
        self.pending_human_bid = None
        
        # Logging
        self.auction_logs = []
        self.log_callback: Optional[Callable] = None
        
        # Threading
        self.auction_thread = None
        self.should_stop = False

        # Initialize all agents
        for agent in self.agents:
            agent.initialize(players, slots, initial_credits, len(agents))
                
    def set_log_callback(self, callback: Callable):
        """Set callback function for log events"""
        self.log_callback = callback
        
    def log(self, message: str, level: str = "info"):
        """Log a message and call callback if set"""
        log_entry = {
            'timestamp': time.time(),
            'level': level,
            'message': message,
            'player': self.current_player.name if self.current_player else None,
            'price': self.current_price,
            'highest_bidder': self.highest_bidder
        }
        
        self.auction_logs.append(log_entry)
        
        if self.log_callback:
            self.log_callback(log_entry)
            
        # Also log to standard logger
        getattr(logger, level)(message)

    def can_participate_in_bid(self, agent: Agent, offer_price: int, position: str, slots: Dict[str, int]) -> bool:
        """Check if agent can participate in current bid"""
        # Check if they have a free slot in this position
        role_map = {"GK": agent.squad.gk, "DEF": agent.squad.def_, "MID": agent.squad.mid, "ATT": agent.squad.att}
        if len(role_map[position]) >= slots[position]:
            return False
        
        # Check if the offer is valid (higher than current price)
        if offer_price <= self.current_price:
            return False
        
        # Check if they can afford it
        credits_after_purchase = agent.current_credits - offer_price
        total_slots = sum(slots.values())
        total_players = len(agent.squad)
        remaining_slots = total_slots - (total_players + 1)

        return credits_after_purchase >= remaining_slots

    def get_valid_human_agents(self) -> List[dict]:
        """Get list of human agent info that can bid on current player"""
        valid_agents = []
        for agent in self.human_agents:
            if self.can_participate_in_bid(agent, self.current_price + 1, self.current_player.role, self.slots):
                valid_agents.append({
                    'id': agent.agent_id,
                    'name': agent.name
                })
        return valid_agents

    def process_automatic_agents(self):
        """Process bids from automatic agents"""
        agent_index = 0
        no_bid_count = 0
        
        while no_bid_count < len([a for a in self.agents if not isinstance(a, HumanAgent)]):
            if self.should_stop:
                return False
                
            current_agent = self.agents[agent_index]
            
            # Skip human agents
            if isinstance(current_agent, HumanAgent):
                agent_index = (agent_index + 1) % len(self.agents)
                continue
            
            # Check if agent can participate in this bid
            if self.can_participate_in_bid(agent=current_agent, offer_price=self.current_price + 1, 
                                            position=self.current_player.role, slots=self.slots):
                
                # Get all other agents except the current one
                other_agents = [agent for agent in self.agents if agent.agent_id != current_agent.agent_id]
                
                # Get current agent's decision
                decision = current_agent.make_offer_decision(
                    current_player=self.current_player,
                    current_price=self.current_price,
                    highest_bidder=self.highest_bidder,
                    player_list=self.players,
                    other_agents=other_agents
                )
                
                if decision == "offer_+1":
                    self.current_price += 1
                    self.highest_bidder = current_agent.agent_id
                    no_bid_count = 0
                    
                    self.log(f"💰 {current_agent.name} bids {self.current_price}")
                    time.sleep(0.5)  # Small delay for realism
                else:
                    no_bid_count += 1
                    self.log(f"❌ {current_agent.name} passes")
            else:
                no_bid_count += 1
                self.log(f"🚫 {current_agent.name} cannot bid")
            
            agent_index = (agent_index + 1) % len(self.agents)
            
        return True

    def wait_for_human_bids(self, timeout: int = 30) -> bool:
        """Wait for human bids with timeout"""
        valid_human_agents = self.get_valid_human_agents()
        
        if not valid_human_agents:
            self.log("No human agents can bid on this player")
            return True  # Continue to next player
        
        self.waiting_for_human_bid = True
        self.state = AuctionState.WAITING_FOR_HUMAN
        
        valid_agent_names = [agent['name'] for agent in valid_human_agents]
        self.log(f"⏳ Waiting for human bids. Valid agents: {', '.join(valid_agent_names)}")
        
        # Wait for bid or timeout
        bid_received = self.human_bid_event.wait(timeout)
        
        self.waiting_for_human_bid = False
        self.human_bid_event.clear()
        
        if bid_received and self.pending_human_bid:
            agent_id, bid_amount = self.pending_human_bid
            self.pending_human_bid = None
            
            # Check if humans decided to pass
            if agent_id == "nessuno":
                self.log("❌ Human agents pass - no bid")
                return True  # End auction for this player
            
            # Validate the bid
            bidding_agent = next((agent for agent in self.agents if agent.agent_id == agent_id), None)
            
            if bidding_agent and self.can_participate_in_bid(bidding_agent, bid_amount, self.current_player.role, self.slots):
                self.current_price = bid_amount
                self.highest_bidder = agent_id
                self.log(f"💰 {bidding_agent.name} (HUMAN) bids {bid_amount}")
                return False  # Continue auction round
            else:
                self.log(f"❌ Invalid bid from {bidding_agent.name if bidding_agent else agent_id}: {bid_amount}")
                
        return True  # No valid bids, end auction for this player

    def submit_human_bid(self, agent_id: str, bid_amount: int) -> dict:
        """Submit a bid from a human agent"""
        if not self.waiting_for_human_bid:
            return {'success': False, 'error': 'Not waiting for human bids'}
        
        valid_agents = self.get_valid_human_agents()
        valid_agent_ids = [agent['id'] for agent in valid_agents]
        
        if agent_id not in valid_agent_ids:
            return {'success': False, 'error': 'Agent cannot bid on this player'}
        
        if bid_amount <= self.current_price:
            return {'success': False, 'error': f'Bid must be higher than {self.current_price}'}
        
        # Check if agent can afford the bid
        bidding_agent = next((agent for agent in self.agents if agent.agent_id == agent_id), None)
        if not self.can_participate_in_bid(bidding_agent, bid_amount, self.current_player.role, self.slots):
            return {'success': False, 'error': 'Agent cannot afford this bid'}
        
        self.pending_human_bid = (agent_id, bid_amount)
        self.human_bid_event.set()
        
        return {'success': True, 'message': f'Bid of {bid_amount} submitted for {bidding_agent.name}'}

    def submit_human_pass(self) -> dict:
        """Submit a pass (no bid) from human agents"""
        if not self.waiting_for_human_bid:
            return {'success': False, 'error': 'Not waiting for human bids'}
        
        self.log("❌ Human agents decide to pass")
        self.human_bid_event.set()  # Signal that decision was made (no bid)
        
        return {'success': True, 'message': 'Human agents pass'}

    def submit_human_pass(self) -> dict:
        """Submit a pass (no bid) from human agents"""
        if not self.waiting_for_human_bid:
            return {'success': False, 'error': 'Not waiting for human bids'}
        
        self.pending_human_bid = ("nessuno", 0)  # Special marker for pass
        self.human_bid_event.set()
        
        return {'success': True, 'message': 'Human agents pass - no bid submitted'}

    def auction_single_player(self, player: Player):
        """Auction a single player with web interaction support"""
        self.current_player = player
        self.current_price = 0
        self.highest_bidder = None
        
        self.log(f"\n🏆 AUCTION START: {player.name} ({player.role}) - Evaluation: {player.evaluation}")
        
        while not self.should_stop:
            # Process automatic agents
            if not self.process_automatic_agents():
                continue
                
            # Wait for human bids
            if self.wait_for_human_bids():
                break  # No more bids, end auction
        
        # Finalize sale
        if self.highest_bidder and not self.should_stop:
            player.fantasy_team = self.highest_bidder
            player.final_cost = self.current_price
            
            winner_agent = next(agent for agent in self.agents if agent.agent_id == self.highest_bidder)
            winner_agent._squad.append(player)
            winner_agent.current_credits -= self.current_price
            
            self.log(f"✅ {player.name} SOLD to {winner_agent.name} for {self.current_price}")
            return self.highest_bidder
        else:
            player.fantasy_team = "UNSOLD"
            player.final_cost = 0
            self.log(f"❌ {player.name} UNSOLD")
            return None

    def run_auction_thread(self, auction_type: str = "random", per_ruolo: bool = True):
        """Run auction in separate thread"""
        try:
            self.state = AuctionState.RUNNING
            self.log("🚀 Auction started!")
            
            roles = ["GK", "DEF", "MID", "ATT"]
            
            players_to_auction = self.players[:]
            if auction_type == "random":
                random.shuffle(players_to_auction)
            elif auction_type == "listone":
                players_to_auction.sort(key=lambda p: p.name)
            elif auction_type == "chiamata":
                players_to_auction.sort(key=lambda p: p.evaluation, reverse=True)

            if per_ruolo:
                for role in roles:
                    if self.should_stop:
                        break
                        
                    self.log(f"\n🏈 STARTING {role} AUCTIONS")
                    
                    while not self.should_stop:
                        all_full = all(len(getattr(agent.squad, role.lower() if role != "DEF" else "def_")) >= self.slots[role] 
                                     for agent in self.agents)
                        if all_full:
                            break
                            
                        next_player = next((p for p in players_to_auction if p.role == role and p.fantasy_team is None), None)
                        if next_player is None:
                            break
                            
                        self.auction_single_player(next_player)
            else:
                total_slots = sum(self.slots.values())
                while not self.should_stop:
                    all_full = all(len(agent.squad) >= total_slots for agent in self.agents)
                    if all_full:
                        break
                        
                    next_player = next((p for p in players_to_auction if p.fantasy_team is None), None)
                    if next_player is None:
                        break
                        
                    self.auction_single_player(next_player)
            
            if not self.should_stop:
                self.state = AuctionState.COMPLETED
                self.log("🏁 Auction completed!")
            else:
                self.log("⏹️ Auction stopped")
                
        except Exception as e:
            self.state = AuctionState.ERROR
            self.log(f"❌ Auction error: {str(e)}", "error")
            
    def start_auction(self, auction_type: str = "random", per_ruolo: bool = True):
        """Start auction in background thread"""
        if self.auction_thread and self.auction_thread.is_alive():
            raise RuntimeError("Auction is already running")
            
        self.should_stop = False
        self.auction_thread = Thread(target=self.run_auction_thread, args=(auction_type, per_ruolo))
        self.auction_thread.start()
        
    def stop_auction(self):
        """Stop running auction"""
        self.should_stop = True
        if self.waiting_for_human_bid:
            self.human_bid_event.set()  # Unblock any waiting
            
    def get_current_state(self) -> dict:
        """Get current auction state for API"""
        return {
            'state': self.state.value,
            'current_player': {
                'name': self.current_player.name,
                'role': self.current_player.role,
                'evaluation': self.current_player.evaluation
            } if self.current_player else None,
            'current_price': self.current_price,
            'highest_bidder': self.highest_bidder,
            'waiting_for_human_bid': self.waiting_for_human_bid,
            'valid_human_agents': self.get_valid_human_agents() if self.waiting_for_human_bid else [],
            'logs': self.auction_logs[-10:]  # Last 10 log entries
        }
