"""
Auction system
"""

import random
import logging
from typing import List, Dict
from agents.human_agent import HumanAgent
from models import Player, Slots
from agents.agent_class import Agent

# Get logger
logger = logging.getLogger(__name__)


class Auction:
    """Main auction system that manages the bidding process"""
    
    def __init__(self, slots: Slots, agents: List[Agent], players: List[Player], initial_credits: int = 1000):
        # Rules 
        self.slots = slots.to_dict()
        self.initial_credits = initial_credits
        self.verbose = False  # Add verbose flag for logging

        # Agents and listone
        self.agents = agents
        self.players = players

        # Initialize all agents
        for agent in self.agents:
            agent.initialize(players, slots, initial_credits, len(agents))
                
    def can_participate_in_bid(self, agent: Agent, offer_price: int, position: str, slots: Dict[str, int]) -> bool:
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

    def loop_automatic_agents(self):
        
        agent_index = 0
        no_bid_count = 0
    
        while True:
            current_agent = self.agents[agent_index]
            
            # Check if agent can participate in this bid
            if self.can_participate_in_bid(agent=current_agent, offer_price=self.current_price + 1, 
                                            position=self.current_player.role, slots=self.slots) is True:
                
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
                    
                    if self.verbose:
                        logger.info(f"  💰 {current_agent.agent_id} bids {self.current_price}")
                else:
                    no_bid_count += 1
                    if self.verbose:
                        logger.info(f"  ❌ {current_agent.agent_id} passes")
            else:
                no_bid_count += 1
                if self.verbose:
                    logger.info(f"  🚫 {current_agent.agent_id} cannot bid")
            
            agent_index = (agent_index + 1) % len(self.agents)
            
            if no_bid_count >= (len(self.agents)-1):
                break

        return None
        
    # Loop for one player auction    
    def single_player(self, player: Player): 
        self.current_player = player
        self.current_price = 0
        self.highest_bidder = None
        

        
        if self.verbose:
            logger.info(f"\n🏆 AUCTION START: {player.name} ({player.role}) - Evaluation: {player.evaluation}")
        

        while True:

            self.loop_automatic_agents()

            # Control for manual bidding agent
            while True:
                agent_id = input("Chi vuole offrire? ")
                # Check if input is either "nessuno" or a valid agent ID
                valid_agent_ids = [agent.agent_id for agent in self.agents]
                if agent_id == "nessuno" or agent_id in valid_agent_ids:
                    break
                else: 
                    print("ID non valido. Riprova.")

            if agent_id == "nessuno":
                break

            # Get the agent object for validation
            bidding_agent = next(agent for agent in self.agents if agent.agent_id == agent_id)
            
            # Control for manual offer
            while True:
                try:
                    offerta = int(input("Quanto offre? "))
                    # Check if the offer is valid using the modified can_participate_in_bid function
                    if self.can_participate_in_bid(agent=bidding_agent, offer_price=offerta, 
                                                 position=player.role, slots=self.slots):
                        break
                    else:
                        print(f"Offerta non valida. Deve essere > {self.current_price} e l'agente deve poterla permettere.")
                        print(f"Crediti disponibili: {bidding_agent.current_credits}")
                except ValueError:
                    print("Inserire un numero valido per l'offerta.")

            self.current_price = offerta
            self.highest_bidder = agent_id


        
        # Finalize sale
        if self.highest_bidder:
            player.fantasy_team = self.highest_bidder
            player.final_cost = self.current_price
            
            winner_agent = next(agent for agent in self.agents if agent.agent_id == self.highest_bidder)
            winner_agent._squad.append(player)
            winner_agent.current_credits -= self.current_price
            
            if self.verbose:
                logger.info(f"✅ {player.name} SOLD to {self.highest_bidder} for {self.current_price}")
            return self.highest_bidder
        else:
            player.fantasy_team = "UNSOLD"
            player.final_cost = 0
            if self.verbose:
                logger.info(f"❌ {player.name} UNSOLD")
            return None
        
    def run_all(self, auction_type: str = "random", per_ruolo: bool = True, verbose: bool = False):
        """Run auction for all players"""
        self.verbose = verbose

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
                if verbose:
                    logger.info(f"\n🏈 STARTING {role} AUCTIONS")
                    
                while True:
                    all_full = all(len(getattr(agent.squad, role.lower() if role != "DEF" else "def_")) >= self.slots[role] 
                                 for agent in self.agents)
                    if all_full:
                        break
                        
                    next_player = next((p for p in players_to_auction if p.role == role and p.fantasy_team is None), None)
                    if next_player is None:
                        break
                    self.single_player(next_player)
        else:
            total_slots = sum(self.slots.values())
            while True:
                all_full = all(len(agent.squad) >= total_slots for agent in self.agents)
                if all_full:
                    break
                    
                next_player = next((p for p in players_to_auction if p.fantasy_team is None), None)
                if next_player is None:
                    break
                self.single_player(next_player)

    def start_player_auction(self, player: Player):
        """Start auction for a specific player (API version)"""
        if player.fantasy_team and player.fantasy_team != "UNSOLD":
            return {'success': False, 'error': f'Player {player.name} is already assigned'}
            
        self.current_player = player
        self.current_price = 0
        self.highest_bidder = None
        
        if self.verbose:
            logger.info(f"\n🏆 AUCTION START: {player.name} ({player.role}) - Evaluation: {player.evaluation}")
        
        return {
            'success': True,
            'message': f'Auction started for {player.name}',
            'player': {
                'name': player.name,
                'role': player.role,
                'team': player.team,
                'evaluation': player.evaluation
            }
        }
