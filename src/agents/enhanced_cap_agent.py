"""
Enhanced Cap Based agent with improved strategy and real-time adaptations
"""

from src.agents.agent_class import Agent
from src.models import Player, Squad, Slots
from typing import List, Dict
import logging


class EnhancedCapAgent(Agent):
    """Enhanced Cap Based Agent with adaptive strategy and improved decision making"""
    
    def __init__(self, agent_id: str, cap_strategy: str = "adaptive", bestxi_budget: float = 0.7,
                 aggression_level: float = 0.5, adaptability: bool = True):
        super().__init__(agent_id)
        self.cap_strategy = cap_strategy
        self.bestxi_budget = bestxi_budget
        self.aggression_level = aggression_level  # 0.0 = very conservative, 1.0 = very aggressive
        self.adaptability = adaptability
        
        # Strategy data
        self.players_caps = {}
        self.position_priorities = {"GK": 1.0, "DEF": 1.0, "MID": 1.2, "ATT": 1.3}
        self.market_history = []  # Track market prices for adaptation
        self.target_players = {}  # Players we really want
        
        # Real-time tracking
        self.auction_round = 0
        self.players_sold_count = 0
        self.avg_market_inflation = 1.0
        
    def initialize(self, listone: List[Player], slots: Slots, initial_credits: int, num_participants: int):
        """Initialize agent with enhanced market analysis"""
        super().initialize(listone, slots, initial_credits, num_participants)
        
        if self.cap_strategy == "adaptive":
            self._calculate_adaptive_caps(listone, slots, initial_credits, num_participants)
        elif self.cap_strategy == "bestxi_focused":
            self._calculate_bestxi_caps(listone, slots, initial_credits, num_participants)
        elif self.cap_strategy == "value_hunting":
            self._calculate_value_caps(listone, slots, initial_credits, num_participants)
        else:
            # Fallback to original logic
            self._calculate_original_caps(listone, slots, initial_credits, num_participants)
        
        # Identify target players
        self._identify_target_players(listone)
        
        logging.info(f"🤖 {self.agent_id} initialized with {len(self.players_caps)} player caps")
    
    def _calculate_adaptive_caps(self, listone: List[Player], slots: Slots, 
                                initial_credits: int, num_participants: int):
        """Calculate caps with adaptive strategy based on current squad needs"""
        total_budget = initial_credits
        
        # Analyze position requirements
        position_needs = {
            "GK": slots.gk,
            "DEF": slots.def_,
            "MID": slots.mid,
            "ATT": slots.att
        }
        
        # Allocate budget by position with priorities
        position_budgets = {}
        total_priority = sum(position_needs[pos] * self.position_priorities[pos] for pos in position_needs)
        
        for position in position_needs:
            position_weight = (position_needs[position] * self.position_priorities[position]) / total_priority
            position_budgets[position] = total_budget * position_weight
        
        # Calculate caps for each position
        for position in ["GK", "DEF", "MID", "ATT"]:
            position_players = sorted(
                [p for p in listone if p.role == position],
                key=lambda p: p.evaluation, reverse=True
            )
            
            if not position_players:
                continue
            
            # Target top players for this position
            target_count = min(position_needs[position] * num_participants * 2, len(position_players))
            target_players = position_players[:target_count]
            
            # Calculate value per point for this position
            total_evaluation = sum(p.evaluation for p in target_players)
            if total_evaluation > 0:
                budget_for_position = position_budgets[position] * num_participants
                value_per_point = budget_for_position / total_evaluation
                
                # Apply caps with some variance based on player ranking
                for i, player in enumerate(target_players):
                    # Premium for top players, discount for lower-tier
                    tier_multiplier = self._get_tier_multiplier(i, len(target_players))
                    cap = player.evaluation * value_per_point * tier_multiplier
                    
                    # Apply aggression factor
                    cap *= (1 + self.aggression_level * 0.3)
                    
                    self.players_caps[player.name] = max(1, int(cap))
    
    def _calculate_bestxi_caps(self, listone: List[Player], slots: Slots,
                              initial_credits: int, num_participants: int):
        """Focus budget on best XI players"""
        best_xi_budget = initial_credits * self.bestxi_budget
        remaining_budget = initial_credits * (1 - self.bestxi_budget)
        
        # Best XI: 1 GK, 3 DEF, 3 MID, 3 ATT
        best_xi_needs = {"GK": 1, "DEF": 3, "MID": 3, "ATT": 3}
        
        # Calculate caps for best XI players
        for position in best_xi_needs:
            position_players = sorted(
                [p for p in listone if p.role == position],
                key=lambda p: p.evaluation, reverse=True
            )
            
            target_count = best_xi_needs[position] * num_participants
            if target_count <= len(position_players):
                best_players = position_players[:target_count]
                
                # Allocate higher budget for these players
                position_budget = (best_xi_budget * best_xi_needs[position] / 10) * num_participants
                total_evaluation = sum(p.evaluation for p in best_players)
                
                if total_evaluation > 0:
                    value_per_point = position_budget / total_evaluation
                    for player in best_players:
                        cap = player.evaluation * value_per_point * 1.5  # Premium for best XI
                        self.players_caps[player.name] = max(1, int(cap))
        
        # Calculate caps for remaining players (bench)
        for position in ["GK", "DEF", "MID", "ATT"]:
            position_players = sorted(
                [p for p in listone if p.role == position],
                key=lambda p: p.evaluation, reverse=True
            )
            
            start_idx = best_xi_needs.get(position, 0) * num_participants
            end_idx = getattr(slots, position.lower() if position != "DEF" else "def_") * num_participants
            
            if start_idx < len(position_players):
                bench_players = position_players[start_idx:end_idx]
                
                if bench_players:
                    position_budget = (remaining_budget * 0.25) * num_participants
                    total_evaluation = sum(p.evaluation for p in bench_players)
                    
                    if total_evaluation > 0:
                        value_per_point = position_budget / total_evaluation
                        for player in bench_players:
                            cap = player.evaluation * value_per_point
                            self.players_caps[player.name] = max(1, int(cap))
    
    def _calculate_value_caps(self, listone: List[Player], slots: Slots,
                             initial_credits: int, num_participants: int):
        """Focus on value players with good evaluation/price ratio"""
        # Target middle-tier players with good value
        for position in ["GK", "DEF", "MID", "ATT"]:
            position_players = sorted(
                [p for p in listone if p.role == position],
                key=lambda p: p.evaluation, reverse=True
            )
            
            # Focus on players ranked 20-80% (avoid most expensive and cheapest)
            start_idx = max(1, len(position_players) // 5)
            end_idx = min(len(position_players), len(position_players) * 4 // 5)
            
            value_players = position_players[start_idx:end_idx]
            
            if value_players:
                # Allocate moderate budget for good value
                position_budget = (initial_credits * 0.8 / 4) * num_participants  # 20% each position
                total_evaluation = sum(p.evaluation for p in value_players)
                
                if total_evaluation > 0:
                    value_per_point = position_budget / total_evaluation
                    for player in value_players:
                        cap = player.evaluation * value_per_point * 1.2  # Slight premium for value
                        self.players_caps[player.name] = max(1, int(cap))
    
    def _calculate_original_caps(self, listone: List[Player], slots: Slots,
                                initial_credits: int, num_participants: int):
        """Original cap calculation as fallback"""
        # This would contain the original CapAgent logic
        # For brevity, implementing a simplified version
        avg_budget_per_team = initial_credits * 0.8
        
        for position in ["GK", "DEF", "MID", "ATT"]:
            position_players = sorted(
                [p for p in listone if p.role == position],
                key=lambda p: p.evaluation, reverse=True
            )
            
            # Take top players for each position
            slots_for_position = getattr(slots, position.lower() if position != "DEF" else "def_")
            target_players = position_players[:slots_for_position * num_participants]
            
            if target_players:
                total_evaluation = sum(p.evaluation for p in target_players)
                if total_evaluation > 0:
                    value_per_point = avg_budget_per_team / (total_evaluation / len(target_players))
                    for player in target_players:
                        cap = player.evaluation * value_per_point / len(target_players)
                        self.players_caps[player.name] = max(1, int(cap))
    
    def _get_tier_multiplier(self, rank: int, total_players: int) -> float:
        """Get multiplier based on player tier"""
        if rank < total_players * 0.1:  # Top 10%
            return 1.4
        elif rank < total_players * 0.3:  # Top 30%
            return 1.2
        elif rank < total_players * 0.7:  # Top 70%
            return 1.0
        else:  # Bottom 30%
            return 0.8
    
    def _identify_target_players(self, listone: List[Player]):
        """Identify players we really want to target"""
        for position in ["GK", "DEF", "MID", "ATT"]:
            position_players = sorted(
                [p for p in listone if p.role == position],
                key=lambda p: p.evaluation, reverse=True
            )
            
            # Mark top 2-3 players per position as high priority
            high_priority_count = min(3, len(position_players) // 3)
            for player in position_players[:high_priority_count]:
                self.target_players[player.name] = "high"
            
            # Mark next tier as medium priority
            medium_priority_count = min(5, len(position_players) // 2)
            for player in position_players[high_priority_count:high_priority_count + medium_priority_count]:
                self.target_players[player.name] = "medium"
    
    def make_offer_decision(self, current_player, current_price, highest_bidder, 
                          player_list, other_agents) -> str:
        """Enhanced bidding decision with real-time adaptation"""
        self.auction_round += 1
        
        # Check if we have a cap for this player
        if current_player.name not in self.players_caps:
            return "no_offer"
        
        base_cap = self.players_caps[current_player.name]
        
        # Apply real-time adjustments
        adjusted_cap = self._adjust_cap_for_market_conditions(
            current_player, base_cap, current_price, other_agents
        )
        
        # Check if we should bid
        if current_price >= adjusted_cap:
            return "no_offer"
        
        # Determine bid strategy
        return self._determine_bid_strategy(
            current_player, current_price, adjusted_cap, highest_bidder, other_agents
        )
    
    def _adjust_cap_for_market_conditions(self, player: Player, base_cap: int, 
                                        current_price: int, other_agents: List[Agent]) -> int:
        """Adjust cap based on current market conditions"""
        adjusted_cap = base_cap
        
        # Market inflation adjustment
        if self.adaptability and len(self.market_history) > 5:
            recent_inflation = self._calculate_recent_inflation()
            adjusted_cap *= recent_inflation
        
        # Squad needs adjustment
        squad_needs_multiplier = self._calculate_squad_needs_multiplier(player)
        adjusted_cap *= squad_needs_multiplier
        
        # Competition level adjustment
        competition_factor = self._assess_competition_level(other_agents, current_price)
        adjusted_cap *= competition_factor
        
        # Target player bonus
        if player.name in self.target_players:
            if self.target_players[player.name] == "high":
                adjusted_cap *= 1.3
            elif self.target_players[player.name] == "medium":
                adjusted_cap *= 1.15
        
        # Budget scarcity adjustment (if running low on credits)
        budget_factor = self._calculate_budget_scarcity_factor(player)
        adjusted_cap *= budget_factor
        
        return max(current_price + 1, int(adjusted_cap))
    
    def _calculate_recent_inflation(self) -> float:
        """Calculate recent market inflation"""
        if len(self.market_history) < 5:
            return 1.0
        
        recent_sales = self.market_history[-10:]  # Last 10 sales
        inflation_sum = sum(sale['price_ratio'] for sale in recent_sales)
        return min(1.5, max(0.7, inflation_sum / len(recent_sales)))
    
    def _calculate_squad_needs_multiplier(self, player: Player) -> float:
        """Calculate multiplier based on squad needs"""
        position_map = {
            "GK": len(self.squad.gk),
            "DEF": len(self.squad.def_),
            "MID": len(self.squad.mid),
            "ATT": len(self.squad.att)
        }
        
        slots_map = {
            "GK": self.slots.gk,
            "DEF": self.slots.def_,
            "MID": self.slots.mid,
            "ATT": self.slots.att
        }
        
        current_count = position_map[player.role]
        max_slots = slots_map[player.role]
        
        # Increase willingness to pay if we have few players in this position
        if current_count == 0:
            return 1.4  # Really need this position
        elif current_count < max_slots // 2:
            return 1.2  # Still need more
        elif current_count < max_slots:
            return 1.0  # Normal need
        else:
            return 0.3  # Position is full, very low interest
    
    def _assess_competition_level(self, other_agents: List[Agent], current_price: int) -> float:
        """Assess competition level and adjust accordingly"""
        # Count agents who might still bid (have credits and need this position)
        potential_bidders = 0
        for agent in other_agents:
            if (hasattr(agent, 'current_credits') and 
                agent.current_credits > current_price + 1):
                potential_bidders += 1
        
        # Adjust based on competition
        if potential_bidders >= 3:
            return 0.9  # High competition, be more conservative
        elif potential_bidders == 2:
            return 1.0  # Normal competition
        elif potential_bidders == 1:
            return 1.1  # Low competition, can be more aggressive
        else:
            return 1.2  # No competition, very aggressive
    
    def _calculate_budget_scarcity_factor(self, player: Player) -> float:
        """Adjust bidding based on remaining budget"""
        total_slots = sum([self.slots.gk, self.slots.def_, self.slots.mid, self.slots.att])
        current_players = len(self.squad)
        remaining_slots = total_slots - current_players - 1  # -1 for current player
        
        if remaining_slots <= 0:
            return 0.0  # Squad is full
        
        budget_per_remaining_slot = self.current_credits / remaining_slots if remaining_slots > 0 else 0
        
        # If we have plenty of budget relative to remaining needs, be more generous
        if budget_per_remaining_slot > 50:
            return 1.2
        elif budget_per_remaining_slot > 20:
            return 1.0
        elif budget_per_remaining_slot > 10:
            return 0.8
        else:
            return 0.6  # Very tight budget
    
    def _determine_bid_strategy(self, player: Player, current_price: int, 
                              adjusted_cap: int, highest_bidder, other_agents: List[Agent]) -> str:
        """Determine the specific bid strategy"""
        
        # If we're the highest bidder, sometimes wait
        if highest_bidder and highest_bidder.agent_id == self.agent_id:
            return "no_offer"  # Let others bid first
        
        # Calculate our maximum reasonable bid
        max_reasonable_bid = min(adjusted_cap, self.current_credits - 1)
        
        if current_price + 1 > max_reasonable_bid:
            return "no_offer"
        
        # Different strategies based on aggression level
        if self.aggression_level > 0.7:
            # Aggressive: bid up to cap quickly
            aggressive_bid = min(max_reasonable_bid, current_price + 5)
            if aggressive_bid > current_price:
                return f"offer_{aggressive_bid}"
        
        elif self.aggression_level < 0.3:
            # Conservative: only minimal bids
            return "offer_+1"
        
        else:
            # Balanced: bid strategically
            if player.name in self.target_players and self.target_players[player.name] == "high":
                # More aggressive for target players
                target_bid = min(max_reasonable_bid, current_price + 3)
                if target_bid > current_price:
                    return f"offer_{target_bid}"
            else:
                return "offer_+1"
        
        return "offer_+1"
    
    def update_market_history(self, player_name: str, final_price: int, evaluation: int):
        """Update market history for learning"""
        if player_name in self.players_caps:
            expected_price = self.players_caps[player_name]
            price_ratio = final_price / expected_price if expected_price > 0 else 1.0
            
            self.market_history.append({
                'player': player_name,
                'final_price': final_price,
                'expected_price': expected_price,
                'price_ratio': price_ratio,
                'evaluation': evaluation
            })
            
            # Keep only recent history
            if len(self.market_history) > 50:
                self.market_history = self.market_history[-50:]
        
        self.players_sold_count += 1
    
    def get_strategy_summary(self) -> Dict:
        """Get summary of current strategy and performance"""
        return {
            "strategy": self.cap_strategy,
            "aggression_level": self.aggression_level,
            "players_with_caps": len(self.players_caps),
            "target_players": len(self.target_players),
            "market_samples": len(self.market_history),
            "current_credits": self.current_credits,
            "squad_size": len(self.squad),
            "avg_inflation": self.avg_market_inflation
        }
