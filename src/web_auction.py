"""
Web Auction module - contains utilities for web-based auction management
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

from .models import Slots, Player
from .agents.agent_class import RandomAgent
from .agents.cap_based_agent import CapAgent
from .agents.dynamic_cap_based_agent import DynamicCapAgent
from .agents.rl_deep_agent import RLDeepAgent
from .agents.human_agent import HumanAgent
from .auction import Auction
from .web_interactive_auction import WebInteractiveAuction
from .data_loader import load_players_from_excel


class WebAuctionManager:
    """Manager for web-based auctions"""
    
    def __init__(self):
        self.active_auctions: Dict[str, WebInteractiveAuction] = {}
        self.auction_configs: Dict[str, dict] = {}
        self.auction_logs: Dict[str, List[dict]] = {}
    
    def create_auction(self, config: dict) -> str:
        """Create a new auction and return its ID"""
        auction_id = str(uuid.uuid4())
        
        # Extract configuration with defaults
        initial_credits = config.get('initial_credits', 1000)
        
        # Slots configuration
        slots_config = config.get('slots', {})
        slots = Slots(
            gk=slots_config.get('gk', 3),
            def_=slots_config.get('def', 8),
            mid=slots_config.get('mid', 8),
            att=slots_config.get('att', 6)
        )
        
        # Create agents
        agent_configs = config.get('agents', self._get_default_agents())
        agents = self._create_agents(agent_configs)
        
        # Shuffle agents if requested
        if config.get('shuffle_agents', True):
            import random
            random.shuffle(agents)
        
        # Load players
        listone = load_players_from_excel()
        
        # Create auction
        auction = WebInteractiveAuction(slots, agents, listone, initial_credits)
        
        # Setup log callback
        def log_callback(log_entry):
            if auction_id not in self.auction_logs:
                self.auction_logs[auction_id] = []
            self.auction_logs[auction_id].append(log_entry)
        
        auction.set_log_callback(log_callback)
        
        # Store auction and config
        self.active_auctions[auction_id] = auction
        self.auction_configs[auction_id] = {
            'initial_credits': initial_credits,
            'slots': slots.to_dict(),
            'agents': [{'id': agent.agent_id, 'type': type(agent).__name__} for agent in agents],
            'auction_type': config.get('auction_type', 'chiamata'),
            'per_ruolo': config.get('per_ruolo', True),
            'players_count': len(listone),
            'created_at': datetime.now().isoformat(),
            'status': 'created'
        }
        self.auction_logs[auction_id] = []
        
        return auction_id
    
    def start_auction(self, auction_id: str, run_config: dict = None) -> dict:
        """Start an auction and return results"""
        if auction_id not in self.active_auctions:
            raise ValueError(f"Auction {auction_id} not found")

        auction = self.active_auctions[auction_id]
        config = self.auction_configs[auction_id]
        
        # Get auction parameters
        run_config = run_config or {}
        auction_type = run_config.get('auction_type', config.get('auction_type', 'chiamata'))
        per_ruolo = run_config.get('per_ruolo', config.get('per_ruolo', True))
        
        try:
            # Start the auction in background
            auction.start_auction(auction_type=auction_type, per_ruolo=per_ruolo)
            
            # Update status
            config['status'] = 'running'
            config['started_at'] = datetime.now().isoformat()
            
            return {
                'auction_id': auction_id,
                'status': 'running',
                'message': 'Auction started successfully'
            }
            
        except Exception as e:
            config['status'] = 'error'
            config['error'] = str(e)
            raise
    
    def get_auction_status(self, auction_id: str) -> dict:
        """Get auction status and results"""
        if auction_id not in self.active_auctions:
            raise ValueError(f"Auction {auction_id} not found")
        
        auction = self.active_auctions[auction_id]
        config = self.auction_configs[auction_id]
        
        result = {
            'auction_id': auction_id,
            'config': config,
            'logs': self.auction_logs.get(auction_id, []),
            'current_state': auction.get_current_state()
        }
        
        # Add results if auction is completed
        if auction.state.value == 'completed':
            result['results'] = self._get_auction_results(auction)
            config['status'] = 'completed'
        elif auction.state.value == 'error':
            config['status'] = 'error'
        elif auction.state.value == 'running':
            config['status'] = 'running'
        
        return result

    def submit_human_bid(self, auction_id: str, agent_id: str, bid_amount: int) -> dict:
        """Submit a bid from a human agent"""
        if auction_id not in self.active_auctions:
            raise ValueError(f"Auction {auction_id} not found")
        
        auction = self.active_auctions[auction_id]
        return auction.submit_human_bid(agent_id, bid_amount)

    def submit_human_pass(self, auction_id: str) -> dict:
        """Submit a pass (no bid) from human agents"""
        if auction_id not in self.active_auctions:
            raise ValueError(f"Auction {auction_id} not found")
        
        auction = self.active_auctions[auction_id]
        return auction.submit_human_pass()

    def submit_human_pass(self, auction_id: str) -> dict:
        """Submit a pass (no bid) from human agents"""
        if auction_id not in self.active_auctions:
            raise ValueError(f"Auction {auction_id} not found")
        
        auction = self.active_auctions[auction_id]
        return auction.submit_human_pass()

    def get_auction_live_state(self, auction_id: str) -> dict:
        """Get live auction state for real-time updates"""
        if auction_id not in self.active_auctions:
            raise ValueError(f"Auction {auction_id} not found")
        
        auction = self.active_auctions[auction_id]
        return auction.get_current_state()

    def stop_auction(self, auction_id: str) -> dict:
        """Stop a running auction"""
        if auction_id not in self.active_auctions:
            raise ValueError(f"Auction {auction_id} not found")
        
        auction = self.active_auctions[auction_id]
        auction.stop_auction()
        
        config = self.auction_configs[auction_id]
        config['status'] = 'stopped'
        config['stopped_at'] = datetime.now().isoformat()
        
        return {
            'auction_id': auction_id,
            'status': 'stopped',
            'message': 'Auction stopped successfully'
        }
    
    def list_auctions(self) -> List[dict]:
        """List all auctions"""
        auctions_list = []
        for auction_id, config in self.auction_configs.items():
            auctions_list.append({
                'auction_id': auction_id,
                'created_at': config.get('created_at'),
                'status': config.get('status', 'created'),
                'agents_count': len(config.get('agents', [])),
                'players_count': config.get('players_count', 0)
            })
        
        return auctions_list
    
    def delete_auction(self, auction_id: str) -> bool:
        """Delete an auction"""
        if auction_id in self.active_auctions:
            del self.active_auctions[auction_id]
        if auction_id in self.auction_configs:
            del self.auction_configs[auction_id]
        if auction_id in self.auction_logs:
            del self.auction_logs[auction_id]
        return True
    
    def _get_default_agents(self) -> List[dict]:
        """Get default agent configuration from single_auction.py"""
        return [
            {'type': 'human', 'id': 'cucco'},
            {'type': 'human', 'id': 'andrea'},
            {'type': 'cap', 'id': 'noi', 'strategy': 'conservative'},
            {'type': 'human', 'id': 'gg pace'},
            {'type': 'human', 'id': 'davide'},
            {'type': 'human', 'id': 'tommi'},
            {'type': 'human', 'id': 'cantiello'},
            {'type': 'human', 'id': 'miky'}
        ]
    
    def _create_agents(self, agent_configs: List[dict]) -> List:
        """Create agents from configuration"""
        agents = []
        
        for config in agent_configs:
            agent_type = config.get('type', 'human')
            agent_id = config.get('id', f'agent_{len(agents)}')
            agent_name = config.get('name', f'Agent {len(agents) + 1}')
            
            if agent_type == 'human':
                agents.append(HumanAgent(agent_id=agent_id, name=agent_name))
            elif agent_type == 'random':
                agents.append(RandomAgent(agent_id=agent_id, name=agent_name))
            elif agent_type == 'cap':
                strategy = config.get('strategy', 'conservative')
                agents.append(CapAgent(agent_id=agent_id, name=agent_name, cap_strategy=strategy))
            elif agent_type == 'dynamic_cap':
                agents.append(DynamicCapAgent(agent_id=agent_id, name=agent_name))
            elif agent_type == 'rl_deep':
                agents.append(RLDeepAgent(agent_id=agent_id, name=agent_name))
            else:
                # Default to human agent
                agents.append(HumanAgent(agent_id=agent_id, name=agent_name))
        
        return agents
    
    def _setup_auction_logging(self, auction_id: str):
        """Setup logging for a specific auction"""
        # Create a simple log handler that stores logs in memory
        class MemoryLogHandler(logging.Handler):
            def __init__(self):
                super().__init__()
                self.logs = []
                
            def emit(self, record):
                log_entry = self.format(record)
                self.logs.append({
                    'timestamp': datetime.now().isoformat(),
                    'message': log_entry
                })
        
        logger = logging.getLogger(f'auction_{auction_id}')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add memory handler
        handler = MemoryLogHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return handler
    
    def _get_auction_results(self, auction: Auction) -> dict:
        """Extract results from completed auction"""
        results = {
            'teams': [],
            'summary': {}
        }
        
        total_spent = 0
        total_credits = 0
        
        for agent in auction.agents:
            # Calculate squad metrics
            total_eval = agent.squad.objective(standardized=False)
            total_std_eval = agent.squad.objective(standardized=True)
            bestxi_eval = agent.squad.objective(bestxi=True, standardized=False)
            bestxi_std_eval = agent.squad.objective(bestxi=True, standardized=True)
            
            credits_spent = auction.initial_credits - agent.current_credits
            total_spent += credits_spent
            total_credits += agent.current_credits
            
            team_data = {
                'agent_id': agent.agent_id,
                'agent_type': type(agent).__name__,
                'players': [
                    {
                        'name': player.name,
                        'team': player.team,
                        'role': player.role,
                        'evaluation': player.evaluation,
                        'standardized_evaluation': round(player.standardized_evaluation, 3),
                        'final_cost': player.final_cost,
                        'ranking': player.ranking
                    }
                    for player in agent.squad
                ],
                'metrics': {
                    'total_evaluation': total_eval,
                    'total_standardized_evaluation': round(total_std_eval, 3),
                    'bestxi_evaluation': bestxi_eval,
                    'bestxi_standardized_evaluation': round(bestxi_std_eval, 3),
                    'credits_remaining': agent.current_credits,
                    'credits_spent': credits_spent,
                    'players_count': len(agent.squad),
                    'squad_slots': {
                        'gk': len(agent.squad.gk),
                        'def': len(agent.squad.def_),
                        'mid': len(agent.squad.mid),
                        'att': len(agent.squad.att)
                    }
                }
            }
            
            results['teams'].append(team_data)
        
        # Sort teams by best XI standardized evaluation
        results['teams'].sort(
            key=lambda t: t['metrics']['bestxi_standardized_evaluation'], 
            reverse=True
        )
        
        # Add summary
        results['summary'] = {
            'total_agents': len(auction.agents),
            'total_credits_spent': total_spent,
            'total_credits_remaining': total_credits,
            'average_credits_spent': round(total_spent / len(auction.agents), 2) if auction.agents else 0,
            'winner': results['teams'][0]['agent_id'] if results['teams'] else None
        }
        
        return results


# Singleton instance for the web application
web_auction_manager = WebAuctionManager()
