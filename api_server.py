"""
API Server for Fantasy Football Auction
Espone le API per collegare la logica dell'asta a un'interfaccia web
"""

from flask import Flask, jsonify, request, send_from_directory, render_template_string
from flask_cors import CORS
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional
import uuid

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.models import Slots, Player
from src.agents.agent_class import RandomAgent
from src.agents.cap_based_agent import CapAgent
from src.agents.dynamic_cap_based_agent import DynamicCapAgent
from src.agents.rl_deep_agent import RLDeepAgent
from src.agents.human_agent import HumanAgent
from src.auction import Auction
from src.data_loader import load_players_from_excel

app = Flask(__name__)
CORS(app)  # Enable CORS for web UI

# Configure static file serving
app.static_folder = 'frontend'
app.static_url_path = '/static'

# Global auction state
current_auction: Optional[Auction] = None
auction_sessions: Dict[str, dict] = {}

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AuctionAPI:
    """Wrapper class to manage auction state and provide API methods"""
    
    def __init__(self):
        self.auction = None
        self.session_id = None
        self.current_player = None
        self.auction_state = "not_started"  # not_started, running, player_auction, completed
        
    def create_auction(self, agents_config: List[dict], auction_config: dict):
        """Create a new auction with given configuration"""
        try:
            # Create agents
            agents = []
            for agent_config in agents_config:
                agent_type = agent_config.get('type', 'human')
                agent_id = agent_config.get('id')
                
                if agent_type == 'human':
                    agents.append(HumanAgent(agent_id=agent_id))
                elif agent_type == 'cap':
                    agents.append(CapAgent(agent_id=agent_id))
                elif agent_type == 'dynamic_cap':
                    agents.append(DynamicCapAgent(agent_id=agent_id))
                elif agent_type == 'random':
                    agents.append(RandomAgent(agent_id=agent_id))
                elif agent_type == 'rl_deep':
                    agents.append(RLDeepAgent(agent_id=agent_id))
                    
            # Create slots
            slots = Slots(
                gk=auction_config.get('slots_gk', 3),
                def_=auction_config.get('slots_def', 8),
                mid=auction_config.get('slots_mid', 8),
                att=auction_config.get('slots_att', 6)
            )
            
            # Load players
            players = load_players_from_excel()
            
            # Create auction
            initial_credits = auction_config.get('initial_credits', 1000)
            self.auction = Auction(slots, agents, players, initial_credits)
            self.session_id = str(uuid.uuid4())
            self.auction_state = "created"
            
            return {
                'success': True,
                'session_id': self.session_id,
                'message': 'Auction created successfully'
            }
            
        except Exception as e:
            logger.error(f"Error creating auction: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_auction_status(self):
        """Get current auction status"""
        if not self.auction:
            return {
                'state': 'not_started',
                'current_player': None,
                'agents': [],
                'message': 'No auction created'
            }
            
        # Get agents status
        agents_status = []
        for agent in self.auction.agents:
            agents_status.append({
                'id': agent.agent_id,
                'type': type(agent).__name__,
                'credits': agent.current_credits,
                'squad_size': len(agent.squad),
                'squad_gk': len(agent.squad.gk),
                'squad_def': len(agent.squad.def_),
                'squad_mid': len(agent.squad.mid),
                'squad_att': len(agent.squad.att)
            })
            
        current_player_info = None
        if hasattr(self.auction, 'current_player') and self.auction.current_player:
            current_player_info = {
                'name': self.auction.current_player.name,
                'role': self.auction.current_player.role,
                'team': self.auction.current_player.team,
                'evaluation': self.auction.current_player.evaluation,
                'current_price': getattr(self.auction, 'current_price', 0),
                'highest_bidder': getattr(self.auction, 'highest_bidder', None)
            }
            
        return {
            'state': self.auction_state,
            'session_id': self.session_id,
            'current_player': current_player_info,
            'agents': agents_status,
            'slots': self.auction.slots
        }
    
    def start_next_player_auction(self, role_filter: Optional[str] = None):
        """Start auction for next available player"""
        if not self.auction:
            return {'success': False, 'error': 'No auction created'}
            
        try:
            # Find next player to auction
            next_player = None
            for player in self.auction.players:
                if player.fantasy_team is None:  # Not yet sold
                    if role_filter is None or player.role == role_filter:
                        next_player = player
                        break
                        
            if not next_player:
                self.auction_state = "completed"
                return {
                    'success': True,
                    'completed': True,
                    'message': 'All players have been auctioned'
                }
                
            # Set up player auction
            self.auction.current_player = next_player
            self.auction.current_price = 0
            self.auction.highest_bidder = None
            self.current_player = next_player
            self.auction_state = "player_auction"
            
            return {
                'success': True,
                'player': {
                    'name': next_player.name,
                    'role': next_player.role,
                    'team': next_player.team,
                    'evaluation': next_player.evaluation
                }
            }
            
        except Exception as e:
            logger.error(f"Error starting player auction: {e}")
            return {'success': False, 'error': str(e)}
    
    def make_bid(self, agent_id: str, amount: int):
        """Make a bid for current player"""
        if not self.auction or not self.current_player:
            return {'success': False, 'error': 'No active player auction'}
            
        try:
            # Find the bidding agent
            bidding_agent = None
            for agent in self.auction.agents:
                if agent.agent_id == agent_id:
                    bidding_agent = agent
                    break
                    
            if not bidding_agent:
                return {'success': False, 'error': 'Agent not found'}
                
            # Validate bid
            if not self.auction.can_participate_in_bid(
                agent=bidding_agent, 
                offer_price=amount, 
                position=self.current_player.role, 
                slots=self.auction.slots
            ):
                return {
                    'success': False, 
                    'error': 'Invalid bid - insufficient credits or no available slots'
                }
                
            # Accept bid
            self.auction.current_price = amount
            self.auction.highest_bidder = agent_id
            
            return {
                'success': True,
                'current_price': amount,
                'highest_bidder': agent_id
            }
            
        except Exception as e:
            logger.error(f"Error making bid: {e}")
            return {'success': False, 'error': str(e)}
    
    def process_bot_bids(self):
        """Process automatic bids from bot agents"""
        if not self.auction or not self.current_player:
            return {'success': False, 'error': 'No active player auction'}
        
        try:
            bot_agents = [agent for agent in self.auction.agents if not isinstance(agent, HumanAgent)]
            bid_results = []
            
            for agent in bot_agents:
                # Check if agent can participate in this bid
                if self.auction.can_participate_in_bid(
                    agent=agent, 
                    offer_price=self.auction.current_price + 1, 
                    position=self.current_player.role, 
                    slots=self.auction.slots
                ):
                    # Get all other agents except the current one
                    other_agents = [a for a in self.auction.agents if a.agent_id != agent.agent_id]
                    
                    # Get current agent's decision
                    decision = agent.make_offer_decision(
                        current_player=self.current_player,
                        current_price=self.auction.current_price,
                        highest_bidder=self.auction.highest_bidder,
                        player_list=self.auction.players,
                        other_agents=other_agents
                    )
                    
                    if decision == "offer_+1":
                        self.auction.current_price += 1
                        self.auction.highest_bidder = agent.agent_id
                        bid_results.append({
                            'agent_id': agent.agent_id,
                            'action': 'bid',
                            'amount': self.auction.current_price
                        })
                    else:
                        bid_results.append({
                            'agent_id': agent.agent_id,
                            'action': 'pass'
                        })
                else:
                    bid_results.append({
                        'agent_id': agent.agent_id,
                        'action': 'cannot_bid',
                        'reason': 'insufficient_credits_or_slots'
                    })
            
            return {
                'success': True,
                'bids': bid_results,
                'current_price': self.auction.current_price,
                'highest_bidder': self.auction.highest_bidder
            }
            
        except Exception as e:
            logger.error(f"Error processing bot bids: {e}")
            return {'success': False, 'error': str(e)}

    def finalize_player_auction(self):
        """Finalize current player auction"""
        if not self.auction or not self.current_player:
            return {'success': False, 'error': 'No active player auction'}
            
        try:
            player = self.current_player
            
            if self.auction.highest_bidder:
                # Sale successful
                player.fantasy_team = self.auction.highest_bidder
                player.final_cost = self.auction.current_price
                
                # Find winner agent and update their squad/credits
                winner_agent = None
                for agent in self.auction.agents:
                    if agent.agent_id == self.auction.highest_bidder:
                        winner_agent = agent
                        break
                        
                if winner_agent:
                    winner_agent._squad.append(player)
                    winner_agent.current_credits -= self.auction.current_price
                    
                result = {
                    'success': True,
                    'sold': True,
                    'buyer': self.auction.highest_bidder,
                    'price': self.auction.current_price,
                    'player': player.name
                }
            else:
                # No bids - player unsold
                player.fantasy_team = "UNSOLD"
                player.final_cost = 0
                result = {
                    'success': True,
                    'sold': False,
                    'player': player.name
                }
                
            # Reset auction state
            self.current_player = None
            self.auction.current_player = None
            self.auction_state = "running"
            
            return result
            
        except Exception as e:
            logger.error(f"Error finalizing auction: {e}")
            return {'success': False, 'error': str(e)}


# Global API instance
auction_api = AuctionAPI()


# Frontend Routes
@app.route('/')
def serve_frontend():
    """Serve the main frontend page"""
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:filename>')
def serve_static_files(filename):
    """Serve static files (CSS, JS, etc.)"""
    return send_from_directory('frontend', filename)


# API Endpoints
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})


@app.route('/api/auction/create', methods=['POST'])
def create_auction():
    """Create a new auction"""
    data = request.get_json()
    
    # Validate required fields
    if 'agents' not in data or 'config' not in data:
        return jsonify({'success': False, 'error': 'Missing agents or config'}), 400
        
    result = auction_api.create_auction(data['agents'], data['config'])
    
    if result['success']:
        return jsonify(result)
    else:
        return jsonify(result), 400


@app.route('/api/auction/status', methods=['GET'])
def get_auction_status():
    """Get current auction status"""
    return jsonify(auction_api.get_auction_status())


@app.route('/api/auction/next-player', methods=['POST'])
def start_next_player():
    """Start auction for next player"""
    data = request.get_json() or {}
    role_filter = data.get('role_filter')
    
    result = auction_api.start_next_player_auction(role_filter)
    
    if result['success']:
        return jsonify(result)
    else:
        return jsonify(result), 400


@app.route('/api/auction/bid', methods=['POST'])
def make_bid():
    """Make a bid for current player"""
    data = request.get_json()
    
    if 'agent_id' not in data or 'amount' not in data:
        return jsonify({'success': False, 'error': 'Missing agent_id or amount'}), 400
        
    result = auction_api.make_bid(data['agent_id'], data['amount'])
    
    if result['success']:
        return jsonify(result)
    else:
        return jsonify(result), 400


@app.route('/api/auction/bot-bids', methods=['POST'])
def process_bot_bids():
    """Process automatic bids from bot agents"""
    result = auction_api.process_bot_bids()
    
    if result['success']:
        return jsonify(result)
    else:
        return jsonify(result), 400


@app.route('/api/auction/finalize', methods=['POST'])
def finalize_auction():
    """Finalize current player auction"""
    result = auction_api.finalize_player_auction()
    
    if result['success']:
        return jsonify(result)
    else:
        return jsonify(result), 400


@app.route('/api/players', methods=['GET'])
def get_players():
    """Get all available players"""
    try:
        players = load_players_from_excel()
        players_data = []
        
        for player in players:
            players_data.append({
                'name': player.name,
                'team': player.team,
                'role': player.role,
                'evaluation': player.evaluation,
                'standardized_evaluation': player.standardized_evaluation,
                'ranking': player.ranking,
                'fantasy_team': player.fantasy_team,
                'final_cost': player.final_cost
            })
            
        return jsonify({
            'success': True,
            'players': players_data,
            'total': len(players_data)
        })
        
    except Exception as e:
        logger.error(f"Error getting players: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/auction/results', methods=['GET'])
def get_auction_results():
    """Get final auction results"""
    if not auction_api.auction:
        return jsonify({'success': False, 'error': 'No auction available'}), 400
        
    try:
        results = []
        
        for agent in auction_api.auction.agents:
            squad_players = []
            for player in agent.squad:
                squad_players.append({
                    'name': player.name,
                    'role': player.role,
                    'team': player.team,
                    'evaluation': player.evaluation,
                    'final_cost': player.final_cost
                })
                
            # Calculate squad metrics
            total_eval = agent.squad.objective(standardized=False)
            total_std_eval = agent.squad.objective(standardized=True)
            bestxi_eval = agent.squad.objective(bestxi=True, standardized=False)
            bestxi_std_eval = agent.squad.objective(bestxi=True, standardized=True)
            
            results.append({
                'agent_id': agent.agent_id,
                'squad': squad_players,
                'metrics': {
                    'total_evaluation': total_eval,
                    'total_standardized': total_std_eval,
                    'bestxi_evaluation': bestxi_eval,
                    'bestxi_standardized': bestxi_std_eval,
                    'credits_remaining': agent.current_credits,
                    'credits_spent': 1000 - agent.current_credits
                }
            })
            
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error getting auction results: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting Fantasy Football Auction API Server...")
    print("📋 Available endpoints:")
    print("  GET  / - Frontend Web UI")
    print("  GET  /api/health - Health check")
    print("  POST /api/auction/create - Create new auction")
    print("  GET  /api/auction/status - Get auction status")
    print("  POST /api/auction/next-player - Start next player auction")
    print("  POST /api/auction/bid - Make a bid")
    print("  POST /api/auction/bot-bids - Process bot bids")
    print("  POST /api/auction/finalize - Finalize player auction")
    print("  GET  /api/players - Get all players")
    print("  GET  /api/auction/results - Get final results")
    print("\n🌐 Server starting on http://localhost:8081")
    print("🎯 Frontend available at: http://localhost:8081")
    
    app.run(debug=True, host='0.0.0.0', port=8081)
