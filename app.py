from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.web_auction import web_auction_manager

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

@app.route('/api/auction/create', methods=['POST'])
def create_auction():
    """Create a new auction with specified parameters"""
    try:
        data = request.get_json() or {}
        auction_id = web_auction_manager.create_auction(data)
        
        # Get the created auction config
        auction_status = web_auction_manager.get_auction_status(auction_id)
        
        return jsonify({
            'auction_id': auction_id,
            'status': 'created',
            'config': auction_status['config']
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auction/<auction_id>/start', methods=['POST'])
def start_auction(auction_id: str):
    """Start an auction"""
    try:
        data = request.get_json() or {}
        result = web_auction_manager.start_auction(auction_id, data)
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auction/<auction_id>/stop', methods=['POST'])
def stop_auction(auction_id: str):
    """Stop a running auction"""
    try:
        result = web_auction_manager.stop_auction(auction_id)
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auction/<auction_id>/bid', methods=['POST'])
def submit_bid(auction_id: str):
    """Submit a bid from a human agent"""
    try:
        data = request.get_json()
        if not data or 'agent_id' not in data or 'bid_amount' not in data:
            return jsonify({'error': 'agent_id and bid_amount are required'}), 400
        
        result = web_auction_manager.submit_human_bid(
            auction_id, 
            data['agent_id'], 
            data['bid_amount']
        )
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auction/<auction_id>/pass', methods=['POST'])
def submit_pass(auction_id: str):
    """Submit a pass (no bid) from human agents"""
    try:
        result = web_auction_manager.submit_human_pass(auction_id)
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auction/<auction_id>/live', methods=['GET'])
def get_auction_live_state(auction_id: str):
    """Get live auction state for real-time updates"""
    try:
        result = web_auction_manager.get_auction_live_state(auction_id)
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auction/<auction_id>/status', methods=['GET'])
def get_auction_status(auction_id: str):
    """Get auction status and results"""
    try:
        result = web_auction_manager.get_auction_status(auction_id)
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auction/<auction_id>', methods=['DELETE'])
def delete_auction(auction_id: str):
    """Delete an auction"""
    try:
        web_auction_manager.delete_auction(auction_id)
        return jsonify({'message': 'Auction deleted successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auctions', methods=['GET'])
def list_auctions():
    """List all auctions"""
    try:
        auctions_list = web_auction_manager.list_auctions()
        return jsonify({'auctions': auctions_list})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/agents/types', methods=['GET'])
def get_agent_types():
    """Get available agent types and their configurations"""
    agent_types = [
        {
            'type': 'human',
            'name': 'Human Agent',
            'description': 'Manual bidding agent for human players',
            'parameters': []
        },
        {
            'type': 'random',
            'name': 'Random Agent',
            'description': 'Agent that bids randomly',
            'parameters': []
        },
        {
            'type': 'cap',
            'name': 'Cap-based Agent',
            'description': 'Agent that bids based on budget management strategy',
            'parameters': [
                {
                    'name': 'strategy',
                    'type': 'select',
                    'options': ['conservative', 'aggressive', 'balanced'],
                    'default': 'conservative'
                }
            ]
        },
        {
            'type': 'dynamic_cap',
            'name': 'Dynamic Cap Agent',
            'description': 'Agent with dynamic budget management',
            'parameters': []
        },
        {
            'type': 'rl_deep',
            'name': 'Deep RL Agent',
            'description': 'Reinforcement learning agent',
            'parameters': []
        }
    ]
    
    return jsonify({'agent_types': agent_types})

@app.route('/api/players', methods=['GET'])
def get_players():
    """Get all available players for the auction"""
    try:
        from src.data_loader import load_players_from_excel
        players = load_players_from_excel()
        
        players_data = [
            {
                'name': player.name,
                'team': player.team,
                'role': player.role,
                'evaluation': player.evaluation,
                'standardized_evaluation': round(player.standardized_evaluation, 3),
                'ranking': player.ranking
            }
            for player in players
        ]
        
        # Group by role for easier frontend consumption
        players_by_role = {
            'GK': [p for p in players_data if p['role'] == 'GK'],
            'DEF': [p for p in players_data if p['role'] == 'DEF'],
            'MID': [p for p in players_data if p['role'] == 'MID'],
            'ATT': [p for p in players_data if p['role'] == 'ATT']
        }
        
        return jsonify({
            'players': players_data,
            'players_by_role': players_by_role,
            'total_count': len(players_data)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)