"""
WebSocket server for real-time auction frontend
"""

import asyncio
import websockets
import json
import threading
import time
import sys
import os
import socket
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.models import Slots
from src.agents.agent_class import RandomAgent
from src.agents.cap_based_agent import CapAgent
from src.auction import Auction
from src.data_loader import load_players_from_excel


class WebAuction:
    """Auction wrapper with WebSocket support for real-time updates"""
    
    def __init__(self, slots, agents, players, initial_credits=1000):
        self.auction = Auction(slots, agents, players, initial_credits)
        self.connected_clients = set()
        self.current_auction_state = {
            'phase': 'waiting',  # waiting, auction, completed
            'current_player': None,
            'current_price': 0,
            'highest_bidder': None,
            'agents': [],
            'completed_auctions': []
        }
        self.auction_delay = 2  # Seconds between bids for visibility
        
    async def register_client(self, websocket):
        """Register a new WebSocket client"""
        print(f"📝 Registering client: {websocket.remote_address}")
        self.connected_clients.add(websocket)
        print(f"📊 Total connected clients: {len(self.connected_clients)}")
        # Send current state to new client
        await self.send_state_update()
        
    async def unregister_client(self, websocket):
        """Unregister a WebSocket client"""
        print(f"📝 Unregistering client: {websocket.remote_address}")
        self.connected_clients.discard(websocket)
        print(f"📊 Total connected clients: {len(self.connected_clients)}")
        
    async def broadcast_message(self, message):
        """Broadcast message to all connected clients"""
        if self.connected_clients:
            await asyncio.gather(
                *[client.send(json.dumps(message)) for client in self.connected_clients],
                return_exceptions=True
            )
    
    async def send_state_update(self):
        """Send current auction state to all clients"""
        # Update agents data
        agents_data = []
        for agent in self.auction.agents:
            squad_by_role = {
                'GK': [{'name': p.name, 'cost': p.final_cost, 'evaluation': p.evaluation} 
                       for p in agent.squad.gk],
                'DEF': [{'name': p.name, 'cost': p.final_cost, 'evaluation': p.evaluation} 
                        for p in agent.squad.def_],
                'MID': [{'name': p.name, 'cost': p.final_cost, 'evaluation': p.evaluation} 
                        for p in agent.squad.mid],
                'ATT': [{'name': p.name, 'cost': p.final_cost, 'evaluation': p.evaluation} 
                        for p in agent.squad.att]
            }
            
            agents_data.append({
                'id': agent.agent_id,
                'credits': agent.current_credits,
                'initial_credits': agent.initial_credits,
                'squad': squad_by_role,
                'total_players': len(agent._squad)
            })
        
        self.current_auction_state['agents'] = agents_data
        
        message = {
            'type': 'state_update',
            'data': self.current_auction_state
        }
        
        await self.broadcast_message(message)
    
    async def single_player_async(self, player):
        """Async version of single_player with WebSocket updates"""
        self.current_auction_state['phase'] = 'auction'
        self.current_auction_state['current_player'] = {
            'name': player.name,
            'role': player.role,
            'evaluation': player.evaluation
        }
        self.current_auction_state['current_price'] = 0
        self.current_auction_state['highest_bidder'] = None
        
        await self.send_state_update()
        await asyncio.sleep(1)  # Brief pause to show new player
        
        agent_index = 0
        no_bid_count = 0
        
        while True:
            current_agent = self.auction.agents[agent_index]
            
            # Check if agent can participate
            if self.auction.can_participate_in_bid(
                agent=current_agent, 
                current_price=self.current_auction_state['current_price'], 
                position=player.role, 
                slots=self.auction.slots
            ):
                # Get other agents
                other_agents = [agent for agent in self.auction.agents if agent.agent_id != current_agent.agent_id]
                
                # Get decision
                decision = current_agent.make_offer_decision(
                    current_player=player,
                    current_price=self.current_auction_state['current_price'],
                    highest_bidder=self.current_auction_state['highest_bidder'],
                    player_list=self.auction.players,
                    other_agents=other_agents
                )
                
                if decision == "offer_+1":
                    self.current_auction_state['current_price'] += 1
                    self.current_auction_state['highest_bidder'] = current_agent.agent_id
                    no_bid_count = 0
                    
                    # Broadcast bid update
                    await self.broadcast_message({
                        'type': 'bid_update',
                        'data': {
                            'agent': current_agent.agent_id,
                            'action': 'bid',
                            'new_price': self.current_auction_state['current_price']
                        }
                    })
                else:
                    no_bid_count += 1
                    
                    # Broadcast pass update
                    await self.broadcast_message({
                        'type': 'bid_update',
                        'data': {
                            'agent': current_agent.agent_id,
                            'action': 'pass',
                            'price': self.current_auction_state['current_price']
                        }
                    })
            else:
                no_bid_count += 1
                
                # Broadcast cannot bid update
                await self.broadcast_message({
                    'type': 'bid_update',
                    'data': {
                        'agent': current_agent.agent_id,
                        'action': 'cannot_bid',
                        'price': self.current_auction_state['current_price']
                    }
                })
            
            await self.send_state_update()
            await asyncio.sleep(self.auction_delay)  # Delay for visualization
            
            agent_index = (agent_index + 1) % len(self.auction.agents)
            
            if no_bid_count >= (len(self.auction.agents) - 1):
                break
        
        # Finalize sale
        if self.current_auction_state['highest_bidder']:
            player.fantasy_team = self.current_auction_state['highest_bidder']
            player.final_cost = self.current_auction_state['current_price']
            
            winner_agent = next(agent for agent in self.auction.agents 
                              if agent.agent_id == self.current_auction_state['highest_bidder'])
            winner_agent._squad.append(player)
            winner_agent.current_credits -= self.current_auction_state['current_price']
            
            result = 'sold'
        else:
            player.fantasy_team = "UNSOLD"
            player.final_cost = 0
            result = 'unsold'
        
        # Add to completed auctions
        self.current_auction_state['completed_auctions'].append({
            'player': {
                'name': player.name,
                'role': player.role,
                'evaluation': player.evaluation
            },
            'final_price': player.final_cost,
            'winner': player.fantasy_team,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
        # Broadcast final result
        await self.broadcast_message({
            'type': 'auction_complete',
            'data': {
                'player': player.name,
                'final_price': player.final_cost,
                'winner': player.fantasy_team,
                'result': result
            }
        })
        
        await self.send_state_update()
        await asyncio.sleep(2)  # Pause before next auction
        
        return self.current_auction_state['highest_bidder']
    
    async def run_auction_async(self, auction_type="random", per_ruolo=True):
        """Run the complete auction asynchronously"""
        roles = ["GK", "DEF", "MID", "ATT"]
        
        players_to_auction = self.auction.players[:]
        if auction_type == "random":
            import random
            random.shuffle(players_to_auction)
        elif auction_type == "listone":
            players_to_auction.sort(key=lambda p: p.name)
        elif auction_type == "chiamata":
            players_to_auction.sort(key=lambda p: p.evaluation, reverse=True)
        
        if per_ruolo:
            for role in roles:
                await self.broadcast_message({
                    'type': 'role_start',
                    'data': {'role': role}
                })
                
                while True:
                    all_full = all(len(getattr(agent.squad, role.lower() if role != "DEF" else "def_")) >= self.auction.slots[role] 
                                 for agent in self.auction.agents)
                    if all_full:
                        break
                        
                    next_player = next((p for p in players_to_auction if p.role == role and p.fantasy_team is None), None)
                    if next_player is None:
                        break
                    await self.single_player_async(next_player)
        else:
            total_slots = sum(self.auction.slots.values())
            while True:
                all_full = all(len(agent.squad) >= total_slots for agent in self.auction.agents)
                if all_full:
                    break
                    
                next_player = next((p for p in players_to_auction if p.fantasy_team is None), None)
                if next_player is None:
                    break
                await self.single_player_async(next_player)
        
        # Auction completed
        self.current_auction_state['phase'] = 'completed'
        await self.broadcast_message({
            'type': 'auction_finished',
            'data': {'message': 'All auctions completed!'}
        })
        await self.send_state_update()


# Global auction instance
web_auction = None

async def handle_client(websocket, path):
    """Handle WebSocket client connections"""
    global web_auction
    
    print(f"🔌 New client connected from {websocket.remote_address}")
    
    try:
        await web_auction.register_client(websocket)
        print(f"✅ Client registered successfully")
        
        async for message in websocket:
            try:
                data = json.loads(message)
                print(f"📨 Received message: {data}")
                
                if data['type'] == 'start_auction':
                    print("🏁 Starting auction...")
                    # Start auction in background
                    asyncio.create_task(web_auction.run_auction_async(
                        auction_type=data.get('auction_type', 'random'),
                        per_ruolo=data.get('per_ruolo', True)
                    ))
                elif data['type'] == 'get_state':
                    print("📊 Sending state update...")
                    await web_auction.send_state_update()
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
            except Exception as e:
                print(f"❌ Error handling message: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        print(f"🔌 Client disconnected")
    except Exception as e:
        print(f"❌ Client error: {e}")
    finally:
        await web_auction.unregister_client(websocket)
        print(f"🔌 Client unregistered")


def setup_auction():
    """Setup the auction with default configuration"""
    global web_auction
    
    # Create agents
    agents = [
        CapAgent(agent_id="cap_agent_1", cap_strategy="bestxi_based", bestxi_budget=0.9),
        CapAgent(agent_id="cap_agent_2", cap_strategy="bestxi_based", bestxi_budget=0.85),
        CapAgent(agent_id="cap_agent_3", cap_strategy="bestxi_based", bestxi_budget=0.8),
        RandomAgent(agent_id="random_agent_1")
    ]
    
    # Create slots
    slots = Slots(gk=1, def_=3, mid=3, att=3)
    
    # Load players
    players = load_players_from_excel("data/players_list.xlsx")
    
    # Create web auction
    web_auction = WebAuction(slots, agents, players, initial_credits=1000)
    
    print("✅ Auction setup complete!")
    print(f"   Agents: {len(agents)}")
    print(f"   Players: {len(players)}")
    print(f"   Slots: {slots.to_dict()}")


def find_available_port(start_port=8765, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise OSError(f"No available port found in range {start_port}-{start_port + max_attempts}")

async def main():
    """Main function to start the WebSocket server"""
    setup_auction()
    
    # Find available port
    try:
        port = find_available_port()
        print(f"🚀 Starting WebSocket server on port {port}...")
        print(f"   Server will run on ws://localhost:{port}")
        print("   Open frontend/index.html in your browser")
        print(f"   Note: Update frontend JavaScript files to use port {port} if different from 8765")
    except OSError as e:
        print(f"❌ Error finding available port: {e}")
        return
    
    try:
        async with websockets.serve(handle_client, "localhost", port):
            print("✅ WebSocket server started!")
            
            await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
