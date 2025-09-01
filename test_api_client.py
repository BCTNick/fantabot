"""
Test client per le API dell'asta di fantacalcio
Questo script dimostra come usare le API per gestire un'asta
"""

import requests
import json
import time
from typing import Dict, List

class AuctionClient:
    """Client per interagire con le API dell'asta"""
    
    def __init__(self, base_url: str = "http://localhost:8081"):
        self.base_url = base_url
        self.session_id = None
        
    def health_check(self):
        """Test connessione al server"""
        try:
            response = requests.get(f"{self.base_url}/api/health")
            return response.json()
        except Exception as e:
            return {'error': str(e)}
    
    def create_auction(self, agents_config: List[Dict], auction_config: Dict):
        """Crea una nuova asta"""
        payload = {
            'agents': agents_config,
            'config': auction_config
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/auction/create",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            result = response.json()
            if result.get('success'):
                self.session_id = result.get('session_id')
            return result
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_status(self):
        """Ottieni stato corrente dell'asta"""
        try:
            response = requests.get(f"{self.base_url}/api/auction/status")
            return response.json()
        except Exception as e:
            return {'error': str(e)}
    
    def start_next_player(self, role_filter: str = None):
        """Inizia asta per il prossimo giocatore"""
        payload = {}
        if role_filter:
            payload['role_filter'] = role_filter
            
        try:
            response = requests.post(
                f"{self.base_url}/api/auction/next-player",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            return response.json()
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def make_bid(self, agent_id: str, amount: int):
        """Fai un'offerta per il giocatore corrente"""
        payload = {
            'agent_id': agent_id,
            'amount': amount
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/auction/bid",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            return response.json()
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def finalize_auction(self):
        """Finalizza l'asta del giocatore corrente"""
        try:
            response = requests.post(f"{self.base_url}/api/auction/finalize")
            return response.json()
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_players(self):
        """Ottieni lista di tutti i giocatori"""
        try:
            response = requests.get(f"{self.base_url}/api/players")
            return response.json()
        except Exception as e:
            return {'error': str(e)}
    
    def get_results(self):
        """Ottieni risultati finali dell'asta"""
        try:
            response = requests.get(f"{self.base_url}/api/auction/results")
            return response.json()
        except Exception as e:
            return {'error': str(e)}


def demo_auction():
    """Dimostrazione di come usare le API"""
    
    print("🚀 Demo delle API dell'Asta di Fantacalcio")
    print("=" * 50)
    
    # Crea client
    client = AuctionClient()
    
    # Test connessione
    print("\n1. Test connessione al server...")
    health = client.health_check()
    if 'error' in health:
        print(f"❌ Errore connessione: {health['error']}")
        print("💡 Assicurati che il server sia avviato con: python api_server.py")
        return
    else:
        print(f"✅ Server OK - {health}")
    
    # Configurazione agenti
    agents_config = [
        {'type': 'human', 'id': 'player1'},
        {'type': 'human', 'id': 'player2'},
        {'type': 'cap', 'id': 'bot1'},
        {'type': 'random', 'id': 'bot2'}
    ]
    
    # Configurazione asta
    auction_config = {
        'initial_credits': 1000,
        'slots_gk': 3,
        'slots_def': 8,
        'slots_mid': 8,
        'slots_att': 6
    }
    
    # Crea asta
    print("\n2. Creazione asta...")
    result = client.create_auction(agents_config, auction_config)
    if not result.get('success'):
        print(f"❌ Errore creazione asta: {result.get('error')}")
        return
    
    print(f"✅ Asta creata - Session ID: {result.get('session_id')}")
    
    # Stato iniziale
    print("\n3. Stato iniziale asta...")
    status = client.get_status()
    print(f"Stato: {status.get('state')}")
    print(f"Agenti: {len(status.get('agents', []))}")
    for agent in status.get('agents', []):
        print(f"  - {agent['id']} ({agent['type']}): {agent['credits']} crediti")
    
    # Inizia asta per primo giocatore
    print("\n4. Inizio asta primo giocatore...")
    next_player = client.start_next_player("GK")  # Inizia con i portieri
    if not next_player.get('success'):
        print(f"❌ Errore: {next_player.get('error')}")
        return
    
    if next_player.get('completed'):
        print("✅ Asta completata - nessun giocatore disponibile")
        return
        
    player = next_player.get('player')
    print(f"🏆 Giocatore in asta: {player['name']} ({player['role']}) - Valutazione: {player['evaluation']}")
    
    # Simula alcune offerte
    print("\n5. Simulazione offerte...")
    
    # Prima offerta
    bid1 = client.make_bid('player1', 50)
    if bid1.get('success'):
        print(f"💰 player1 offre 50 - Offerta attuale: {bid1.get('current_price')}")
    
    # Seconda offerta
    bid2 = client.make_bid('player2', 75)
    if bid2.get('success'):
        print(f"💰 player2 offre 75 - Offerta attuale: {bid2.get('current_price')}")
    
    # Stato durante l'asta
    status = client.get_status()
    current_player = status.get('current_player')
    if current_player:
        print(f"🔥 Offerta più alta: {current_player.get('current_price')} da {current_player.get('highest_bidder')}")
    
    # Finalizza l'asta
    print("\n6. Finalizzazione asta giocatore...")
    finalize = client.finalize_auction()
    if finalize.get('success'):
        if finalize.get('sold'):
            print(f"✅ {finalize.get('player')} venduto a {finalize.get('buyer')} per {finalize.get('price')}")
        else:
            print(f"❌ {finalize.get('player')} invenduto")
    
    # Stato finale
    print("\n7. Stato dopo vendita...")
    status = client.get_status()
    for agent in status.get('agents', []):
        print(f"  - {agent['id']}: {agent['credits']} crediti, {agent['squad_size']} giocatori")
    
    print("\n🎯 Demo completata! Il server è pronto per essere collegato a un'interfaccia web.")
    print("\n📋 Prossimi passi:")
    print("  1. Creare un'interfaccia web (React/Vue/Angular)")
    print("  2. Collegare gli endpoint API")
    print("  3. Implementare real-time updates (WebSocket)")
    print("  4. Aggiungere autenticazione utenti")


if __name__ == "__main__":
    demo_auction()
