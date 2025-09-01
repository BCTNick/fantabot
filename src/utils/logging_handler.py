"""
Logging utilities for the auction system
"""

import logging
import queue
from datetime import datetime
from typing import Optional


class QueueHandler(logging.Handler):
    """Custom logging handler that puts log messages in a queue for GUI display"""
    
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


class AuctionLogger:
    """Centralized logging system for auction events"""
    
    def __init__(self, log_queue: queue.Queue):
        self.log_queue = log_queue
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        # Clear existing handlers
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create queue handler for GUI
        queue_handler = QueueHandler(self.log_queue)
        queue_handler.setFormatter(logging.Formatter('%(message)s'))
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[queue_handler],
            force=True
        )
    
    @staticmethod
    def log_auction_start(slots, initial_credits, num_agents, num_players):
        """Log auction configuration"""
        logger = logging.getLogger()
        logger.info("⚙️ CONFIGURAZIONE ASTA")
        logger.info("-" * 40)
        logger.info(f"📊 Slot per squadra:")
        logger.info(f"  GK: {slots.gk}")
        logger.info(f"  DEF: {slots.def_}")
        logger.info(f"  MID: {slots.mid}")
        logger.info(f"  ATT: {slots.att}")
        logger.info(f"💰 Crediti iniziali: {initial_credits}")
        logger.info(f"👥 Numero agenti: {num_agents}")
        logger.info(f"📋 Giocatori disponibili: {num_players}")
        logger.info(f"🏁 ASTA INIZIATA - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
    
    @staticmethod
    def log_player_auction_start(player):
        """Log start of player auction"""
        logger = logging.getLogger()
        logger.info(f"\n💫 GIOCATORE IN ASTA: {player.name} ({player.role}) - Valore: {player.evaluation}")
    
    @staticmethod
    def log_bid(agent_name, amount):
        """Log a bid"""
        logger = logging.getLogger()
        logger.info(f"💰 {agent_name} offre {amount} crediti")
    
    @staticmethod
    def log_no_bid(agent_name):
        """Log no bid"""
        logger = logging.getLogger()
        logger.info(f"🚫 {agent_name} non offre")
    
    @staticmethod
    def log_player_sold(player_name, winner, amount, remaining_credits):
        """Log player assignment"""
        logger = logging.getLogger()
        logger.info(f"✅ {player_name} assegnato a {winner} per {amount} crediti")
        logger.info(f"💳 {winner} - Crediti rimanenti: {remaining_credits}")
    
    @staticmethod
    def log_player_unsold(player_name):
        """Log unsold player"""
        logger = logging.getLogger()
        logger.info(f"❌ {player_name} non assegnato - nessuna offerta")
    
    @staticmethod
    def log_auction_end():
        """Log auction completion"""
        logger = logging.getLogger()
        logger.info("🏁 ASTA COMPLETATA!")
        logger.info(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
    
    @staticmethod
    def log_role_start(role):
        """Log start of role auction"""
        logger = logging.getLogger()
        logger.info(f"\n🔄 Inizio asta per {role}")
    
    @staticmethod
    def log_bid_round(current_price):
        """Log bidding round"""
        logger = logging.getLogger()
        logger.info(f"\n🔄 Round di offerte - Prezzo attuale: {current_price}")
    
    @staticmethod
    def log_highest_bid(agent_name, amount):
        """Log highest bid in round"""
        logger = logging.getLogger()
        logger.info(f"🏆 Offerta più alta automatica: {agent_name} con {amount} crediti")
    
    @staticmethod
    def log_error(error_msg, exception=None):
        """Log error"""
        logger = logging.getLogger()
        logger.error(f"❌ Errore: {error_msg}")
        if exception:
            import traceback
            logger.error(traceback.format_exc())
