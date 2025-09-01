"""
Enhanced main GUI application with modular design
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import sys
import os
from typing import Dict, List

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.models import Slots
from src.agents.agent_class import RandomAgent
from src.agents.cap_based_agent import CapAgent
from src.agents.enhanced_cap_agent import EnhancedCapAgent
from src.agents.dynamic_cap_based_agent import DynamicCapAgent
from src.agents.rl_deep_agent import RLDeepAgent
from src.agents.human_agent import HumanAgent
from src.data_loader import load_players_from_excel
from src.core.enhanced_auction import EnhancedAuction
from src.utils.logging_handler import AuctionLogger
from src.utils.tts_manager import TTSManager
from src.utils.file_manager import FileManager
from src.gui.config_tab import ConfigurationTab
from src.gui.enhanced_auction_tab import EnhancedAuctionTab


class GUIHumanAgent(HumanAgent):
    """Human agent that participates through GUI"""
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
    
    def make_offer_decision(self, current_player, current_price, highest_bidder, player_list, other_agents) -> str:
        # This will be handled by the GUI
        return "no_offer"


class EnhancedFantaAuctionGUI:
    """Enhanced FantaBot GUI with modular design and improved functionality"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("FantaBot Pro - Sistema Asta Fantacalcio 🏆")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Initialize core components
        self.log_queue = queue.Queue()
        self.auction = None
        self.auction_thread = None
        self.tts_manager = TTSManager()
        self.logger = AuctionLogger(self.log_queue)
        
        # Agent types available
        self.agent_types = {
            "Human Agent": GUIHumanAgent,
            "Random Agent": RandomAgent,
            "Cap Agent": CapAgent,
            "Enhanced Cap Agent": EnhancedCapAgent,
            "Dynamic Cap Agent": DynamicCapAgent,
            "RL Deep Agent": RLDeepAgent
        }
        
        # UI state
        self.auction_running = False
        
        self.setup_ui()
        self.start_log_processing()
    
    def setup_ui(self):
        """Setup the main user interface"""
        # Configure style
        self.setup_styles()
        
        # Create main notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configuration tab
        self.config_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.config_frame, text="⚙️ Configurazione")
        self.config_tab = ConfigurationTab(
            self.config_frame, 
            self.agent_types,
            on_config_changed=self.on_config_changed
        )
        
        # Enhanced auction tab
        self.auction_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.auction_frame, text="🏆 Asta Live")
        self.auction_tab = EnhancedAuctionTab(
            self.auction_frame,
            on_start_auction=self.start_auction,
            on_stop_auction=self.stop_auction
        )
        
        # Results tab
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="📊 Risultati")
        self.setup_results_tab()
        
        # Logs tab
        self.logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.logs_frame, text="📝 Log Completi")
        self.setup_logs_tab()
    
    def setup_styles(self):
        """Setup custom styles"""
        style = ttk.Style()
        
        # Configure modern theme
        if "clam" in style.theme_names():
            style.theme_use("clam")
        
        # Custom button styles
        style.configure("Accent.TButton", font=("Arial", 10, "bold"))
        style.configure("Success.TButton", font=("Arial", 9, "bold"))
        style.configure("Danger.TButton", font=("Arial", 9, "bold"))
    
    def setup_results_tab(self):
        """Setup results analysis tab"""
        # Results will be implemented as needed
        placeholder = ttk.Label(
            self.results_frame, 
            text="📊 I risultati dell'asta appariranno qui dopo il completamento",
            font=("Arial", 12),
            anchor="center"
        )
        placeholder.pack(expand=True)
    
    def setup_logs_tab(self):
        """Setup comprehensive logs tab"""
        # Controls frame
        controls_frame = ttk.Frame(self.logs_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(
            controls_frame, text="🗑️ Pulisci Log", 
            command=self.clear_logs
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            controls_frame, text="💾 Salva Log", 
            command=self.save_logs
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            controls_frame, text="📤 Esporta Risultati", 
            command=self.export_results
        ).pack(side=tk.LEFT)
        
        # Log display
        log_frame = ttk.LabelFrame(self.logs_frame, text="📝 Log Completo", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        from tkinter import scrolledtext
        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=20, state=tk.DISABLED,
            font=("Consolas", 9), wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def create_agents(self, agents_config: List[Dict]):
        """Create agent instances from configuration"""
        agents = []
        
        for agent_config in agents_config:
            agent_class = self.agent_types[agent_config["type"]]
            
            if agent_class == GUIHumanAgent:
                agent = GUIHumanAgent(agent_id=agent_config["name"])
            else:
                agent = agent_class(agent_id=agent_config["name"])
            
            agents.append(agent)
        
        return agents
    
    def start_auction(self):
        """Start the auction with validation"""
        # Get configuration
        config = self.config_tab.get_configuration()
        
        if not config["valid"]:
            messagebox.showerror("Errore Configurazione", config["error"])
            return
        
        # Validate player data
        try:
            players = load_players_from_excel()
            if not players:
                messagebox.showerror("Errore", "Nessun giocatore trovato nel file Excel")
                return
        except Exception as e:
            messagebox.showerror("Errore", f"Errore nel caricamento giocatori:\n{str(e)}")
            return
        
        # Create agents
        try:
            agents = self.create_agents(config["agents"])
        except Exception as e:
            messagebox.showerror("Errore", f"Errore nella creazione agenti:\n{str(e)}")
            return
        
        # Create slots
        slots_config = config["settings"]["slots"]
        slots = Slots(**slots_config)
        
        # Create enhanced auction
        self.auction = EnhancedAuction(
            slots=slots,
            agents=agents,
            players=players,
            initial_credits=config["settings"]["initial_credits"]
        )
        
        # Configure auction tab with slots
        self.auction_tab.set_auction_slots(self.auction.slots_dict)
        
        # Set auction callbacks
        self.auction.set_callbacks(
            on_player_start=self.on_player_auction_start,
            on_bid_made=self.on_bid_made,
            on_player_sold=self.on_player_sold,
            on_human_input_needed=self.on_human_input_needed
        )
        
        # Update UI
        self.auction_running = True
        self.auction_tab.set_auction_running(True)
        
        # Start auction in separate thread
        self.auction_thread = threading.Thread(
            target=self._run_auction_thread,
            args=(config["settings"]["auction_type"], config["settings"]["per_ruolo"]),
            daemon=True
        )
        self.auction_thread.start()
        
        # Switch to auction tab
        self.notebook.select(1)
    
    def _run_auction_thread(self, auction_type: str, per_ruolo: bool):
        """Run auction in separate thread"""
        try:
            self.auction.run_full_auction(auction_type, per_ruolo)
        except Exception as e:
            self.logger.log_error(f"Errore nell'esecuzione dell'asta: {str(e)}", e)
        finally:
            # Update UI in main thread
            self.root.after(0, self.auction_finished)
    
    def stop_auction(self):
        """Stop current auction"""
        if self.auction:
            self.auction.stop_auction()
    
    def auction_finished(self):
        """Handle auction completion"""
        self.auction_running = False
        self.auction_tab.set_auction_running(False)
        
        if self.auction:
            # Update final summary
            summary = self.auction.get_auction_summary()
            self.auction_tab.update_auction_summary(self.auction.agents)
            self.auction_tab.update_quick_stats(summary)
            
            # Show completion message
            messagebox.showinfo(
                "Asta Completata! 🎉", 
                f"L'asta è stata completata con successo!\n\n"
                f"Giocatori venduti: {summary['sold_players']}\n"
                f"Giocatori non assegnati: {summary['unsold_players']}"
            )
    
    # Auction event callbacks
    def on_player_auction_start(self, player):
        """Called when a new player auction starts"""
        self.auction_tab.update_current_player(player)
        
        # TTS announcement
        if self.tts_manager.available:
            self.tts_manager.announce_player(player.name, player.role, player.evaluation)
    
    def on_bid_made(self, agent_name: str, amount: int):
        """Called when a bid is made"""
        self.auction_tab.update_bidding_info(amount, agent_name)
        
        # Update summary periodically
        if self.auction:
            self.auction_tab.update_auction_summary(self.auction.agents)
    
    def on_player_sold(self, player, winner: str, amount: int):
        """Called when a player is sold"""
        # TTS announcement
        if self.tts_manager.available:
            self.tts_manager.announce_winner(player.name, winner, amount)
        
        # Update summary
        if self.auction:
            summary = self.auction.get_auction_summary()
            self.auction_tab.update_auction_summary(self.auction.agents)
            self.auction_tab.update_quick_stats(summary)
    
    def on_human_input_needed(self, player, human_agents, current_price, highest_bidder):
        """Called when human input is needed"""
        # Show human input in main thread
        self.root.after(0, lambda: self.auction_tab.show_human_input(
            player, human_agents, current_price, highest_bidder
        ))
        
        # Wait for decision
        return self.auction_tab.wait_for_human_decision()
    
    def on_config_changed(self):
        """Called when configuration changes"""
        # Could implement auto-save or validation here
        pass
    
    def start_log_processing(self):
        """Start processing log messages"""
        self.process_log_queue()
    
    def process_log_queue(self):
        """Process messages from log queue"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                
                # Add to main log
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, message + "\n")
                self.log_text.see(tk.END)
                self.log_text.config(state=tk.DISABLED)
                
                # Add to auction tab if running
                if self.auction_running:
                    self.auction_tab.add_log_message(message)
                
                # Process for TTS
                self._process_message_for_tts(message)
                
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_log_queue)
    
    def _process_message_for_tts(self, message: str):
        """Process message for TTS output"""
        if not self.tts_manager.available or not self.auction_running:
            return
        
        config = self.config_tab.get_configuration()
        if config["valid"]:
            self.tts_manager.speak_for_agent("", message, config["agents"])
    
    # Menu actions
    def clear_logs(self):
        """Clear log display"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def save_logs(self):
        """Save logs to file"""
        try:
            logs = self.log_text.get(1.0, tk.END)
            filepath = FileManager.save_logs_to_file(logs)
            messagebox.showinfo("Successo", f"Log salvati in:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Errore", f"Errore nel salvare i log:\n{str(e)}")
    
    def export_results(self):
        """Export auction results"""
        if not self.auction or not self.auction.agents:
            messagebox.showwarning("Attenzione", "Nessun risultato di asta da esportare")
            return
        
        try:
            filepath = FileManager.save_auction_results(self.auction.agents)
            messagebox.showinfo("Successo", f"Risultati esportati in:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Errore", f"Errore nell'esportazione:\n{str(e)}")


def main():
    """Main application entry point"""
    root = tk.Tk()
    
    # Set app icon if available
    try:
        # You can add an icon file here
        # root.iconbitmap("icon.ico")
        pass
    except:
        pass
    
    app = EnhancedFantaAuctionGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()
