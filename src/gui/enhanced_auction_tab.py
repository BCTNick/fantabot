"""
Enhanced auction tab with improved UI for practical use
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import queue
from typing import Dict, List, Callable, Optional
from src.models import Player
from src.agents.agent_class import Agent
from src.utils.validators import AuctionValidator


class EnhancedAuctionTab:
    """Enhanced auction tab with better information display and controls"""
    
    def __init__(self, parent, on_start_auction: Callable, on_stop_auction: Callable):
        self.parent = parent
        self.on_start_auction = on_start_auction
        self.on_stop_auction = on_stop_auction
        
        # State variables
        self.auction_running = False
        self.human_decision_result = None
        self.human_bid_amounts = {}
        self.bid_choice = tk.StringVar(value="nessuno")
        self.current_slots = {"GK": 3, "DEF": 8, "MID": 8, "ATT": 6}  # Default, will be updated
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the enhanced auction tab UI"""
        # Main container
        main_container = ttk.Frame(self.parent)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top control panel
        self.setup_control_panel(main_container)
        
        # Main content area with three columns
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Left column - Auction log (40%)
        self.setup_auction_log(content_frame)
        
        # Center column - Current auction info (35%)
        self.setup_current_auction_info(content_frame)
        
        # Right column - Human input / Auction stats (25%)
        self.setup_right_panel(content_frame)
    
    def setup_control_panel(self, parent):
        """Setup control panel with buttons and status"""
        control_frame = ttk.LabelFrame(parent, text="Controlli Asta", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Buttons row
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X)
        
        self.start_button = ttk.Button(
            button_frame, text="🚀 Avvia Asta", 
            command=self._on_start_clicked, style="Accent.TButton"
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(
            button_frame, text="⏹️ Ferma Asta", 
            command=self._on_stop_clicked, state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.save_button = ttk.Button(
            button_frame, text="💾 Salva Stato", 
            command=self._on_save_clicked
        )
        self.save_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Status display
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(status_frame, text="Stato:", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="Pronto per iniziare l'asta")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                     font=("Arial", 9))
        self.status_label.pack(side=tk.LEFT, padx=(5, 0))
    
    def setup_auction_log(self, parent):
        """Setup auction log display"""
        log_frame = ttk.LabelFrame(parent, text="📋 Log dell'Asta", padding=10)
        log_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Log text area with enhanced formatting
        self.auction_info_text = scrolledtext.ScrolledText(
            log_frame, height=25, state=tk.DISABLED, wrap=tk.WORD,
            font=("Consolas", 9)
        )
        self.auction_info_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags for colored output
        self.auction_info_text.tag_configure("player", foreground="#2E86AB", font=("Consolas", 9, "bold"))
        self.auction_info_text.tag_configure("bid", foreground="#A23B72", font=("Consolas", 9))
        self.auction_info_text.tag_configure("sold", foreground="#F18F01", font=("Consolas", 9, "bold"))
        self.auction_info_text.tag_configure("error", foreground="#C73E1D")
    
    def setup_current_auction_info(self, parent):
        """Setup current auction information panel"""
        info_frame = ttk.LabelFrame(parent, text="🏆 Giocatore in Asta", padding=10)
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 5))
        
        # Current player card
        self.current_player_card = ttk.Frame(info_frame)
        self.current_player_card.pack(fill=tk.X, pady=(0, 10))
        
        # Player name and info
        self.player_name_label = ttk.Label(
            self.current_player_card, text="In attesa...", 
            font=("Arial", 14, "bold"), foreground="#2E86AB"
        )
        self.player_name_label.pack(anchor=tk.W)
        
        self.player_details_label = ttk.Label(
            self.current_player_card, text="", font=("Arial", 10)
        )
        self.player_details_label.pack(anchor=tk.W)
        
        # Bidding info
        ttk.Separator(info_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        self.bidding_info_frame = ttk.Frame(info_frame)
        self.bidding_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.current_bid_label = ttk.Label(
            self.bidding_info_frame, text="Prezzo: - €", 
            font=("Arial", 12, "bold")
        )
        self.current_bid_label.pack(anchor=tk.W)
        
        self.highest_bidder_label = ttk.Label(
            self.bidding_info_frame, text="", font=("Arial", 10)
        )
        self.highest_bidder_label.pack(anchor=tk.W)
        
        # Quick stats
        ttk.Separator(info_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        self.quick_stats_frame = ttk.LabelFrame(info_frame, text="📊 Statistiche Rapide", padding=5)
        self.quick_stats_frame.pack(fill=tk.X)
        
        self.stats_text = tk.Text(self.quick_stats_frame, height=8, state=tk.DISABLED, 
                                 font=("Arial", 9), wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
    
    def setup_right_panel(self, parent):
        """Setup right panel for human input and auction summary"""
        right_frame = ttk.Frame(parent)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # Human input section (initially hidden)
        self.setup_human_input_section(right_frame)
        
        # Auction summary section
        self.setup_auction_summary(right_frame)
    
    def setup_human_input_section(self, parent):
        """Setup human input section"""
        self.human_input_frame = ttk.LabelFrame(parent, text="👤 Offerte Umane", padding=10)
        
        # Current player info for humans
        self.human_player_info = ttk.Label(self.human_input_frame, text="", 
                                          font=("Arial", 10, "bold"))
        self.human_player_info.pack(anchor=tk.W, pady=(0, 10))
        
        self.human_bid_info = ttk.Label(self.human_input_frame, text="")
        self.human_bid_info.pack(anchor=tk.W, pady=(0, 10))
        
        # No offer option
        self.no_offer_radio = ttk.Radiobutton(
            self.human_input_frame, text="❌ Nessuno vuole offrire", 
            variable=self.bid_choice, value="nessuno"
        )
        self.no_offer_radio.pack(anchor=tk.W, pady=5)
        
        # Separator
        ttk.Separator(self.human_input_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Frame for agent options
        self.agents_options_frame = ttk.Frame(self.human_input_frame)
        self.agents_options_frame.pack(fill=tk.X)
        
        # Buttons
        button_frame = ttk.Frame(self.human_input_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.confirm_button = ttk.Button(
            button_frame, text="✅ Conferma", 
            command=self._on_human_confirm, style="Accent.TButton"
        )
        self.confirm_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        self.no_offer_button = ttk.Button(
            button_frame, text="⏭️ Salta", 
            command=self._on_human_no_offer
        )
        self.no_offer_button.pack(side=tk.RIGHT)
        
        # Initially hide
        self.human_input_frame.pack_forget()
    
    def setup_auction_summary(self, parent):
        """Setup auction summary section"""
        self.summary_frame = ttk.LabelFrame(parent, text="📈 Riassunto Asta", padding=10)
        self.summary_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Summary tree view
        self.summary_tree = ttk.Treeview(
            self.summary_frame,
            columns=("Crediti", "Giocatori", "Valore"),
            show="tree headings",
            height=10
        )
        
        self.summary_tree.heading("#0", text="Agente")
        self.summary_tree.heading("Crediti", text="Crediti")
        self.summary_tree.heading("Giocatori", text="Giocatori")
        self.summary_tree.heading("Valore", text="Valore")
        
        self.summary_tree.column("#0", width=80)
        self.summary_tree.column("Crediti", width=60)
        self.summary_tree.column("Giocatori", width=60)
        self.summary_tree.column("Valore", width=60)
        
        self.summary_tree.pack(fill=tk.BOTH, expand=True)
    
    def update_current_player(self, player: Player):
        """Update current player display"""
        self.player_name_label.config(text=player.name)
        self.player_details_label.config(
            text=f"{player.role} • {player.team} • Valutazione: {player.evaluation}"
        )
    
    def set_auction_slots(self, slots_dict: Dict[str, int]):
        """Set the current auction slots configuration"""
        self.current_slots = slots_dict.copy()
    
    def update_bidding_info(self, current_price: int, highest_bidder: Optional[str] = None):
        """Update bidding information"""
        self.current_bid_label.config(text=f"Prezzo: {current_price} €")
        
        if highest_bidder:
            self.highest_bidder_label.config(text=f"Offerta più alta: {highest_bidder}")
        else:
            self.highest_bidder_label.config(text="Nessuna offerta")
    
    def show_human_input(self, player: Player, human_agents: List[Agent], 
                        current_price: int, highest_bidder: Optional[Agent]):
        """Show human input section"""
        # Update player info
        self.human_player_info.config(text=f"{player.name} ({player.role})")
        
        if highest_bidder:
            bid_text = f"Offerta attuale: {current_price} € (da {highest_bidder.agent_id})"
        else:
            bid_text = f"Prezzo base: {current_price} €"
        self.human_bid_info.config(text=bid_text)
        
        # Clear previous options
        for widget in self.agents_options_frame.winfo_children():
            widget.destroy()
        
        self.human_bid_amounts.clear()
        
        # Add agent options
        for agent in human_agents:
            max_bid = AuctionValidator.get_max_bid_for_agent(
                agent, current_price, player.role, 
                self.current_slots
            )
            
            if max_bid <= current_price:
                continue
            
            # Agent frame
            agent_frame = ttk.Frame(self.agents_options_frame)
            agent_frame.pack(fill=tk.X, pady=3)
            
            # Radio button
            ttk.Radiobutton(
                agent_frame, 
                text=f"💰 {agent.agent_id} ({agent.current_credits}€)", 
                variable=self.bid_choice, 
                value=agent.agent_id
            ).pack(anchor=tk.W)
            
            # Bid amount
            bid_frame = ttk.Frame(agent_frame)
            bid_frame.pack(fill=tk.X, padx=(20, 0), pady=(2, 0))
            
            ttk.Label(bid_frame, text="Offerta:").pack(side=tk.LEFT)
            
            self.human_bid_amounts[agent.agent_id] = tk.IntVar(value=current_price + 1)
            ttk.Spinbox(
                bid_frame, 
                from_=current_price + 1, 
                to=max_bid,
                width=8,
                textvariable=self.human_bid_amounts[agent.agent_id]
            ).pack(side=tk.LEFT, padx=(5, 10))
            
            ttk.Label(bid_frame, text=f"(max: {max_bid}€)").pack(side=tk.LEFT)
        
        # Show human input frame
        self.summary_frame.pack_forget()
        self.human_input_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
    
    def hide_human_input(self):
        """Hide human input and show summary"""
        self.human_input_frame.pack_forget()
        self.summary_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
    
    def wait_for_human_decision(self) -> Dict:
        """Wait for human decision"""
        self.human_decision_result = None
        
        while self.human_decision_result is None and self.auction_running:
            self.parent.update()
        
        result = self.human_decision_result or {"agent_id": "nessuno", "amount": 0}
        self.hide_human_input()
        return result
    
    def update_auction_summary(self, agents: List[Agent]):
        """Update auction summary display"""
        # Clear existing items
        for item in self.summary_tree.get_children():
            self.summary_tree.delete(item)
        
        # Add agents
        for agent in agents:
            squad_value = agent.squad.objective(standardized=False) if len(agent.squad) > 0 else 0
            
            self.summary_tree.insert(
                "", tk.END,
                text=agent.agent_id,
                values=(
                    f"{agent.current_credits}€",
                    len(agent.squad),
                    f"{squad_value:.1f}"
                )
            )
    
    def add_log_message(self, message: str):
        """Add message to auction log with formatting"""
        self.auction_info_text.config(state=tk.NORMAL)
        
        # Determine tag based on message content
        tag = None
        if "GIOCATORE IN ASTA" in message:
            tag = "player"
        elif "offre" in message and "crediti" in message:
            tag = "bid"
        elif "assegnato" in message:
            tag = "sold"
        elif "❌" in message or "Errore" in message:
            tag = "error"
        
        self.auction_info_text.insert(tk.END, message + "\n", tag)
        self.auction_info_text.see(tk.END)
        self.auction_info_text.config(state=tk.DISABLED)
    
    def update_quick_stats(self, summary: Dict):
        """Update quick statistics display"""
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        
        stats_text = f"🎯 Giocatori venduti: {summary['sold_players']}/{summary['total_players']}\n"
        stats_text += f"🚫 Non assegnati: {summary['unsold_players']}\n\n"
        
        stats_text += "👥 AGENTI:\n"
        for agent in summary['agents']:
            stats_text += f"• {agent['name']}: {agent['credits_remaining']}€\n"
            stats_text += f"  ({agent['players_count']} giocatori)\n"
        
        self.stats_text.insert(tk.END, stats_text)
        self.stats_text.config(state=tk.DISABLED)
    
    def set_auction_running(self, running: bool):
        """Update UI based on auction state"""
        self.auction_running = running
        
        if running:
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_var.set("⚡ Asta in corso...")
        else:
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_var.set("✅ Asta completata" if self.auction_running else "⏸️ Pronto per iniziare")
    
    # Event handlers
    def _on_start_clicked(self):
        if self.on_start_auction:
            self.on_start_auction()
    
    def _on_stop_clicked(self):
        if self.on_stop_auction:
            self.on_stop_auction()
    
    def _on_save_clicked(self):
        messagebox.showinfo("Info", "Funzione di salvataggio in sviluppo")
    
    def _on_human_confirm(self):
        """Handle human confirm button"""
        choice = self.bid_choice.get()
        
        if choice == "nessuno":
            self.human_decision_result = {"agent_id": "nessuno", "amount": 0}
        else:
            amount = self.human_bid_amounts[choice].get()
            self.human_decision_result = {"agent_id": choice, "amount": amount}
    
    def _on_human_no_offer(self):
        """Handle no offer button"""
        self.bid_choice.set("nessuno")
        self.human_decision_result = {"agent_id": "nessuno", "amount": 0}
