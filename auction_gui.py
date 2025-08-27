import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, simpledialog
import threading
import queue
import logging
import sys
import os
from datetime import datetime
from typing import List, Dict

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.models import Slots
from src.agents.agent_class import RandomAgent
from src.agents.cap_based_agent import CapAgent
from src.agents.dynamic_cap_based_agent import DynamicCapAgent
from src.agents.rl_deep_agent import RLDeepAgent
from src.agents.human_agent import HumanAgent
from src.auction import Auction
from src.data_loader import load_players_from_excel


class QueueHandler(logging.Handler):
    """Custom logging handler that puts log messages in a queue for GUI display"""
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


class GUIHumanAgent(HumanAgent):
    """Human agent that participates through GUI group selection"""
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
    
    def make_offer_decision(self, current_player, current_price, highest_bidder, player_list, other_agents) -> str:
        # This method won't be called in the new system as human agents
        # participate through the group selector dialog
        return "no_offer"


class FantaAuctionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FantaBot - Asta di Fantacalcio")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.agents_config = []
        self.auction = None
        self.log_queue = queue.Queue()
        self.auction_running = False
        
        # Available agent types
        self.agent_types = {
            "Human Agent": GUIHumanAgent,
            "Random Agent": RandomAgent,
            "Cap Agent": CapAgent,
            "Dynamic Cap Agent": DynamicCapAgent,
            "RL Deep Agent": RLDeepAgent
        }
        
        # Queue for human agent decisions
        self.human_decision_queue = queue.Queue()
        self.waiting_for_human = False
        
        self.setup_ui()
        self.setup_logging()
        
        # Start checking for log messages
        self.check_log_queue()
    
    def setup_ui(self):
        """Setup the main user interface"""
        # Create main notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Configuration
        self.config_frame = ttk.Frame(notebook)
        notebook.add(self.config_frame, text="Configurazione")
        self.setup_config_tab()
        
        # Tab 2: Auction
        self.auction_frame = ttk.Frame(notebook)
        notebook.add(self.auction_frame, text="Asta")
        self.setup_auction_tab()
        
        # Tab 3: Logs
        self.logs_frame = ttk.Frame(notebook)
        notebook.add(self.logs_frame, text="Log")
        self.setup_logs_tab()
    
    def setup_config_tab(self):
        """Setup the configuration tab"""
        # Main container with scrollbar
        canvas = tk.Canvas(self.config_frame)
        scrollbar = ttk.Scrollbar(self.config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Auction Settings Section
        settings_group = ttk.LabelFrame(scrollable_frame, text="Impostazioni Asta", padding=10)
        settings_group.pack(fill=tk.X, padx=5, pady=5)
        
        # Initial credits
        ttk.Label(settings_group, text="Crediti iniziali:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.initial_credits_var = tk.IntVar(value=1000)
        ttk.Entry(settings_group, textvariable=self.initial_credits_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        # Slots configuration
        ttk.Label(settings_group, text="Portieri:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.gk_var = tk.IntVar(value=3)
        ttk.Entry(settings_group, textvariable=self.gk_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(settings_group, text="Difensori:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.def_var = tk.IntVar(value=8)
        ttk.Entry(settings_group, textvariable=self.def_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(settings_group, text="Centrocampisti:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.mid_var = tk.IntVar(value=8)
        ttk.Entry(settings_group, textvariable=self.mid_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(settings_group, text="Attaccanti:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.att_var = tk.IntVar(value=6)
        ttk.Entry(settings_group, textvariable=self.att_var, width=10).grid(row=4, column=1, sticky=tk.W, padx=(5, 0))
        
        # Auction type
        ttk.Label(settings_group, text="Tipo asta:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.auction_type_var = tk.StringVar(value="chiamata")
        auction_type_combo = ttk.Combobox(settings_group, textvariable=self.auction_type_var, 
                                        values=["chiamata", "classica"], state="readonly", width=15)
        auction_type_combo.grid(row=5, column=1, sticky=tk.W, padx=(5, 0))
        
        # Per ruolo
        self.per_ruolo_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_group, text="Per ruolo", variable=self.per_ruolo_var).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # Agents Section
        agents_group = ttk.LabelFrame(scrollable_frame, text="Agenti Partecipanti", padding=10)
        agents_group.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add agent controls
        add_agent_frame = ttk.Frame(agents_group)
        add_agent_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(add_agent_frame, text="Nome:").pack(side=tk.LEFT)
        self.agent_name_var = tk.StringVar()
        ttk.Entry(add_agent_frame, textvariable=self.agent_name_var, width=15).pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Label(add_agent_frame, text="Tipo:").pack(side=tk.LEFT)
        self.agent_type_var = tk.StringVar(value="Human Agent")
        agent_type_combo = ttk.Combobox(add_agent_frame, textvariable=self.agent_type_var, 
                                      values=list(self.agent_types.keys()), state="readonly", width=15)
        agent_type_combo.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Button(add_agent_frame, text="Aggiungi Agente", command=self.add_agent).pack(side=tk.LEFT, padx=(5, 0))
        
        # Agents list
        self.agents_tree = ttk.Treeview(agents_group, columns=("Type",), show="tree headings", height=8)
        self.agents_tree.heading("#0", text="Nome Agente")
        self.agents_tree.heading("Type", text="Tipo")
        self.agents_tree.column("#0", width=200)
        self.agents_tree.column("Type", width=150)
        self.agents_tree.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Remove agent button
        ttk.Button(agents_group, text="Rimuovi Agente Selezionato", command=self.remove_agent).pack()
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add some default agents
        self.add_default_agents()
    
    def setup_auction_tab(self):
        """Setup the auction tab"""
        # Control buttons
        control_frame = ttk.Frame(self.auction_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_button = ttk.Button(control_frame, text="Avvia Asta", command=self.start_auction, style="Accent.TButton")
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="Ferma Asta", command=self.stop_auction, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Status
        self.status_var = tk.StringVar(value="Pronto per iniziare l'asta")
        ttk.Label(control_frame, textvariable=self.status_var, font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(20, 0))
        
        # Main content area with two sections
        content_frame = ttk.Frame(self.auction_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Left side - Auction info (60% width)
        left_frame = ttk.LabelFrame(content_frame, text="Informazioni Asta", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.auction_info_text = scrolledtext.ScrolledText(left_frame, height=20, state=tk.DISABLED)
        self.auction_info_text.pack(fill=tk.BOTH, expand=True)
        
        # Right side - Human agent input (40% width)
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # Human agent input section (initially hidden)
        self.human_input_frame = ttk.LabelFrame(right_frame, text="Chi vuole offrire?", padding=15)
        self.human_input_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initially hide the human input frame
        self.human_input_frame.pack_forget()
        
        # Setup human input UI
        self.setup_human_input_ui()
        
        # Placeholder frame when human input is not needed
        self.placeholder_frame = ttk.LabelFrame(right_frame, text="Asta in Corso", padding=15)
        self.placeholder_frame.pack(fill=tk.BOTH, expand=True)
        
        placeholder_text = tk.Text(self.placeholder_frame, height=20, state=tk.DISABLED, wrap=tk.WORD)
        placeholder_text.pack(fill=tk.BOTH, expand=True)
        placeholder_text.config(state=tk.NORMAL)
        placeholder_text.insert(tk.END, "🤖 Gli agenti automatici stanno partecipando all'asta...\n\n")
        placeholder_text.insert(tk.END, "Quando sarà il momento per gli agenti umani di fare offerte, ")
        placeholder_text.insert(tk.END, "apparirà qui il pannello di controllo.\n\n")
        placeholder_text.insert(tk.END, "Monitora il progresso dell'asta nella sezione 'Informazioni Asta'.\n\n")
        placeholder_text.insert(tk.END, "💡 Prendi tutto il tempo necessario per decidere!")
        placeholder_text.config(state=tk.DISABLED)
    
    def setup_human_input_ui(self):
        """Setup the human input UI components"""
        # Current player info
        self.current_player_frame = ttk.LabelFrame(self.human_input_frame, text="Giocatore in Asta", padding=10)
        self.current_player_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.player_info_label = ttk.Label(self.current_player_frame, text="", font=("Arial", 10, "bold"))
        self.player_info_label.pack(anchor=tk.W)
        
        self.auction_status_label = ttk.Label(self.current_player_frame, text="")
        self.auction_status_label.pack(anchor=tk.W)
        
        # Agent selection
        self.agent_selection_frame = ttk.LabelFrame(self.human_input_frame, text="Seleziona Agente e Offerta", padding=10)
        self.agent_selection_frame.pack(fill=tk.X, pady=(0, 10))
        
        # No offer option
        self.bid_choice = tk.StringVar(value="nessuno")
        
        self.no_offer_radio = ttk.Radiobutton(
            self.agent_selection_frame, 
            text="Nessuno vuole offrire", 
            variable=self.bid_choice, 
            value="nessuno"
        )
        self.no_offer_radio.pack(anchor=tk.W, pady=5)
        
        # Separator
        ttk.Separator(self.agent_selection_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Frame for agent options (will be populated dynamically)
        self.agents_options_frame = ttk.Frame(self.agent_selection_frame)
        self.agents_options_frame.pack(fill=tk.X)
        
        # Buttons
        button_frame = ttk.Frame(self.human_input_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.confirm_button = ttk.Button(button_frame, text="Conferma", command=self.on_human_confirm, style="Accent.TButton")
        self.confirm_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        self.no_offer_button = ttk.Button(button_frame, text="Nessuno offre", command=self.on_human_no_offer)
        self.no_offer_button.pack(side=tk.RIGHT)
        
        # Initialize variables for human input
        self.human_bid_amounts = {}
        self.human_decision_result = None
    
    def setup_logs_tab(self):
        """Setup the logs tab"""
        # Log controls
        log_control_frame = ttk.Frame(self.logs_frame)
        log_control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(log_control_frame, text="Pulisci Log", command=self.clear_logs).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(log_control_frame, text="Salva Log", command=self.save_logs).pack(side=tk.LEFT, padx=(0, 10))
        
        # Log display
        log_frame = ttk.LabelFrame(self.logs_frame, text="Log Asta", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=18, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def setup_logging(self):
        """Setup logging to capture auction logs"""
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
    
    def add_default_agents(self):
        """Add some default agents"""
        default_agents = [
            ("Angelo", "Human Agent"),
            ("Andrea", "Human Agent"),
            ("Bot Cap", "Cap Agent"),
            ("Bot Random", "Random Agent")
        ]
        
        for name, agent_type in default_agents:
            self.agents_config.append({"name": name, "type": agent_type})
            self.agents_tree.insert("", tk.END, text=name, values=(agent_type,))
    
    def add_agent(self):
        """Add a new agent to the configuration"""
        name = self.agent_name_var.get().strip()
        agent_type = self.agent_type_var.get()
        
        if not name:
            messagebox.showerror("Errore", "Inserisci un nome per l'agente")
            return
        
        # Check if name already exists
        if any(agent["name"] == name for agent in self.agents_config):
            messagebox.showerror("Errore", f"Un agente con nome '{name}' esiste già")
            return
        
        self.agents_config.append({"name": name, "type": agent_type})
        self.agents_tree.insert("", tk.END, text=name, values=(agent_type,))
        
        # Clear the input
        self.agent_name_var.set("")
    
    def remove_agent(self):
        """Remove selected agent"""
        selection = self.agents_tree.selection()
        if not selection:
            messagebox.showwarning("Attenzione", "Seleziona un agente da rimuovere")
            return
        
        item = selection[0]
        agent_name = self.agents_tree.item(item, "text")
        
        # Remove from config
        self.agents_config = [agent for agent in self.agents_config if agent["name"] != agent_name]
        
        # Remove from tree
        self.agents_tree.delete(item)
    
    def create_agents(self):
        """Create agent instances from configuration"""
        agents = []
        for agent_config in self.agents_config:
            agent_class = self.agent_types[agent_config["type"]]
            
            if agent_class == GUIHumanAgent:
                # Create human agent without individual callback (now using group selector)
                agent = GUIHumanAgent(agent_id=agent_config["name"])
            else:
                agent = agent_class(agent_id=agent_config["name"])
            
            agents.append(agent)
        return agents
    
    def start_auction(self):
        """Start the auction in a separate thread"""
        if len(self.agents_config) < 2:
            messagebox.showerror("Errore", "Servono almeno 2 agenti per l'asta")
            return
        
        # Update UI
        self.auction_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Asta in corso...")
        
        # Clear previous auction info
        self.auction_info_text.config(state=tk.NORMAL)
        self.auction_info_text.delete(1.0, tk.END)
        self.auction_info_text.config(state=tk.DISABLED)
        
        # Start auction in thread
        thread = threading.Thread(target=self.run_auction, daemon=True)
        thread.start()
    
    def run_auction(self):
        """Run the auction (called in separate thread)"""
        try:
            # Create agents
            agents = self.create_agents()
            
            # Get configuration
            initial_credits = self.initial_credits_var.get()
            slots = Slots(
                gk=self.gk_var.get(),
                def_=self.def_var.get(),
                mid=self.mid_var.get(),
                att=self.att_var.get()
            )
            
            # Load players
            listone = load_players_from_excel()
            
            # Create auction
            self.auction = Auction(slots, agents, listone, initial_credits)
            
            # Log configuration
            logger = logging.getLogger()
            logger.info("⚙️ CONFIGURAZIONE ASTA")
            logger.info("-" * 40)
            logger.info(f"📊 Slot per squadra:")
            logger.info(f"  GK: {slots.gk}")
            logger.info(f"  DEF: {slots.def_}")
            logger.info(f"  MID: {slots.mid}")
            logger.info(f"  ATT: {slots.att}")
            logger.info(f"💰 Crediti iniziali: {initial_credits}")
            logger.info(f"👥 Numero agenti: {len(agents)}")
            logger.info(f"📋 Giocatori disponibili: {len(listone)}")
            logger.info(f"🏁 ASTA INIZIATA - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("="*60)
            
            # Run auction
            auction_type = self.auction_type_var.get()
            per_ruolo = self.per_ruolo_var.get()
            
            # Run automated auction (no human interaction)
            self.run_automated_auction(auction_type, per_ruolo)
            
            logger.info("🏁 ASTA COMPLETATA!")
            logger.info(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("="*60)
            
        except Exception as e:
            logger = logging.getLogger()
            logger.error(f"❌ Errore durante l'asta: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        finally:
            # Update UI in main thread
            self.root.after(0, self.auction_finished)
    
    def run_automated_auction(self, auction_type, per_ruolo):
        """Run automated auction without user input"""
        import random
        logger = logging.getLogger()
        
        if auction_type == "chiamata":
            logger.info("🎯 Modalità: Asta a chiamata (automatizzata)")
            
            # Get all players and shuffle them
            available_players = self.auction.players.copy()
            if per_ruolo:
                # Group by role
                roles = ['GK', 'DEF', 'MID', 'ATT']
                for role in roles:
                    role_players = [p for p in available_players if p.role == role]
                    random.shuffle(role_players)
                    
                    logger.info(f"\n🔄 Inizio asta per {role}")
                    
                    for player in role_players:
                        if not self.auction_running:
                            return
                        self.auction_single_player_automated(player)
            else:
                random.shuffle(available_players)
                for player in available_players:
                    if not self.auction_running:
                        return
                    self.auction_single_player_automated(player)
        else:
            # Fallback to original method for other auction types
            logger.info("🎯 Modalità: Asta classica")
            self.auction.run_all(auction_type=auction_type, per_ruolo=per_ruolo, verbose=True)
    
    def auction_single_player_automated(self, player):
        """Auction a single player with support for human agents"""
        logger = logging.getLogger()
        
        logger.info(f"\n💫 GIOCATORE IN ASTA: {player.name} ({player.role}) - Valore: {player.evaluation}")
        
        # Set initial state
        self.auction.current_player = player
        self.auction.current_price = 1
        self.auction.highest_bidder = None
        
        # Run bidding rounds
        rounds_without_bids = 0
        max_rounds_without_bids = 3
        
        while rounds_without_bids < max_rounds_without_bids:
            if not self.auction_running:
                return
                
            logger.info(f"\n🔄 Round di offerte - Prezzo attuale: {self.auction.current_price}")
            
            # First, get offers from automatic agents
            offers = []
            for agent in self.auction.agents:
                if isinstance(agent, GUIHumanAgent):
                    continue  # Skip human agents for now
                    
                if self.auction.can_participate_in_bid(agent, self.auction.current_price + 1, 
                                                    player.role, self.auction.slots):
                    
                    decision = agent.make_offer_decision(
                        player, self.auction.current_price, self.auction.highest_bidder,
                        self.auction.players, self.auction.agents
                    )
                    
                    if decision != "no_offer":
                        if decision == "offer_+1":
                            offer_price = self.auction.current_price + 1
                        else:
                            try:
                                offer_price = int(decision.split("_")[1])
                            except:
                                offer_price = self.auction.current_price + 1
                        
                        offers.append((agent, offer_price))
                        logger.info(f"💰 {agent.agent_id} offre {offer_price} crediti")
                    else:
                        logger.info(f"🚫 {agent.agent_id} non offre")
            
            # Update current price if there were automatic offers
            if offers:
                highest_offer = max(offers, key=lambda x: x[1])
                agent, offer_price = highest_offer
                self.auction.current_price = offer_price
                self.auction.highest_bidder = agent
                logger.info(f"🏆 Offerta più alta automatica: {agent.agent_id} con {offer_price} crediti")
            
            # Now ask human agents if they want to bid
            human_agents = [agent for agent in self.auction.agents if isinstance(agent, GUIHumanAgent)]
            
            if human_agents:
                # Show human input in main GUI
                self.root.after(0, lambda: self.show_human_input(
                    player, human_agents, self.auction.current_price, self.auction.highest_bidder
                ))
                
                # Wait for human decision
                human_decision = self.wait_for_human_decision()
                
                if human_decision["agent_id"] != "nessuno":
                    # Human agent wants to bid
                    human_agent = next(agent for agent in human_agents if agent.agent_id == human_decision["agent_id"])
                    human_offer_price = human_decision["amount"]
                    
                    # Validate the human offer
                    if (human_offer_price > self.auction.current_price and 
                        self.auction.can_participate_in_bid(human_agent, human_offer_price, player.role, self.auction.slots)):
                        
                        self.auction.current_price = human_offer_price
                        self.auction.highest_bidder = human_agent
                        logger.info(f"💰 {human_agent.agent_id} offre {human_offer_price} crediti")
                        rounds_without_bids = 0
                    else:
                        logger.info(f"❌ Offerta di {human_agent.agent_id} non valida")
                        rounds_without_bids += 1
                else:
                    logger.info("🚫 Nessun agente umano vuole offrire")
                    if not offers:  # No automatic offers either
                        rounds_without_bids += 1
                    else:
                        rounds_without_bids = 0
            else:
                # No human agents, just check if there were automatic offers
                if not offers:
                    rounds_without_bids += 1
                else:
                    rounds_without_bids = 0
            
            if rounds_without_bids > 0:
                logger.info(f"🔕 Nessuna nuova offerta in questo round - {rounds_without_bids}/{max_rounds_without_bids}")
        
        # Assign player
        if self.auction.highest_bidder:
            self.auction.highest_bidder.current_credits -= self.auction.current_price
            self.auction.highest_bidder._squad.append(player)
            logger.info(f"✅ {player.name} assegnato a {self.auction.highest_bidder.agent_id} per {self.auction.current_price} crediti")
            logger.info(f"💳 {self.auction.highest_bidder.agent_id} - Crediti rimanenti: {self.auction.highest_bidder.current_credits}")
        else:
            logger.info(f"❌ {player.name} non assegnato - nessuna offerta")
    
    def show_human_input(self, player, human_agents, current_price, highest_bidder):
        """Show human input section in the main GUI"""
        # Update player info
        info_text = f"{player.name} ({player.role}) - Valore: {player.evaluation}"
        self.player_info_label.config(text=info_text)
        
        if highest_bidder:
            status_text = f"Offerta attuale: {current_price} crediti (da {highest_bidder.agent_id})"
        else:
            status_text = f"Prezzo base: {current_price} crediti"
        self.auction_status_label.config(text=status_text)
        
        # Clear previous agent options
        for widget in self.agents_options_frame.winfo_children():
            widget.destroy()
        
        self.human_bid_amounts.clear()
        
        # Add agent options
        agents_can_bid = False
        for agent in human_agents:
            # Calculate max possible bid for this agent
            credits_after_purchase = agent.current_credits - (current_price + 1)
            total_slots = sum(agent.slots.to_dict().values())
            total_players = len(agent.squad)
            remaining_slots = total_slots - (total_players + 1)
            max_bid = max(current_price + 1, agent.current_credits - remaining_slots) if remaining_slots >= 0 else current_price + 1
            max_bid = min(max_bid, agent.current_credits)
            
            # Skip if agent can't bid
            if max_bid <= current_price or remaining_slots < 0:
                continue
            
            agents_can_bid = True
            
            # Agent frame
            agent_frame = ttk.Frame(self.agents_options_frame)
            agent_frame.pack(fill=tk.X, pady=3)
            
            # Radio button for agent
            ttk.Radiobutton(
                agent_frame, 
                text=f"{agent.agent_id} (Crediti: {agent.current_credits})", 
                variable=self.bid_choice, 
                value=agent.agent_id
            ).pack(anchor=tk.W)
            
            # Bid amount controls
            bid_frame = ttk.Frame(agent_frame)
            bid_frame.pack(fill=tk.X, padx=(20, 0), pady=(5, 0))
            
            ttk.Label(bid_frame, text="Offerta:").pack(side=tk.LEFT)
            
            self.human_bid_amounts[agent.agent_id] = tk.IntVar(value=current_price + 1)
            bid_spinbox = ttk.Spinbox(
                bid_frame, 
                from_=current_price + 1, 
                to=max_bid,
                width=8,
                textvariable=self.human_bid_amounts[agent.agent_id]
            )
            bid_spinbox.pack(side=tk.LEFT, padx=(5, 0))
            
            ttk.Label(bid_frame, text=f"(max: {max_bid})").pack(side=tk.LEFT, padx=(5, 0))
        
        # Show message if no agents can bid
        if not agents_can_bid:
            no_agents_label = ttk.Label(
                self.agents_options_frame, 
                text="Nessun agente umano può fare offerte per questo giocatore.",
                font=("Arial", 9, "italic")
            )
            no_agents_label.pack(pady=10)
        
        # Switch to human input view
        self.placeholder_frame.pack_forget()
        self.human_input_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
    
    def hide_human_input(self):
        """Hide human input section and show placeholder"""
        self.human_input_frame.pack_forget()
        self.placeholder_frame.pack(fill=tk.BOTH, expand=True)
    
    def on_human_confirm(self):
        """Handle confirm button for human input"""
        choice = self.bid_choice.get()
        
        if choice == "nessuno":
            self.human_decision_result = {"agent_id": "nessuno", "amount": 0}
        else:
            amount = self.human_bid_amounts[choice].get()
            self.human_decision_result = {"agent_id": choice, "amount": amount}
    
    def on_human_no_offer(self):
        """Handle no offer button"""
        self.bid_choice.set("nessuno")
        self.human_decision_result = {"agent_id": "nessuno", "amount": 0}
    
    def wait_for_human_decision(self):
        """Wait for human decision in main thread"""
        self.human_decision_result = None
        
        # Wait until decision is made
        while self.human_decision_result is None:
            self.root.update()
            if not self.auction_running:
                return {"agent_id": "nessuno", "amount": 0}
        
        result = self.human_decision_result
        self.hide_human_input()
        return result

    def stop_auction(self):
        """Stop the current auction"""
        self.auction_running = False
        # Note: This is a simple implementation. A more robust version would 
        # need proper thread communication to gracefully stop the auction
        self.auction_finished()
    
    def auction_finished(self):
        """Called when auction is finished"""
        self.auction_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Asta completata")
    
    def check_log_queue(self):
        """Check for new log messages and display them"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                
                # Add to log tab
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, message + "\n")
                self.log_text.see(tk.END)
                self.log_text.config(state=tk.DISABLED)
                
                # Add to auction info if auction is running
                if self.auction_running:
                    self.auction_info_text.config(state=tk.NORMAL)
                    self.auction_info_text.insert(tk.END, message + "\n")
                    self.auction_info_text.see(tk.END)
                    self.auction_info_text.config(state=tk.DISABLED)
                
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_log_queue)
    
    def clear_logs(self):
        """Clear the log display"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def save_logs(self):
        """Save current logs to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialname=f"auction_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    logs = self.log_text.get(1.0, tk.END)
                    f.write(logs)
                messagebox.showinfo("Successo", f"Log salvati in: {filename}")
            except Exception as e:
                messagebox.showerror("Errore", f"Errore nel salvare i log: {str(e)}")


def main():
    root = tk.Tk()
    app = FantaAuctionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
