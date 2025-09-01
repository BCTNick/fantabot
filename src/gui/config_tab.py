"""
Configuration tab with enhanced features
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from typing import List, Dict, Callable, Optional
from src.utils.validators import AuctionValidator
from src.utils.file_manager import FileManager


class ConfigurationTab:
    """Enhanced configuration tab with validation and save/load features"""
    
    def __init__(self, parent, agent_types: Dict[str, type], 
                 on_config_changed: Optional[Callable] = None):
        self.parent = parent
        self.agent_types = agent_types
        self.on_config_changed = on_config_changed
        
        # Configuration data
        self.agents_config = []
        
        # UI variables
        self.initial_credits_var = tk.IntVar(value=1000)
        self.gk_var = tk.IntVar(value=3)
        self.def_var = tk.IntVar(value=8)
        self.mid_var = tk.IntVar(value=8)
        self.att_var = tk.IntVar(value=6)
        self.auction_type_var = tk.StringVar(value="chiamata")
        self.per_ruolo_var = tk.BooleanVar(value=True)
        self.agent_name_var = tk.StringVar()
        self.agent_type_var = tk.StringVar(value="Human Agent")
        
        self.setup_ui()
        self.add_default_agents()
    
    def setup_ui(self):
        """Setup configuration tab UI"""
        # Main scrollable container
        canvas = tk.Canvas(self.parent)
        scrollbar = ttk.Scrollbar(self.parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Configuration sections
        self.setup_auction_settings(scrollable_frame)
        self.setup_agents_section(scrollable_frame)
        self.setup_config_management(scrollable_frame)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def setup_auction_settings(self, parent):
        """Setup auction settings section"""
        settings_group = ttk.LabelFrame(parent, text="⚙️ Impostazioni Asta", padding=15)
        settings_group.pack(fill=tk.X, padx=10, pady=10)
        
        # Create grid layout
        # Row 0: Credits
        ttk.Label(settings_group, text="💰 Crediti iniziali:").grid(
            row=0, column=0, sticky=tk.W, pady=5, padx=(0, 10)
        )
        credits_spinbox = ttk.Spinbox(
            settings_group, textvariable=self.initial_credits_var, 
            from_=100, to=10000, increment=50, width=10
        )
        credits_spinbox.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Row 1-4: Squad composition
        positions = [
            ("🥅 Portieri:", self.gk_var, 1, 5),
            ("🛡️ Difensori:", self.def_var, 3, 15),
            ("⚽ Centrocampisti:", self.mid_var, 3, 15),
            ("🎯 Attaccanti:", self.att_var, 2, 12)
        ]
        
        for i, (label, var, min_val, max_val) in enumerate(positions, 1):
            ttk.Label(settings_group, text=label).grid(
                row=i, column=0, sticky=tk.W, pady=5, padx=(0, 10)
            )
            ttk.Spinbox(
                settings_group, textvariable=var, 
                from_=min_val, to=max_val, width=10
            ).grid(row=i, column=1, sticky=tk.W, pady=5)
        
        # Row 5: Auction type
        ttk.Label(settings_group, text="🎯 Tipo asta:").grid(
            row=5, column=0, sticky=tk.W, pady=5, padx=(0, 10)
        )
        auction_type_combo = ttk.Combobox(
            settings_group, textvariable=self.auction_type_var,
            values=["chiamata", "classica"], state="readonly", width=15
        )
        auction_type_combo.grid(row=5, column=1, sticky=tk.W, pady=5)
        
        # Row 6: Per ruolo checkbox
        ttk.Checkbutton(
            settings_group, text="📋 Asta per ruolo", 
            variable=self.per_ruolo_var
        ).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Validation button
        ttk.Button(
            settings_group, text="✅ Valida Configurazione",
            command=self._validate_settings
        ).grid(row=7, column=0, columnspan=2, pady=10)
    
    def setup_agents_section(self, parent):
        """Setup agents configuration section"""
        agents_group = ttk.LabelFrame(parent, text="👥 Agenti Partecipanti", padding=15)
        agents_group.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add agent controls
        add_frame = ttk.Frame(agents_group)
        add_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Agent input row
        input_row = ttk.Frame(add_frame)
        input_row.pack(fill=tk.X)
        
        ttk.Label(input_row, text="Nome:").pack(side=tk.LEFT)
        name_entry = ttk.Entry(input_row, textvariable=self.agent_name_var, width=15)
        name_entry.pack(side=tk.LEFT, padx=(5, 15))
        
        ttk.Label(input_row, text="Tipo:").pack(side=tk.LEFT)
        type_combo = ttk.Combobox(
            input_row, textvariable=self.agent_type_var,
            values=list(self.agent_types.keys()), state="readonly", width=18
        )
        type_combo.pack(side=tk.LEFT, padx=(5, 15))
        
        ttk.Button(
            input_row, text="➕ Aggiungi", command=self._add_agent
        ).pack(side=tk.LEFT, padx=(5, 0))
        
        # Agents list with enhanced view
        list_frame = ttk.Frame(agents_group)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview with more columns
        self.agents_tree = ttk.Treeview(
            list_frame, 
            columns=("Type", "TTS", "Credits"),
            show="tree headings", 
            height=10
        )
        
        # Configure columns
        self.agents_tree.heading("#0", text="👤 Nome Agente")
        self.agents_tree.heading("Type", text="🤖 Tipo")
        self.agents_tree.heading("TTS", text="🔊 TTS")
        self.agents_tree.heading("Credits", text="💰 Budget")
        
        self.agents_tree.column("#0", width=120)
        self.agents_tree.column("Type", width=140)
        self.agents_tree.column("TTS", width=50)
        self.agents_tree.column("Credits", width=80)
        
        # Scrollbar for treeview
        tree_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.agents_tree.yview)
        self.agents_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.agents_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        
        # Control buttons
        controls_frame = ttk.Frame(agents_group)
        controls_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(
            controls_frame, text="🗑️ Rimuovi Selezionato", 
            command=self._remove_agent
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            controls_frame, text="🔊 Toggle TTS", 
            command=self._toggle_tts
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            controls_frame, text="📊 Mostra Statistiche", 
            command=self._show_agent_stats
        ).pack(side=tk.LEFT)
    
    def setup_config_management(self, parent):
        """Setup configuration save/load section"""
        config_group = ttk.LabelFrame(parent, text="💾 Gestione Configurazioni", padding=15)
        config_group.pack(fill=tk.X, padx=10, pady=10)
        
        # Save/Load buttons
        button_frame = ttk.Frame(config_group)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(
            button_frame, text="💾 Salva Configurazione", 
            command=self._save_config
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame, text="📁 Carica Configurazione", 
            command=self._load_config
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame, text="🔄 Reset Default", 
            command=self._reset_to_default
        ).pack(side=tk.LEFT)
        
        # Quick presets
        presets_frame = ttk.LabelFrame(config_group, text="⚡ Preset Rapidi", padding=10)
        presets_frame.pack(fill=tk.X, pady=(10, 0))
        
        preset_buttons = ttk.Frame(presets_frame)
        preset_buttons.pack(fill=tk.X)
        
        ttk.Button(
            preset_buttons, text="🏠 Casa (4 giocatori)",
            command=lambda: self._apply_preset("casa")
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            preset_buttons, text="🏢 Ufficio (8 giocatori)",
            command=lambda: self._apply_preset("ufficio")
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            preset_buttons, text="🎮 Test (2 giocatori)",
            command=lambda: self._apply_preset("test")
        ).pack(side=tk.LEFT)
    
    def add_default_agents(self):
        """Add default agents"""
        default_agents = [
            ("Angelo", "Human Agent", True),
            ("Andrea", "Human Agent", True),
            ("Cap Agent", "Cap Agent", True),  # Abilitato TTS per sentire le offerte
            ("Random Bot", "Random Agent", True)  # Abilitato TTS per sentire le offerte
        ]
        
        for name, agent_type, tts_enabled in default_agents:
            self._add_agent_to_config(name, agent_type, tts_enabled)
    
    def _add_agent_to_config(self, name: str, agent_type: str, tts_enabled: bool):
        """Add agent to configuration"""
        credits = self.initial_credits_var.get()
        self.agents_config.append({
            "name": name, 
            "type": agent_type, 
            "tts_enabled": tts_enabled,
            "initial_credits": credits
        })
        
        # Update tree view
        tts_text = "✓" if tts_enabled else "✗"
        self.agents_tree.insert(
            "", tk.END, 
            text=name, 
            values=(agent_type, tts_text, f"{credits}€")
        )
        
        if self.on_config_changed:
            self.on_config_changed()
    
    def get_configuration(self) -> Dict:
        """Get current configuration"""
        is_valid, error_msg = self._validate_current_config()
        
        return {
            "valid": is_valid,
            "error": error_msg,
            "settings": {
                "initial_credits": self.initial_credits_var.get(),
                "slots": {
                    "gk": self.gk_var.get(),
                    "def_": self.def_var.get(),
                    "mid": self.mid_var.get(),
                    "att": self.att_var.get()
                },
                "auction_type": self.auction_type_var.get(),
                "per_ruolo": self.per_ruolo_var.get()
            },
            "agents": self.agents_config.copy()
        }
    
    def get_agents_config(self) -> List[Dict]:
        """Get agents configuration for TTS and other features"""
        return self.agents_config.copy()
    
    # Event handlers
    def _add_agent(self):
        """Add new agent"""
        name = self.agent_name_var.get().strip()
        agent_type = self.agent_type_var.get()
        
        if not name:
            messagebox.showerror("Errore", "Inserisci un nome per l'agente")
            return
        
        if any(agent["name"] == name for agent in self.agents_config):
            messagebox.showerror("Errore", f"Un agente con nome '{name}' esiste già")
            return
        
        # Abilita TTS per tutti i tipi di agenti di default
        tts_enabled = True  # Cambiato da (agent_type == "Human Agent")
        self._add_agent_to_config(name, agent_type, tts_enabled)
        self.agent_name_var.set("")
    
    def _remove_agent(self):
        """Remove selected agent"""
        selection = self.agents_tree.selection()
        if not selection:
            messagebox.showwarning("Attenzione", "Seleziona un agente da rimuovere")
            return
        
        item = selection[0]
        agent_name = self.agents_tree.item(item, "text")
        
        # Remove from config
        self.agents_config = [a for a in self.agents_config if a["name"] != agent_name]
        self.agents_tree.delete(item)
        
        if self.on_config_changed:
            self.on_config_changed()
    
    def _toggle_tts(self):
        """Toggle TTS for selected agent"""
        selection = self.agents_tree.selection()
        if not selection:
            messagebox.showwarning("Attenzione", "Seleziona un agente per modificare il TTS")
            return
        
        item = selection[0]
        agent_name = self.agents_tree.item(item, "text")
        
        for agent in self.agents_config:
            if agent["name"] == agent_name:
                agent["tts_enabled"] = not agent["tts_enabled"]
                
                # Update tree
                current_values = list(self.agents_tree.item(item, "values"))
                current_values[1] = "✓" if agent["tts_enabled"] else "✗"
                self.agents_tree.item(item, values=current_values)
                break
        
        if self.on_config_changed:
            self.on_config_changed()
    
    def _validate_settings(self):
        """Validate current settings"""
        is_valid, error_msg = self._validate_current_config()
        
        if is_valid:
            messagebox.showinfo("✅ Validazione", "Configurazione valida!")
        else:
            messagebox.showerror("❌ Errore di Validazione", error_msg)
    
    def _validate_current_config(self) -> tuple[bool, str]:
        """Validate current configuration"""
        # Validate agents
        is_valid, error = AuctionValidator.validate_agent_config(self.agents_config)
        if not is_valid:
            return False, error
        
        # Validate settings
        from src.models import Slots
        slots = Slots(
            gk=self.gk_var.get(),
            def_=self.def_var.get(),
            mid=self.mid_var.get(),
            att=self.att_var.get()
        )
        
        is_valid, error = AuctionValidator.validate_auction_settings(
            self.initial_credits_var.get(), slots
        )
        
        return is_valid, error
    
    def _save_config(self):
        """Save current configuration"""
        try:
            config = self.get_configuration()
            if not config["valid"]:
                messagebox.showerror("Errore", f"Configurazione non valida: {config['error']}")
                return
            
            filepath = FileManager.save_auction_state(
                config["agents"], config["settings"]
            )
            messagebox.showinfo("Successo", f"Configurazione salvata in:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Errore", f"Errore nel salvare la configurazione:\n{str(e)}")
    
    def _load_config(self):
        """Load configuration from file"""
        filename = filedialog.askopenfilename(
            title="Carica Configurazione",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="logs"
        )
        
        if filename:
            try:
                config_data = FileManager.load_auction_state(filename)
                self._apply_loaded_config(config_data)
                messagebox.showinfo("Successo", "Configurazione caricata con successo!")
            except Exception as e:
                messagebox.showerror("Errore", f"Errore nel caricare la configurazione:\n{str(e)}")
    
    def _apply_loaded_config(self, config_data: Dict):
        """Apply loaded configuration"""
        # Apply settings
        settings = config_data.get("settings", {})
        self.initial_credits_var.set(settings.get("initial_credits", 1000))
        
        slots = settings.get("slots", {})
        self.gk_var.set(slots.get("gk", 3))
        self.def_var.set(slots.get("def_", 8))
        self.mid_var.set(slots.get("mid", 8))
        self.att_var.set(slots.get("att", 6))
        
        self.auction_type_var.set(settings.get("auction_type", "chiamata"))
        self.per_ruolo_var.set(settings.get("per_ruolo", True))
        
        # Clear and apply agents
        self.agents_config.clear()
        for item in self.agents_tree.get_children():
            self.agents_tree.delete(item)
        
        for agent_data in config_data.get("agents", []):
            self._add_agent_to_config(
                agent_data["name"],
                agent_data["type"],
                agent_data.get("tts_enabled", False)
            )
    
    def _reset_to_default(self):
        """Reset to default configuration"""
        result = messagebox.askyesno(
            "Conferma Reset", 
            "Sei sicuro di voler ripristinare la configurazione predefinita?\n"
            "Tutte le modifiche non salvate andranno perse."
        )
        
        if result:
            # Reset all variables
            self.initial_credits_var.set(1000)
            self.gk_var.set(3)
            self.def_var.set(8)
            self.mid_var.set(8)
            self.att_var.set(6)
            self.auction_type_var.set("chiamata")
            self.per_ruolo_var.set(True)
            
            # Clear agents
            self.agents_config.clear()
            for item in self.agents_tree.get_children():
                self.agents_tree.delete(item)
            
            # Add defaults
            self.add_default_agents()
    
    def _apply_preset(self, preset_type: str):
        """Apply a preset configuration"""
        presets = {
            "casa": {
                "agents": [
                    ("Giocatore 1", "Human Agent", True),
                    ("Giocatore 2", "Human Agent", True),
                    ("Giocatore 3", "Human Agent", True),
                    ("Giocatore 4", "Human Agent", True)
                ],
                "credits": 1000
            },
            "ufficio": {
                "agents": [
                    ("Team 1", "Human Agent", True),
                    ("Team 2", "Human Agent", True),
                    ("Team 3", "Human Agent", True),
                    ("Team 4", "Human Agent", True),
                    ("Bot Cap 1", "Cap Agent", False),
                    ("Bot Cap 2", "Cap Agent", False),
                    ("Bot Random 1", "Random Agent", False),
                    ("Bot Random 2", "Random Agent", False)
                ],
                "credits": 1200
            },
            "test": {
                "agents": [
                    ("Test Human", "Human Agent", True),
                    ("Test Bot", "Cap Agent", False)
                ],
                "credits": 500
            }
        }
        
        if preset_type in presets:
            preset = presets[preset_type]
            
            # Clear current agents
            self.agents_config.clear()
            for item in self.agents_tree.get_children():
                self.agents_tree.delete(item)
            
            # Set credits
            self.initial_credits_var.set(preset["credits"])
            
            # Add preset agents
            for name, agent_type, tts in preset["agents"]:
                self._add_agent_to_config(name, agent_type, tts)
    
    def _show_agent_stats(self):
        """Show agents statistics"""
        if not self.agents_config:
            messagebox.showinfo("Info", "Nessun agente configurato")
            return
        
        stats_window = tk.Toplevel(self.parent)
        stats_window.title("📊 Statistiche Agenti")
        stats_window.geometry("400x300")
        
        stats_text = scrolledtext.ScrolledText(stats_window, wrap=tk.WORD)
        stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Generate stats
        total_agents = len(self.agents_config)
        human_agents = len([a for a in self.agents_config if a["type"] == "Human Agent"])
        bot_agents = total_agents - human_agents
        tts_enabled = len([a for a in self.agents_config if a.get("tts_enabled", False)])
        
        stats = f"""📊 STATISTICHE AGENTI
        
Totale agenti: {total_agents}
Agenti umani: {human_agents}
Agenti bot: {bot_agents}
TTS abilitato: {tts_enabled}

📋 DETTAGLI:
"""
        
        for agent in self.agents_config:
            tts_status = "🔊" if agent.get("tts_enabled", False) else "🔇"
            stats += f"\n• {agent['name']} ({agent['type']}) {tts_status}"
        
        stats_text.insert(tk.END, stats)
        stats_text.config(state=tk.DISABLED)
