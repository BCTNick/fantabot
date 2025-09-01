"""
Teams management tab for viewing team compositions and purchases
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Any
from src.models import Player


class TeamsTab:
    """Tab for managing and viewing team compositions"""
    
    def __init__(self, parent_frame):
        self.frame = parent_frame
        self.agents = []
        self.players = []
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the teams tab UI"""
        # Main container with padding
        main_frame = ttk.Frame(self.frame, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="🏆 COMPOSIZIONE SQUADRE", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Statistics frame
        self.stats_frame = ttk.LabelFrame(main_frame, text="📊 Statistiche Generali", padding="10")
        self.stats_frame.pack(fill="x", pady=(0, 10))
        
        self.stats_text = tk.Text(self.stats_frame, height=4, wrap="word", 
                                 font=("Consolas", 10), bg="#f8f9fa")
        self.stats_text.pack(fill="x")
        
        # Teams container
        teams_container = ttk.Frame(main_frame)
        teams_container.pack(fill="both", expand=True)
        
        # Scrollable frame for teams
        canvas = tk.Canvas(teams_container)
        scrollbar = ttk.Scrollbar(teams_container, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Refresh button
        refresh_frame = ttk.Frame(main_frame)
        refresh_frame.pack(fill="x", pady=(10, 0))
        
        self.refresh_button = ttk.Button(refresh_frame, text="🔄 Aggiorna", 
                                        command=self.refresh_teams)
        self.refresh_button.pack(side="right")
        
        # Initialize with empty state
        self.show_empty_state()
    
    def show_empty_state(self):
        """Show empty state when no auction data is available"""
        # Clear existing content
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Empty state message
        empty_frame = ttk.Frame(self.scrollable_frame)
        empty_frame.pack(fill="both", expand=True, pady=50)
        
        empty_label = ttk.Label(empty_frame, 
                               text="📋 Nessuna asta avviata\n\nConfigura e avvia un'asta per vedere le squadre qui.",
                               font=("Arial", 12), 
                               justify="center",
                               foreground="#6c757d")
        empty_label.pack(expand=True)
        
        # Clear stats
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, "Avvia un'asta per vedere le statistiche delle squadre.")
    
    def set_auction_data(self, agents: List, players: List[Player]):
        """Set auction data for the teams display"""
        self.agents = agents
        self.players = players
        self.refresh_teams()
    
    def refresh_teams(self):
        """Refresh the teams display"""
        if not self.agents:
            self.show_empty_state()
            return
        
        # Clear existing content
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Update statistics
        self.update_statistics()
        
        # Create team frames
        teams_per_row = 2
        row_frame = None
        
        for i, agent in enumerate(self.agents):
            # Create new row every 2 teams
            if i % teams_per_row == 0:
                row_frame = ttk.Frame(self.scrollable_frame)
                row_frame.pack(fill="x", pady=5)
            
            # Create team frame
            team_frame = self.create_team_frame(row_frame, agent)
            team_frame.pack(side="left", fill="both", expand=True, padx=5)
    
    def create_team_frame(self, parent, agent) -> ttk.Frame:
        """Create a frame for displaying a single team"""
        # Team container
        team_container = ttk.LabelFrame(parent, text=f"🎯 {agent.agent_id}", padding="10")
        
        # Team header with stats
        header_frame = ttk.Frame(team_container)
        header_frame.pack(fill="x", pady=(0, 10))
        
        # Credits info
        credits_label = ttk.Label(header_frame, 
                                 text=f"💰 Crediti: {agent.current_credits:,}",
                                 font=("Arial", 10, "bold"))
        credits_label.pack(side="left")
        
        # Squad size
        squad_size = len(getattr(agent, '_squad', []))
        size_label = ttk.Label(header_frame, 
                              text=f"👥 Giocatori: {squad_size}",
                              font=("Arial", 10))
        size_label.pack(side="right")
        
        # Players list
        if hasattr(agent, '_squad') and agent._squad:
            # Create treeview for players
            columns = ("Nome", "Ruolo", "Costo", "Valutazione")
            tree = ttk.Treeview(team_container, columns=columns, show="headings", height=8)
            
            # Configure columns
            tree.heading("Nome", text="👤 Nome")
            tree.heading("Ruolo", text="⚽ Ruolo")
            tree.heading("Costo", text="💰 Costo")
            tree.heading("Valutazione", text="⭐ Val.")
            
            tree.column("Nome", width=120, minwidth=100)
            tree.column("Ruolo", width=60, minwidth=50)
            tree.column("Costo", width=70, minwidth=60)
            tree.column("Valutazione", width=70, minwidth=60)
            
            # Add players to tree
            total_spent = 0
            role_counts = {"GK": 0, "DEF": 0, "MID": 0, "ATT": 0}
            
            # Sort players by role and cost
            sorted_players = sorted(agent._squad, 
                                  key=lambda p: (self._role_priority(p.role), -p.final_cost))
            
            for player in sorted_players:
                cost = getattr(player, 'final_cost', 0)
                total_spent += cost
                role_counts[player.role] = role_counts.get(player.role, 0) + 1
                
                # Color coding by role
                tags = self._get_role_tags(player.role)
                
                tree.insert("", "end", 
                           values=(player.name, player.role, f"{cost:,}", player.evaluation),
                           tags=tags)
            
            # Configure role colors
            tree.tag_configure("portiere", background="#e3f2fd")
            tree.tag_configure("difensore", background="#f3e5f5") 
            tree.tag_configure("centrocampista", background="#e8f5e8")
            tree.tag_configure("attaccante", background="#fff3e0")
            
            tree.pack(fill="both", expand=True, pady=(0, 10))
            
            # Team summary
            summary_frame = ttk.Frame(team_container)
            summary_frame.pack(fill="x")
            
            summary_text = (f"💸 Speso: {total_spent:,} | "
                           f"GK:{role_counts.get('GK', 0)} DEF:{role_counts.get('DEF', 0)} "
                           f"MID:{role_counts.get('MID', 0)} ATT:{role_counts.get('ATT', 0)}")
            
            summary_label = ttk.Label(summary_frame, text=summary_text, 
                                     font=("Arial", 9, "italic"))
            summary_label.pack()
            
        else:
            # No players yet
            no_players_label = ttk.Label(team_container, 
                                        text="📋 Nessun giocatore acquistato",
                                        font=("Arial", 10, "italic"),
                                        foreground="#6c757d")
            no_players_label.pack(pady=20)
        
        return team_container
    
    def _role_priority(self, role: str) -> int:
        """Get priority for role sorting"""
        priorities = {"GK": 1, "DEF": 2, "MID": 3, "ATT": 4}
        return priorities.get(role, 5)
    
    def _get_role_tags(self, role: str) -> tuple:
        """Get tags for role-based styling"""
        role_tags = {
            "GK": ("portiere",),
            "DEF": ("difensore",),
            "MID": ("centrocampista",),
            "ATT": ("attaccante",)
        }
        return role_tags.get(role, ())
    
    def update_statistics(self):
        """Update general statistics"""
        self.stats_text.delete(1.0, tk.END)
        
        if not self.agents:
            self.stats_text.insert(tk.END, "Nessun dato disponibile")
            return
        
        # Calculate statistics
        total_players_sold = 0
        total_money_spent = 0
        avg_credits_remaining = 0
        
        team_stats = []
        
        for agent in self.agents:
            squad_size = len(getattr(agent, '_squad', []))
            total_players_sold += squad_size
            
            spent = 0
            if hasattr(agent, '_squad'):
                spent = sum(getattr(p, 'final_cost', 0) for p in agent._squad)
            
            total_money_spent += spent
            
            team_stats.append({
                'name': agent.agent_id,
                'players': squad_size,
                'spent': spent,
                'remaining': agent.current_credits
            })
        
        avg_credits_remaining = sum(agent.current_credits for agent in self.agents) / len(self.agents)
        
        # Display statistics
        stats_text = f"""📊 STATISTICHE GENERALI:
• Giocatori venduti: {total_players_sold}
• Denaro speso totale: {total_money_spent:,} crediti
• Crediti medi rimanenti: {avg_credits_remaining:,.0f}
• Prezzo medio per giocatore: {(total_money_spent / total_players_sold):,.0f} crediti""" if total_players_sold > 0 else "Nessun giocatore ancora venduto"
        
        self.stats_text.insert(tk.END, stats_text)
    
    def get_team_by_name(self, team_name: str) -> Optional[Any]:
        """Get team data by name"""
        for agent in self.agents:
            if agent.agent_id == team_name:
                return agent
        return None
    
    def export_teams_data(self) -> Dict:
        """Export teams data for analysis"""
        if not self.agents:
            return {}
        
        teams_data = {}
        
        for agent in self.agents:
            team_data = {
                'name': agent.agent_id,
                'credits_remaining': agent.current_credits,
                'players': [],
                'total_spent': 0,
                'squad_size': 0
            }
            
            if hasattr(agent, '_squad') and agent._squad:
                for player in agent._squad:
                    player_data = {
                        'name': player.name,
                        'role': player.role,
                        'cost': getattr(player, 'final_cost', 0),
                        'evaluation': player.evaluation
                    }
                    team_data['players'].append(player_data)
                    team_data['total_spent'] += player_data['cost']
                
                team_data['squad_size'] = len(agent._squad)
            
            teams_data[agent.agent_id] = team_data
        
        return teams_data
