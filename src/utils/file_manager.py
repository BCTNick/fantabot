"""
File handling utilities for auction system
"""

import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path


class FileManager:
    """File operations for auction system"""
    
    @staticmethod
    def save_auction_state(agents_config: List[Dict], auction_settings: Dict, 
                          filename: Optional[str] = None) -> str:
        """Save current auction configuration to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"auction_config_{timestamp}.json"
        
        config_data = {
            "timestamp": datetime.now().isoformat(),
            "agents": agents_config,
            "settings": auction_settings
        }
        
        filepath = Path("logs") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    @staticmethod
    def load_auction_state(filepath: str) -> Dict:
        """Load auction configuration from JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_auction_results(agents: List, filename: Optional[str] = None) -> str:
        """Save auction results to Excel"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"auction_results_{timestamp}.xlsx"
        
        filepath = Path("logs") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        # Create workbook with multiple sheets
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for agent in agents:
                summary_data.append({
                    "Agente": agent.agent_id,
                    "Crediti_Rimanenti": agent.current_credits,
                    "Giocatori_Totali": len(agent.squad),
                    "Portieri": len(agent.squad.gk),
                    "Difensori": len(agent.squad.def_),
                    "Centrocampisti": len(agent.squad.mid),
                    "Attaccanti": len(agent.squad.att),
                    "Valutazione_Totale": agent.squad.objective(standardized=False),
                    "Best_XI": agent.squad.objective(bestxi=True, standardized=False)
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Riassunto", index=False)
            
            # Detailed sheets for each agent
            for agent in agents:
                if len(agent.squad) > 0:
                    squad_data = []
                    for player in agent.squad:
                        squad_data.append({
                            "Nome": player.name,
                            "Squadra": player.team,
                            "Ruolo": player.role,
                            "Valutazione": player.evaluation,
                            "Costo_Finale": player.final_cost or 0
                        })
                    
                    squad_df = pd.DataFrame(squad_data)
                    # Excel sheet names have character limits
                    sheet_name = agent.agent_id[:31] if len(agent.agent_id) > 31 else agent.agent_id
                    squad_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        return str(filepath)
    
    @staticmethod
    def save_logs_to_file(log_content: str, filename: Optional[str] = None) -> str:
        """Save log content to text file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"auction_log_{timestamp}.txt"
        
        filepath = Path("logs") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(log_content)
        
        return str(filepath)
    
    @staticmethod
    def get_available_configs() -> List[str]:
        """Get list of available configuration files"""
        logs_dir = Path("logs")
        if not logs_dir.exists():
            return []
        
        config_files = list(logs_dir.glob("auction_config_*.json"))
        return [f.name for f in sorted(config_files, reverse=True)]
    
    @staticmethod
    def backup_current_data() -> str:
        """Create backup of current data directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}"
        
        import shutil
        shutil.copytree("data", f"logs/{backup_name}")
        
        return backup_name
