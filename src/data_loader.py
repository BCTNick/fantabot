"""
Data loading utilities
"""

import pandas as pd
import os
from models import Player, calculate_standardized_and_ranking


def load_players_from_excel(file_path: str = None):
    """Load players from Excel file"""
    if file_path is None:
        # Get the directory where this file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the project root and then to data folder
        project_root = os.path.dirname(current_dir)
        file_path = os.path.join(project_root, "data", "players_list.xlsx")
    
    df = pd.read_excel(file_path)

    players = []
    for _, row in df.iterrows():
        player = Player(
            name=row['name'],
            team=row['team'], 
            role=row['role'],
            evaluation=row['evaluation']
        )
        players.append(player)

    return calculate_standardized_and_ranking(players)
