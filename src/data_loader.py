"""
Data loading utilities
"""

import pandas as pd
from src.models import Player, calculate_standardized_and_ranking


def load_players_from_excel(file_path: str = "data/players_list.xlsx"):
    """Load players from Excel file"""
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
