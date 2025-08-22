"""
Player and Squad models
"""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Player:
    """Represents a football player in the fantasy auction"""
    # Initial data (from excel file)
    name: str
    team: str
    role: str  # 'GK', 'DEF', 'MID', 'ATT'
    evaluation: int

    # Calculated after the importing of the file
    standardized_evaluation: float = 0.0  # Between 0 and 1
    ranking: int = 0  # Rank based on evaluation
    
    # Final results (after auction)
    fantasy_team: Optional[str] = None  # Which agent bought the player
    final_cost: int = None  # Final price paid


class Squad:
    """Represents a team's squad of players"""
    
    def __init__(self, players: List[Player]):
        self._players = players

    def __iter__(self):
        return iter(self._players)

    def __len__(self):
        return len(self._players)

    def __getitem__(self, idx):
        return self._players[idx]

    @property
    def gk(self):
        return [p for p in self._players if p.role == "GK"]

    @property
    def def_(self):
        return [p for p in self._players if p.role == "DEF"]

    @property
    def mid(self):
        return [p for p in self._players if p.role == "MID"]

    @property
    def att(self):
        return [p for p in self._players if p.role == "ATT"]
    
    def objective(self, bestxi: bool = False, standardized: bool = True) -> float:
        """Calculate objective function for this squad"""
        eval_attr = 'standardized_evaluation' if standardized else 'evaluation'
        
        if bestxi:
            # Best starting XI: 1 GK + 3 DEF + 3 MID + 3 ATT
            best_gk = max(self.gk, key=lambda p: getattr(p, eval_attr), default=None)
            best_def = sorted(self.def_, key=lambda p: getattr(p, eval_attr), reverse=True)[:3]
            best_mid = sorted(self.mid, key=lambda p: getattr(p, eval_attr), reverse=True)[:3]
            best_att = sorted(self.att, key=lambda p: getattr(p, eval_attr), reverse=True)[:3]
            
            total = 0
            if best_gk:
                total += getattr(best_gk, eval_attr)
                
            total += sum(getattr(p, eval_attr) for p in best_def + best_mid + best_att)
            
            return total
        else:
            # Total squad evaluation
            return sum(getattr(p, eval_attr) for p in self._players)
    
    def get_remaining_slots(self, slots: 'Slots', position: str = None) -> dict | int:

        current_counts = {
            'gk': len(self.gk),
            'def': len(self.def_),
            'mid': len(self.mid),
            'att': len(self.att)
        }
        
        remaining = {
            'gk': slots.gk - current_counts['gk'],
            'def': slots.def_ - current_counts['def'],
            'mid': slots.mid - current_counts['mid'],
            'att': slots.att - current_counts['att']
        }
        
        # Add total remaining slots
        remaining['total'] = sum(remaining.values())
        
        # If specific position requested, return just that value
        if position is not None:
            position = position.lower()
            if position in remaining:
                return remaining[position]
            else:
                raise ValueError(f"Invalid position '{position}'. Valid options: 'gk', 'def', 'mid', 'att', 'total'")
        
        return remaining


class Slots:    
    def __init__(self, gk: int = 3, def_: int = 8, mid: int = 8, att: int = 6):
        self.gk = gk
        self.def_ = def_
        self.mid = mid
        self.att = att
        
    def to_dict(self):
        return {'GK': self.gk, 'DEF': self.def_, 'MID': self.mid, 'ATT': self.att}

    def get_numbers(self, position: str = None) -> dict | int:

        number = {
            'gk': self.gk,
            'def': self.def_,
            'mid': self.mid,
            'att': self.att
        }
        
        # Add total remaining slots
        number['total'] = sum(number.values())
        
        # If specific position requested, return just that value
        if position is not None:
            position = position.lower()
            if position in number:
                return number[position]
            else:
                raise ValueError(f"Invalid position '{position}'. Valid options: 'gk', 'def', 'mid', 'att', 'total'")

        return number


def calculate_standardized_and_ranking(players: List[Player]) -> List[Player]:
    """Calculate standardized evaluation and ranking for all players"""
    # Sort players by evaluation (highest first)
    sorted_players = sorted(players, key=lambda p: p.evaluation, reverse=True)
    
    # Get min and max evaluations for standardization
    max_eval = max(p.evaluation for p in players)
    min_eval = min(p.evaluation for p in players)
    eval_range = max_eval - min_eval
    
    # Assign rankings and standardized evaluations
    for i, player in enumerate(sorted_players):
        player.ranking = i + 1  # Ranking starts from 1
        # Standardize: (value - min) / (max - min)
        if eval_range > 0:
            player.standardized_evaluation = (player.evaluation - min_eval) / eval_range
        else:
            player.standardized_evaluation = 0.5  # If all players have same evaluation
    
    return players
