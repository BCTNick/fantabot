import numpy as np
from models import Player
from typing import List

def compute_state(agent, remaining_players: List[Player]):
    """
    Compute the state of an agent given their status and the pool of remaining players.
    """
    credits = agent.current_credits
    
    # Calculate remaining slots needed
    current_players = len(agent.squad.gk) + len(agent.squad.def_) + len(agent.squad.mid) + len(agent.squad.att)
    total_slots = agent.slots.gk + agent.slots.def_ + agent.slots.mid + agent.slots.att if hasattr(agent, 'slots') else 25
    remaining_slots = max(1, total_slots - current_players)  # Avoid division by zero
    
    # Get team evaluations (current team strength, 0-100 scale)
    best_xi = agent.squad.objective(bestxi=True, standardized=True)
    total_eval = agent.squad.objective(standardized=True)
    team_value = best_xi * 0.9 + total_eval * 0.1
    
    # Calculate remaining potential (normalized)
    credit_per_slot = credits / remaining_slots
    
    # Estimate value of top remaining players we could realistically get
    if remaining_players:
        remaining_players_sorted = sorted(remaining_players, key=lambda p: p.evaluation, reverse=True)
        top_players = remaining_players_sorted[:min(remaining_slots, len(remaining_players_sorted))]
        avg_remaining_value = np.mean([p.evaluation for p in top_players]) if top_players else 0
    else:
        avg_remaining_value = 0
    
    # Potential strength: how much of the remaining value can we afford?
    if avg_remaining_value > 0:
        potential_strength = min(credit_per_slot / avg_remaining_value, 1.0)
    else:
        potential_strength = 0
    
    # Combine: current strength + potential to improve (weighted by remaining slots)
    potential_component = potential_strength * remaining_slots * 10  # Scale to be comparable with team_value
    
    return team_value + potential_component

def compute_z_relative_state(agent, agents, remaining_players):
    """
    Computes the z-score of the agent's state relative to all others.
    """
    my_state = compute_state(agent, remaining_players)
    others_states = [compute_state(a, remaining_players) for a in agents if a != agent]

    if not others_states:
        return 0  # No other agents to compare with
    
    mean_others = np.mean(others_states)
    std_others = np.std(others_states)
    
    if std_others == 0:
        return 0  # no variability among others
    return (my_state - mean_others) / std_others

def compute_reward_delta(agent, agents, remaining_players, t0_state, t1_state):
    """
    Computes the delta in the agent's relative position (reward function)
    between two time steps (t1 - t0).
    """
    return t1_state - t0_state

def compute_bidding_reward(agent, agents, remaining_players, decision: str, player: Player, current_price: int, highest_bidder: str) -> float:
    """
    Compute reward for a bidding decision by simulating the state changes.
    
    Args:
        agent: The DeepAgent making the decision
        agents: List of all agents in the auction
        remaining_players: List of players still available
        decision: "offer_+1" or "no_offer"
        player: The player that was being bid on
        current_price: The current price when decision was made
        highest_bidder: Who is currently the highest bidder
        
    Returns:
        reward: Float representing the reward for this decision
    """
    
    # Compute current state before any simulation
    t0_state = compute_z_relative_state(agent, agents, remaining_players)
    
    if decision == "offer_+1":
        # Simulate: Agent wins the player at current_price + 1
        simulated_price = current_price + 1
        
        # Temporarily modify agent state
        original_credits = agent.current_credits
        agent.current_credits -= simulated_price
        
        # Add player to agent's squad temporarily
        added_successfully = False
        try:
            if player.role == "GK":
                agent.squad.gk.append(player)
                added_successfully = True
            elif player.role == "DEF":
                agent.squad.def_.append(player)
                added_successfully = True
            elif player.role == "MID":
                agent.squad.mid.append(player)
                added_successfully = True
            elif player.role == "ATT":
                agent.squad.att.append(player)
                added_successfully = True
                
            # Remove player from remaining players for state calculation
            remaining_after_win = [p for p in remaining_players if p != player]
            
            # Compute new state after winning
            t1_state = compute_z_relative_state(agent, agents, remaining_after_win)
            
        finally:
            # Always restore agent state, even if there was an error
            agent.current_credits = original_credits
            if added_successfully:
                try:
                    if player.role == "GK":
                        agent.squad.gk.remove(player)
                    elif player.role == "DEF":
                        agent.squad.def_.remove(player)
                    elif player.role == "MID":
                        agent.squad.mid.remove(player)
                    elif player.role == "ATT":
                        agent.squad.att.remove(player)
                except ValueError:
                    # Player wasn't in the list, ignore the error
                    pass
            
    else:  # decision == "no_offer"
        # Simulate: Highest bidder wins the player at current_price
        
        # Find the highest bidder agent
        highest_bidder_agent = None
        for a in agents:
            if a.agent_id == highest_bidder:
                highest_bidder_agent = a
                break
                
        if highest_bidder_agent and highest_bidder_agent != agent:
            # Temporarily modify highest bidder's state
            original_credits = highest_bidder_agent.current_credits
            highest_bidder_agent.current_credits -= current_price
            
            # Add player to highest bidder's squad temporarily
            added_successfully = False
            try:
                if player.role == "GK":
                    highest_bidder_agent.squad.gk.append(player)
                    added_successfully = True
                elif player.role == "DEF":
                    highest_bidder_agent.squad.def_.append(player)
                    added_successfully = True
                elif player.role == "MID":
                    highest_bidder_agent.squad.mid.append(player)
                    added_successfully = True
                elif player.role == "ATT":
                    highest_bidder_agent.squad.att.append(player)
                    added_successfully = True
                    
                # Remove player from remaining players for state calculation
                remaining_after_sale = [p for p in remaining_players if p != player]
                
                # Compute new state after highest bidder wins
                t1_state = compute_z_relative_state(agent, agents, remaining_after_sale)
                
            finally:
                # Always restore highest bidder's state
                highest_bidder_agent.current_credits = original_credits
                if added_successfully:
                    try:
                        if player.role == "GK":
                            highest_bidder_agent.squad.gk.remove(player)
                        elif player.role == "DEF":
                            highest_bidder_agent.squad.def_.remove(player)
                        elif player.role == "MID":
                            highest_bidder_agent.squad.mid.remove(player)
                        elif player.role == "ATT":
                            highest_bidder_agent.squad.att.remove(player)
                    except ValueError:
                        # Player wasn't in the list, ignore the error
                        pass
        else:
            # No highest bidder or highest bidder is the agent itself
            # Player goes unsold, no state change
            t1_state = t0_state
    
    # Return the delta in relative state
    return compute_reward_delta(agent, agents, remaining_players, t0_state, t1_state)

def compute_final_auction_reward(agent, agents) -> float:
    """
    Compute final reward at the end of an auction based on relative performance.
    
    Args:
        agent: The DeepAgent 
        agents: List of all agents in the auction
        
    Returns:
        reward: Float representing final performance reward
    """
    
    # Get final evaluations for all agents
    agent_scores = []
    for a in agents:
        best_xi = a.squad.objective(bestxi=True, standardized=True)
        total_eval = a.squad.objective(standardized=True)
        credits_remaining = a.current_credits
        
        # Combined score: team quality + remaining credits efficiency
        score = best_xi * 0.8 + total_eval * 0.1 + (credits_remaining / 1000.0) * 0.1
        agent_scores.append(score)
        
        if a == agent:
            my_score = score
    
    # Compute relative performance
    agent_scores = np.array(agent_scores)
    my_rank = np.sum(agent_scores <= my_score) - 1  # -1 because we include ourselves
    total_agents = len(agents)
    
    # Normalize rank to [-1, 1] where 1 is best, -1 is worst
    if total_agents > 1:
        relative_performance = (my_rank / (total_agents - 1)) * 2 - 1
    else:
        relative_performance = 0
    
    return relative_performance
