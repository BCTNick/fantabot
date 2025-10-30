import sys
import os

from prova_policy_value_network import PolicyValueNet
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from agents.policy_value_agent import PolicyValueAgent
from src.data_loader import load_players_from_excel
from src.models import Slots
from src.agents.agent_class import RandomAgent
from src.agents.cap_based_agent import CapAgent
from src.agents.dynamic_cap_based_agent import DynamicCapAgent
from src.auction import Auction
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import logging
from datetime import datetime

def train_episode(self, episode):
    """
    episode: list of (state, action), and final reward separately.
    """
    states = torch.stack([torch.tensor(s, dtype=torch.float32) for s, _ in episode])
    actions = torch.tensor([a for _, a in episode], dtype=torch.long)
    final_reward = episode[-1][2] if len(episode[0]) == 3 else None  # optional

    # Assume final reward provided separately
    total_reward = final_reward if final_reward is not None else self.last_reward

    self.optimizer.zero_grad()
    total_loss, policy_loss, value_loss, entropy = self.compute_loss(states, actions, total_reward)
    total_loss.backward()
    self.optimizer.step()

    return {
        "total_loss": total_loss.item(),
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "reward": total_reward
    }
    

def refresh_agents():
    num_partecipants = random.choice([6, 8, 10, 12])
    agents = [PolicyValueAgent("agent_to_train", "training"),
            CapAgent(agent_id="cap_bestx1_aggressive", cap_strategy="bestxi_based", bestxi_budget=0.99),
            CapAgent(agent_id="cap_tier", cap_strategy="tier_based"),
            DynamicCapAgent(agent_id="dynamic_cap_bestx1_balanced", cap_strategy="bestxi_based", bestxi_budget=0.95),
            DynamicCapAgent(agent_id="dynamic_cap_bestx1_aggressive", cap_strategy="bestxi_based", bestxi_budget=0.99),
            RandomAgent(agent_id="random_1")]

    for _ in range(num_partecipants - 6): 
        agent_id = f"_{_}"
        agent = random.choice([CapAgent(agent_id=agent_id, cap_strategy="bestxi_based"),
                               CapAgent(agent_id=agent_id, cap_strategy="bestxi_based", bestxi_budget=0.99),
                               CapAgent(agent_id=agent_id, cap_strategy="tier_based"),
                               DynamicCapAgent(agent_id=agent_id, cap_strategy="bestxi_based", bestxi_budget=0.95),
                               DynamicCapAgent(agent_id=agent_id, cap_strategy="bestxi_based", bestxi_budget=0.99),
                               DynamicCapAgent(agent_id=agent_id, cap_strategy="tier_based"),
                               RandomAgent(agent_id=agent_id)])
        agents.append(agent)

        random.shuffle(agents)
    return agents

#TODO: adapt the new final plot to the new model
def create_final_plot(agent_scores_history, agent_bestxi_history, auction_numbers, agent_colors, n_episodes):
    """Create comprehensive final plot with all agents' scores and bestxi"""
    try:
        # Create figure with larger size for better readability
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot all 6 fixed agents
        fixed_agent_ids = ["rldeep_to_train", "cap_bestx1_aggressive", "cap_tier", 
                          "dynamic_cap_bestx1_balanced", "dynamic_cap_bestx1_aggressive", "random_1"]
        
        for agent_id in fixed_agent_ids:
            if agent_id in agent_scores_history and len(agent_scores_history[agent_id]) > 0:
                # Get data length and ensure it matches
                data_length = len(agent_scores_history[agent_id])
                x_data = auction_numbers[:data_length]
                
                if agent_id in agent_bestxi_history and len(agent_bestxi_history[agent_id]) >= data_length:
                    # Get color for this agent
                    base_color = agent_colors.get(agent_id, "#000000")
                    
                    # Convert hex to RGB for lighter shade
                    rgb = mcolors.hex2color(base_color)
                    light_rgb = tuple(min(1.0, c + 0.3) for c in rgb)
                    
                    # Plot scores (solid line)
                    ax.plot(x_data, agent_scores_history[agent_id][:data_length], 
                           color=base_color, linewidth=2, label=f'{agent_id}_score', 
                           marker='o', markersize=2)
                    
                    # Plot bestxi (dashed line, lighter color)
                    ax.plot(x_data, agent_bestxi_history[agent_id][:data_length], 
                           color=light_rgb, linewidth=1, linestyle='--', 
                           label=f'{agent_id}_bestxi', marker='s', markersize=1)
        
        ax.set_xlabel('Auction Number')
        ax.set_ylabel('Weighted Score / BestXI Objective')
        ax.set_title(f'Training Progress - Participant-Weighted Performance (Episodes: {n_episodes})')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error creating final plot: {e}")
        return None

# TODO: redesign the saving of weights and their loading process
def find_most_recent_weights(no_spin_dir):
    """Find the most recent .pth weights file in the no_spin directory"""
    try:
        if not os.path.exists(no_spin_dir):
            return None
            
        # Get all .pth files in the directory
        weight_files = [f for f in os.listdir(no_spin_dir) if f.endswith('.pth')]
        
        if not weight_files:
            return None
            
        # Sort by modification time (most recent first)
        weight_files.sort(key=lambda x: os.path.getmtime(os.path.join(no_spin_dir, x)), reverse=True)
        
        most_recent = os.path.join(no_spin_dir, weight_files[0])
        return most_recent
        
    except Exception as e:
        print(f"Error finding most recent weights: {e}")
        return None

# TODO: redisgn this too, it should be new every time you redeseign the NN architecture
def find_last_episode_number(no_spin_dir):
    """Find the last completed episode number from weight files in no_spin directory"""
    try:
        if not os.path.exists(no_spin_dir):
            return 0
            
        # Get all weight files that match the pattern ep{N}_*.pth
        weight_files = [f for f in os.listdir(no_spin_dir) if f.startswith('ep') and f.endswith('_weights.pth')]
        
        if not weight_files:
            return 0
            
        # Extract episode numbers from filenames
        episode_numbers = []
        for filename in weight_files:
            try:
                # Extract number from pattern ep{N}_timestamp_weights.pth
                ep_part = filename.split('_')[0]  # Gets 'ep{N}'
                if ep_part.startswith('ep'):
                    episode_num = int(ep_part[2:])  # Remove 'ep' prefix and convert to int
                    episode_numbers.append(episode_num)
            except (ValueError, IndexError):
                continue
                
        if episode_numbers:
            return max(episode_numbers)
        else:
            return 0
            
    except Exception as e:
        print(f"Error finding last episode number: {e}")
        return 0
    
def get_reward(agent_to_train, auction):
    #TODO: get the reward based on the list of players: [-1,1] with -1 being no player bought and 1 being all the best players bought (weight by bestxi and not)
    
    # # Get all players by role
    # gks = sorted([p for p in auction.players if p.role == "GK"], 
    #              key=lambda x: x.standardized_evaluation, reverse=True)
    # defs = sorted([p for p in auction.players if p.role == "DEF"], 
    #               key=lambda x: x.standardized_evaluation, reverse=True)
    # mids = sorted([p for p in auction.players if p.role == "MID"], 
    #               key=lambda x: x.standardized_evaluation, reverse=True)
    # atts = sorted([p for p in auction.players if p.role == "ATT"], 
    #               key=lambda x: x.standardized_evaluation, reverse=True)

    # # Calculate best possible XI (1 GK + 3 DEF + 3 MID + 3 ATT)
    # best_possible_xi = sum([
    #     gks[0].standardized_evaluation if gks else 0,  # Best GK
    #     *[p.standardized_evaluation for p in defs[:3]],  # Best 3 DEF
    #     *[p.standardized_evaluation for p in mids[:3]],  # Best 3 MID
    #     *[p.standardized_evaluation for p in atts[:3]]   # Best 3 ATT
    # ])

    # # Calculate second best possible squad based on agent slots
    # best_possible_reserves = sum([
    #     *[p.standardized_evaluation for p in gks[1:agent.slots.gk]],     # From 2nd GK to max slots
    #     *[p.standardized_evaluation for p in defs[3:agent.slots.def_]],  # From 4th DEF to max slots
    #     *[p.standardized_evaluation for p in mids[3:agent.slots.mid]],   # From 4th MID to max slots
    #     *[p.standardized_evaluation for p in atts[3:agent.slots.att]]    # From 4th ATT to max slots
    # ])

    # bestxi = agent.squad.objective(bestxi=True, standardized=True)/ best_possible_xi 
    # best_reserves = (agent.squad.objective(bestxi=False, standardized=True)-agent.squad.objective(bestxi=True, standardized=True))/ best_possible_reserves

    # TODO: Understand if you want to calculate participant weight: if yes, normalize to 12 participants as baseline
    scores = []
    for agent in auction.agents:
        score = agent.squad.objective(bestxi=True, standardized=True)*0.9+agent.squad.objective(bestxi=False, standardized=True)*0.1
        scores.append(score)

    reward = 2 * (agent_to_train.squad.objective(bestxi=True, standardized=True)*0.9+agent_to_train.squad.objective(bestxi=False, standardized=True)*0.1-min(scores)) / (max(scores)-min(scores)+1e-8) - 1  # Normalize to [-1, 1]
    return reward

def setup_logging(log_dir):
    """Setup logging for the training session"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_no_spin_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Create logger
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger, log_filepath

def train():
    # Create round folder for this training session
    base_weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
    current_round_dir = os.path.join(base_weights_dir, f'no_spin')
    os.makedirs(current_round_dir, exist_ok=True)
    
    # Find the last completed episode and continue from there
    last_episode = find_last_episode_number(current_round_dir)
    n_episodes = last_episode
    
    if last_episode > 0:
        print(f"🔄 Resuming training from episode {last_episode + 1}")
        print(f"📁 Found {last_episode} completed episodes in: {current_round_dir}")
    else:
        print(f"🔄 Starting new training session")
        print(f"📁 No previous episodes found in: {current_round_dir}")
    
    # Setup logging
    logger, log_filepath = setup_logging(current_round_dir)
    
    print(f"� Logging to: {log_filepath}")
    
    if last_episode > 0:
        logger.info("=== TRAINING SESSION RESUMED ===")
        logger.info(f"Resuming from episode {last_episode + 1} (found {last_episode} completed episodes)")
    else:
        logger.info("=== TRAINING SESSION STARTED ===")
        logger.info("Starting new training session")
        
    logger.info(f"Training directory: {current_round_dir}")
    logger.info(f"Starting from episode {n_episodes + 1}")

    # Tracking for plotting and comparing
    agent_score_history = {}  # {agent_id: [scores]}

    # Tracking for plotting and visualization
    agent_colors = {
        "rldeep_to_train": "#FF0000",           # Red
        "cap_bestx1_aggressive": "#00FF00",     # Green  
        "cap_tier": "#0000FF",                  # Blue
        "dynamic_cap_bestx1_balanced": "#FF8C00", # Dark Orange
        "dynamic_cap_bestx1_aggressive": "#8A2BE2", # Blue Violet
        "random_1": "#FFD700"                   # Gold
    }

    print("🚀 Training started! Press Ctrl+C to stop gracefully.")
    print("📁 Create 'STOP' or 'STOP.txt' file in current directory to stop after current episode.")
    
    while True:
        # Check for stop file (both STOP and STOP.txt)
        if os.path.exists("STOP") or os.path.exists("STOP.txt"):
            print("🛑 STOP file detected. Finishing current episode and exiting...")
            # Clean up both possible stop files
            if os.path.exists("STOP"):
                os.remove("STOP")
            if os.path.exists("STOP.txt"):
                os.remove("STOP.txt")
            break

            # Log episode start
            n_episodes += 1
            logger.info(f"=== EPISODE {n_episodes} STARTED ===")
            
            # Load most recent weights at the start of each episode
            most_recent_weights = find_most_recent_weights(current_round_dir)
            if most_recent_weights and n_episodes > 1:  # Load weights if we have previous episodes
                logger.info(f"Loading weights: {os.path.basename(most_recent_weights)}")
            else:
                logger.info("Using random weights (first episode)")


            # Randomly select auction configuration
            slots = Slots() #TODO: randomize if sure the thing is slots proof
            agents = refresh_agents()  
            listone = load_players_from_excel() #TODO: randomize listone if sure the thing is players proof
            initial_credits = random.choice([500, 1000, 10000])
            auction_type = random.choice(["chiamata", "listone", "random"]) 
            per_ruolo = random.choice([True, False])
            
            # Log auction configuration
            logger.info(f"Auction {n_episodes}: participants={len(agents)}, credits={initial_credits}, type={auction_type}, per_ruolo={per_ruolo}")

            # Create and run auction with stored configuration
            auction = Auction(slots, agents, listone, initial_credits)

            # Extract the agent to train 
            agent_to_train = next(agent for agent in agents if isinstance(agent, PolicyValueAgent) and agent.agent_id == "agent_to_train")

            # TODO it should get the weights automatically when it is initialized
            if most_recent_weights and (n_episodes > 1):
                try:
                    agent_to_train.model.load_state_dict(torch.load(most_recent_weights))
                    logger.info(f"Weights loaded successfully into Agent")
                except Exception as e:
                    logger.error(f"Failed to load weights: {e}")

            auction.run_all(
                auction_type=auction_type,
                per_ruolo=per_ruolo,
                verbose=False
            )
        
            
            # Define the 6 fixed agent IDs
            fixed_agent_ids = ["agent_to_train", "cap_bestx1_aggressive", "cap_tier", 
                                "dynamic_cap_bestx1_balanced", "dynamic_cap_bestx1_aggressive", "random_1"]
            
            for agent in agents:
                # Only process the 6 fixed agents
                if agent.agent_id in fixed_agent_ids:
                    weighted_score = (agent.squad.objective(bestxi=True, standardized=True)*0.9+agent.squad.objective(bestxi=False, standardized=True)*0.1) * (len(auction.agents) / 12)  
                    agent_score_history[agent.agent_id].append(weighted_score)


            # Print scores after each auction - RL first, then others (weighted scores)
            logger.info(f"Auction {n_episodes} Results:")
            other_scores = []
            log_scores = [f"RL={agent_score_history['agent_to_train'][-1]:.3f}"]
            for agent_id in ["cap_bestx1_aggressive", "cap_tier", "dynamic_cap_bestx1_balanced", "dynamic_cap_bestx1_aggressive", "random_1"]:
                if agent_id in agent_score_history and len(agent_score_history[agent_id]) > 0:
                    score_str = f"{agent_id.split('_')[0]}={agent_score_history[agent_id][-1]:.3f}"
                    other_scores.append(score_str)
                    log_scores.append(score_str)
            logger.info(" | ".join(other_scores))

        # Train the RL agent after the episode finishes and save it
        logger.info(f"Training RL agent after episode {n_episodes} with {len(all_features0_store)} training samples")

        #TODO: adapt to new train step
        net = PolicyValueNet()
        optimizer = optim.Adam(net.parameters(), lr=0.01)
        reward = get_reward(agent_to_train, auction)
        R = torch.tensor([reward], dtype=torch.float32)

        # --- Compute losses ---
        log_probs = agent_to_train.policies_store
        values = agent_to_train.values_store
        policy_loss = 0
        value_loss = 0

        for log_prob, value in zip(log_probs, values):
            advantage = R - value.detach()
            policy_loss += -log_prob * advantage
            value_loss += F.mse_loss(value, R)

        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_time = datetime.now()
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        episode_filename = f"ep{n_episodes}_{timestamp}_weights.pth"
        episode_filepath = os.path.join(current_round_dir, episode_filename)
        agent_to_train.model.save(episode_filepath)
        logger.info(f"Episode {n_episodes} completed. Weights saved: {episode_filename}")
        
        
except KeyboardInterrupt:
    logger.info(f"Training interrupted by user after {n_episodes} episodes")
    
# Create and save final comprehensive plot
logger.info("Creating final training plot...")
final_fig = create_final_plot(agent_scores_history, agent_bestxi_history, auction_numbers, agent_colors, n_episodes)

if final_fig:
    final_plot_filename = f'training_progress_no_spin_ep{n_episodes}.png'
    final_plot_filepath = os.path.join(current_round_dir, final_plot_filename)
    final_fig.savefig(final_plot_filepath, dpi=300, bbox_inches='tight')
    print(f"📊 Final plot saved as: {final_plot_filepath}")
    logger.info(f"Final plot saved: {final_plot_filename}")
    plt.close(final_fig)  # Close the figure to free memory

print("✅ Training completed gracefully.")
logger.info(f"=== TRAINING SESSION COMPLETED === Total episodes: {n_episodes}, Total auctions: {auction_counter}")

# Close logging handlers
for handler in logger.handlers:
    handler.close()
    logger.removeHandler(handler)




if __name__ == '__main__':
    train()