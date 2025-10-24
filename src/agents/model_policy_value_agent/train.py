import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.data_loader import load_players_from_excel
from src.models import Slots
from src.agents.agent_class import RandomAgent
from src.agents.cap_based_agent import CapAgent
from src.agents.dynamic_cap_based_agent import DynamicCapAgent
from src.agents.rl_deep_agent import RLDeepAgent
from src.auction import Auction
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import os
import logging
from datetime import datetime

def refresh_agents():
    num_partecipants = random.choice([6, 8, 10, 12])
    agents = [RLDeepAgent("rldeep_to_train", "training"),
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
    # Training 
    episode_auctions = 1

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
    logger.info(f"Episode auctions per episode: {episode_auctions}")
    logger.info(f"Starting from episode {n_episodes + 1}, auction counter will start from {last_episode * episode_auctions}")

    # Tracking for plotting and comparing
    agent_scores_history = {}  # {agent_id: [scores]}
    agent_bestxi_history = {}  # {agent_id: [bestxi_objectives]}
    auction_numbers = []
    episode_rl_scores = []
    episode_rl_bestxi = []


    # Tracking for plotting and visualization
    auction_counter = last_episode * episode_auctions  # Continue auction numbering from where we left off
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
    
    try:
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

            # Store stuff for episode
            all_features0_store = []
            output_probability_store = []
            reward_store = []

            # Log episode start
            logger.info(f"=== EPISODE {n_episodes + 1} STARTED ===")
            
            # Load most recent weights at the start of each episode
            most_recent_weights = find_most_recent_weights(current_round_dir)
            if most_recent_weights and (n_episodes > 0 or last_episode > 0):  # Load weights if we have previous episodes
                print(f"🔄 Loading most recent weights: {os.path.basename(most_recent_weights)}")
                logger.info(f"Loading weights: {os.path.basename(most_recent_weights)}")
            else:
                logger.info("Using random weights (first episode)")

            for _ in range(episode_auctions):

                # Randomly select auction configuration
                slots = Slots() #TODO: randomize if sure the thing is slots proof
                agents = refresh_agents()  
                listone = load_players_from_excel()
                initial_credits = random.choice([500, 1000, 10000])
                auction_type = random.choice(["chiamata", "listone", "random"]) 
                per_ruolo = random.choice([True, False])
                
                # Log auction configuration
                num_participants = len(agents)
                logger.info(f"Auction {auction_counter + 1}: participants={num_participants}, credits={initial_credits}, type={auction_type}, per_ruolo={per_ruolo}")

                # Create and run auction with stored configuration
                auction = Auction(slots, agents, listone, initial_credits)

                # Load weights into RL agent if available
                if most_recent_weights and (n_episodes > 0 or last_episode > 0):
                    rl_agent = next(agent for agent in agents if isinstance(agent, RLDeepAgent) and agent.agent_id == "rldeep_to_train")
                    try:
                        rl_agent.model.load_state_dict(torch.load(most_recent_weights))
                        if _ == 0:  # Only print once per episode
                            print(f"✅ Loaded weights into RL agent: {os.path.basename(most_recent_weights)}")
                            logger.info(f"Weights loaded successfully into RL agent")
                    except Exception as e:
                        print(f"❌ Error loading weights: {e}")
                        logger.error(f"Failed to load weights: {e}")

                auction.run_all(
                    auction_type=auction_type,
                    per_ruolo=per_ruolo,
                    verbose=False
                )

                # Extract the agent to train 
                rldeep_agent = next(agent for agent in agents if isinstance(agent, RLDeepAgent) and agent.agent_id == "rldeep_to_train")

                # remember
                all_features0_store.extend(rldeep_agent.all_features0_store)
                output_probability_store.extend(rldeep_agent.output_probability_store)
                reward_store.extend(rldeep_agent.reward_store)

                # Track auction number
                auction_counter += 1
                auction_numbers.append(auction_counter)

                # Collect scores and bestxi objectives for the 6 fixed agents only
                scores = {}
                bestxi_objectives = {}
                
                # Calculate participant weight: normalize to 12 participants as baseline
                participant_weight = num_participants / 12.0
                
                # Define the 6 fixed agent IDs
                fixed_agent_ids = ["rldeep_to_train", "cap_bestx1_aggressive", "cap_tier", 
                                  "dynamic_cap_bestx1_balanced", "dynamic_cap_bestx1_aggressive", "random_1"]
                
                for agent in agents:
                    # Only process the 6 fixed agents
                    if agent.agent_id in fixed_agent_ids:
                        # Get raw score and bestxi objective
                        raw_agent_score = rldeep_agent.get_score(agent, rldeep_agent.auction_progress)
                        raw_bestxi_objective = agent.squad.objective(bestxi=True, standardized=True)
                        
                        # Apply participant weighting
                        weighted_agent_score = raw_agent_score * participant_weight
                        weighted_bestxi_objective = raw_bestxi_objective * participant_weight
                        
                        scores[agent.agent_id] = weighted_agent_score
                        bestxi_objectives[agent.agent_id] = weighted_bestxi_objective

                        
                        # Initialize history if first time seeing this agent
                        if agent.agent_id not in agent_scores_history:
                            agent_scores_history[agent.agent_id] = []
                            agent_bestxi_history[agent.agent_id] = []
                        
                        # Store the weighted values
                        agent_scores_history[agent.agent_id].append(weighted_agent_score)
                        agent_bestxi_history[agent.agent_id].append(weighted_bestxi_objective)
                        
                        # Track RL agent performance for episode comparison
                        if agent.agent_id == "rldeep_to_train":
                            episode_rl_scores.append(weighted_agent_score)
                            episode_rl_bestxi.append(weighted_bestxi_objective)
                
                # Print scores after each auction - RL first, then others (weighted scores)
                print(f"Auction {auction_counter} (w={participant_weight:.2f}): RL={agent_scores_history['rldeep_to_train'][-1]:.3f} | ", end="")
                other_scores = []
                log_scores = [f"RL={agent_scores_history['rldeep_to_train'][-1]:.3f}"]
                for agent_id in ["cap_bestx1_aggressive", "cap_tier", "dynamic_cap_bestx1_balanced", "dynamic_cap_bestx1_aggressive", "random_1"]:
                    if agent_id in agent_scores_history and len(agent_scores_history[agent_id]) > 0:
                        score_str = f"{agent_id.split('_')[0]}={agent_scores_history[agent_id][-1]:.3f}"
                        other_scores.append(score_str)
                        log_scores.append(score_str)
                print(" | ".join(other_scores))
                
                # Log auction results with weight info
                logger.info(f"Auction {auction_counter} weighted results (w={participant_weight:.3f}): {' | '.join(log_scores)}")

            # Train the RL agent after the episode finishes and save it
            n_episodes += 1
            logger.info(f"Training RL agent after episode {n_episodes} with {len(all_features0_store)} training samples")
            rldeep_agent.trainer.train_step(all_features0_store, output_probability_store, reward_store)
            current_time = datetime.now()
            timestamp = current_time.strftime("%Y%m%d_%H%M%S")
            episode_filename = f"ep{n_episodes}_{timestamp}_weights.pth"
            episode_filepath = os.path.join(current_round_dir, episode_filename)
            rldeep_agent.model.save(episode_filepath)
            logger.info(f"Episode {n_episodes} completed. Weights saved: {episode_filename}")
            
            
    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted by user after {n_episodes} episodes.")
        logger.info(f"Training interrupted by user after {n_episodes} episodes")
        
    # Create and save final comprehensive plot
    print(f"📊 Creating final training plot...")
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