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

def update_plot(ax, agent_scores_history, agent_bestxi_history, auction_numbers, agent_colors):
    try:
        # Only update plot every 5 auctions to reduce overhead
        if len(auction_numbers) % 5 != 0:
            return
            
        # Check if we have any data to plot
        if not agent_scores_history or not auction_numbers:
            return
            
        ax.clear()
        ax.set_xlabel('Auction Number')
        ax.set_ylabel('Score') 
        ax.set_title(f'Agent Scores (Auction {auction_numbers[-1]})')
        ax.grid(True, alpha=0.3)
        
        # Plot only scores for all 6 fixed agents (simplified)
        fixed_agent_ids = ["rldeep_to_train", "cap_bestx1_aggressive", "cap_tier", 
                          "dynamic_cap_bestx1_balanced", "dynamic_cap_bestx1_aggressive", "random_1"]
        
        for agent_id in fixed_agent_ids:
            if agent_id in agent_scores_history and len(agent_scores_history[agent_id]) > 0:
                # Get data
                data_length = len(agent_scores_history[agent_id])
                x_data = auction_numbers[:data_length]
                y_data = agent_scores_history[agent_id][:data_length]
                
                # Get color for this agent
                color = agent_colors.get(agent_id, "#000000")
                
                # Simple plot - just scores, no bestxi
                ax.plot(x_data, y_data, color=color, linewidth=2, 
                       label=agent_id, marker='o', markersize=3)
        
        ax.legend(fontsize=8)
        
        # Force immediate update without blocking
        plt.draw()
        plt.show(block=False)
        
    except Exception as e:
        print(f"Plot error: {e}")
        pass

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
        ax.set_ylabel('Score / BestXI Objective')
        ax.set_title(f'Training Progress - All Agents Performance (Episodes: {n_episodes})')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error creating final plot: {e}")
        return None

def train():
    # Training 
    episode_auctions = 5
    n_episodes = 0

    # Tracking for plotting and comparing
    agent_scores_history = {}  # {agent_id: [scores]}
    agent_bestxi_history = {}  # {agent_id: [bestxi_objectives]}
    auction_numbers = []
    
    # Track episode vs spinoff performance for RL agent
    episode_rl_scores = []
    episode_rl_bestxi = []
    spinoff_rl_scores = []
    spinoff_rl_bestxi = []

    # Tracking for plotting and visualization
    auction_counter = 0
    agent_colors = {
        "rldeep_to_train": "#FF0000",           # Red
        "cap_bestx1_aggressive": "#00FF00",     # Green  
        "cap_tier": "#0000FF",                  # Blue
        "dynamic_cap_bestx1_balanced": "#FF8C00", # Dark Orange
        "dynamic_cap_bestx1_aggressive": "#8A2BE2", # Blue Violet
        "random_1": "#FFD700"                   # Gold
    }
    
    # Configure matplotlib for interactive plotting
    # plt.ion()  # Turn on interactive mode
    # plt.show(block=False)  # Ensure non-blocking show
    # fig, ax = plt.subplots(figsize=(10, 6))
    # fig.show()  # Explicitly show the figure window
    # ax.set_xlabel('Auction Number')
    # ax.set_ylabel('Score / BestXI Objective')
    # ax.set_title('Agent Performance: Scores and BestXI Objectives Over Time')
    # ax.grid(True, alpha=0.3)

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
            
            # Store auction configurations for fair comparison
            episode_configs = []

            for _ in range(episode_auctions):

                # Randomly select auction configuration
                slots = Slots() #TODO: randomize if sure the thing is slots proof
                agents = refresh_agents()  
                listone = load_players_from_excel()
                initial_credits = random.choice([500, 1000, 10000])
                auction_type = random.choice(["chiamata", "listone", "random"])
                per_ruolo = random.choice([True, False])
                
                # Store configuration for spinoff reuse
                config = {
                    'slots': slots,
                    'agents': agents,
                    'listone': listone,
                    'initial_credits': initial_credits,
                    'auction_type': auction_type,
                    'per_ruolo': per_ruolo
                }
                episode_configs.append(config)

                # Create and run auction with stored configuration
                auction = Auction(config['slots'], config['agents'], config['listone'], config['initial_credits'])
                auction.run_all(
                    auction_type=config['auction_type'],
                    per_ruolo=config['per_ruolo'],
                    verbose=False
                )

                # Extract the agent to train 
                rldeep_agent = next(agent for agent in config['agents'] if isinstance(agent, RLDeepAgent) and agent.agent_id == "rldeep_to_train")

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
                
                # Define the 6 fixed agent IDs
                fixed_agent_ids = ["rldeep_to_train", "cap_bestx1_aggressive", "cap_tier", 
                                  "dynamic_cap_bestx1_balanced", "dynamic_cap_bestx1_aggressive", "random_1"]
                
                for agent in config['agents']:
                    # Only process the 6 fixed agents
                    if agent.agent_id in fixed_agent_ids:
                        # Get score using RL agent's method
                        agent_score = rldeep_agent.get_score(agent, rldeep_agent.auction_progress)
                        scores[agent.agent_id] = agent_score
                        
                        # Get bestxi objective (assuming agents have a bestxi_objective attribute)
                        if hasattr(agent, 'bestxi_objective'):
                            bestxi_objectives[agent.agent_id] = agent.bestxi_objective
                        else:
                            # Fallback - use a default or calculate it
                            bestxi_objectives[agent.agent_id] = agent_score * 1.2  # Example fallback
                        
                        # Initialize history if first time seeing this agent
                        if agent.agent_id not in agent_scores_history:
                            agent_scores_history[agent.agent_id] = []
                            agent_bestxi_history[agent.agent_id] = []
                        
                        # Store the values
                        agent_scores_history[agent.agent_id].append(agent_score)
                        agent_bestxi_history[agent.agent_id].append(bestxi_objectives[agent.agent_id])
                        
                        # Track RL agent performance for episode comparison
                        if agent.agent_id == "rldeep_to_train":
                            episode_rl_scores.append(agent_score)
                            episode_rl_bestxi.append(bestxi_objectives[agent.agent_id])
                
                # Print scores after each auction - RL first, then others
                print(f"Auction {auction_counter}: RL={agent_scores_history['rldeep_to_train'][-1]:.3f} | ", end="")
                other_scores = []
                for agent_id in ["cap_bestx1_aggressive", "cap_tier", "dynamic_cap_bestx1_balanced", "dynamic_cap_bestx1_aggressive", "random_1"]:
                    if agent_id in agent_scores_history and len(agent_scores_history[agent_id]) > 0:
                        other_scores.append(f"{agent_id.split('_')[0]}={agent_scores_history[agent_id][-1]:.3f}")
                print(" | ".join(other_scores))
                
                # Update plot every auction
                # update_plot(ax, agent_scores_history, agent_bestxi_history, auction_numbers, agent_colors)

            # Train the RL agent after the episode finishes and save it 
            n_episodes += 1
            rldeep_agent.trainer.train_step(all_features0_store, output_probability_store, reward_store)
            current_time = datetime.now()
            timestamp = current_time.strftime("%Y%m%d_%H%M%S")
            episode_filename = f"ep{n_episodes}_{timestamp}_weights.pth"
            rldeep_agent.model.save(episode_filename)

            # SPINOFF EPISODE - use same configurations for fair comparison
            
            for i in range(episode_auctions):
                config = episode_configs[i]  # Use the same configuration as episode
                
                # Load the updated weights into the RL agent for this spinoff auction
                if i == 0:  # Load weights only once at the start of spinoff
                    rldeep_agent_in_config = next(agent for agent in config['agents'] if isinstance(agent, RLDeepAgent) and agent.agent_id == "rldeep_to_train")
                    rldeep_agent_in_config.model.load_state_dict(torch.load(f"./weights/{episode_filename}"))

                # Create and run auction with identical configuration
                auction = Auction(config['slots'], config['agents'], config['listone'], config['initial_credits'])

                auction.run_all(
                    auction_type=config['auction_type'],
                    per_ruolo=config['per_ruolo'],
                    verbose=False
                )

                # Extract the agent to train 
                rldeep_agent = next(agent for agent in config['agents'] if isinstance(agent, RLDeepAgent) and agent.agent_id == "rldeep_to_train")

                # Track auction number
                auction_counter += 1
                auction_numbers.append(auction_counter)

                # Collect scores and bestxi objectives for the 6 fixed agents only
                scores = {}
                bestxi_objectives = {}
                
                # Define the 6 fixed agent IDs
                fixed_agent_ids = ["rldeep_to_train", "cap_bestx1_aggressive", "cap_tier", 
                                  "dynamic_cap_bestx1_balanced", "dynamic_cap_bestx1_aggressive", "random_1"]
                
                for agent in config['agents']:
                    # Only process the 6 fixed agents
                    if agent.agent_id in fixed_agent_ids:
                        # Get score using RL agent's method
                        agent_score = rldeep_agent.get_score(agent, rldeep_agent.auction_progress)
                        scores[agent.agent_id] = agent_score
                        
                        # Get bestxi objective (assuming agents have a bestxi_objective attribute)
                        if hasattr(agent, 'bestxi_objective'):
                            bestxi_objectives[agent.agent_id] = agent.bestxi_objective
                        else:
                            # Fallback - use a default or calculate it
                            bestxi_objectives[agent.agent_id] = agent_score * 1.2  # Example fallback
                        
                        # Initialize history if first time seeing this agent
                        if agent.agent_id not in agent_scores_history:
                            agent_scores_history[agent.agent_id] = []
                            agent_bestxi_history[agent.agent_id] = []
                        
                        # Store the values
                        agent_scores_history[agent.agent_id].append(agent_score)
                        agent_bestxi_history[agent.agent_id].append(bestxi_objectives[agent.agent_id])
                        
                        # Track RL agent performance for spinoff comparison
                        if agent.agent_id == "rldeep_to_train":
                            spinoff_rl_scores.append(agent_score)
                            spinoff_rl_bestxi.append(bestxi_objectives[agent.agent_id])

                # Print scores after each auction - RL first, then others
                print(f"Auction {auction_counter}: RL={agent_scores_history['rldeep_to_train'][-1]:.3f} | ", end="")
                other_scores = []
                for agent_id in ["cap_bestx1_aggressive", "cap_tier", "dynamic_cap_bestx1_balanced", "dynamic_cap_bestx1_aggressive", "random_1"]:
                    if agent_id in agent_scores_history and len(agent_scores_history[agent_id]) > 0:
                        other_scores.append(f"{agent_id.split('_')[0]}={agent_scores_history[agent_id][-1]:.3f}")
                print(" | ".join(other_scores))

                # Update plot every auction
                # update_plot(ax, agent_scores_history, agent_bestxi_history, auction_numbers, agent_colors)

            # Compare episode vs spinoff performance for RL agent
            if episode_rl_scores and spinoff_rl_scores and episode_rl_bestxi and spinoff_rl_bestxi:
                # Calculate means for episode
                episode_score_mean = np.mean(episode_rl_scores)
                episode_bestxi_mean = np.mean(episode_rl_bestxi)
                
                # Calculate means for spinoff
                spinoff_score_mean = np.mean(spinoff_rl_scores)
                spinoff_bestxi_mean = np.mean(spinoff_rl_bestxi)
                
                # If both spinoff score and bestxi means are better than episode, save as best weights
                if spinoff_score_mean > episode_score_mean and spinoff_bestxi_mean > episode_bestxi_mean:
                    current_time = datetime.now()
                    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                    best_weights_filename = f"best_weights_ep{n_episodes}_{timestamp}.pth"
                    rldeep_agent.model.save(best_weights_filename)
                    rldeep_agent.model.save("best_weights.pth")  # Also save as standard best weights
                    
            # Reset tracking for next episode
            episode_rl_scores = []
            episode_rl_bestxi = []
            spinoff_rl_scores = []
            spinoff_rl_bestxi = []
            
            # Save plot periodically
            # if n_episodes % 5 == 0:
            #     plt.savefig(f'training_progress_episode_{n_episodes}.png', dpi=300, bbox_inches='tight')
    
    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted by user after {n_episodes} episodes.")
        
    # Create and save final comprehensive plot
    print(f"📊 Creating final training plot...")
    final_fig = create_final_plot(agent_scores_history, agent_bestxi_history, auction_numbers, agent_colors, n_episodes)
    
    if final_fig:
        final_plot_filename = f'training_progress_final_ep{n_episodes}.png'
        final_fig.savefig(final_plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Final plot saved as: {final_plot_filename}")
        plt.close(final_fig)  # Close the figure to free memory
    
    print("✅ Training completed gracefully.")




if __name__ == '__main__':
    train()