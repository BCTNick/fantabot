import random
import logging
import pandas as pd
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.models import Slots
from src.agents.agent_class import RandomAgent
from src.agents.cap_based_agent import CapAgent
from src.agents.dynamic_cap_based_agent import DynamicCapAgent
from src.agents.rl_deep_agent import RLDeepAgent

# Import auction, model and data loader
from src.auction import Auction
from src.data_loader import load_players_from_excel
from src.models import Slots

    
# Set up logging - clear existing handlers first
logger = logging.getLogger()
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

log_filename = f"logs/auction_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()  # This still prints to console too
    ],
    force=True  # Force reconfiguration
)
logger = logging.getLogger()

# Create the characteristics of the auction
agents = [
        CapAgent(agent_id="cap_bestx1_balanced", cap_strategy="bestxi_based"),
        CapAgent(agent_id="cap_bestx1_aggressive", cap_strategy="bestxi_based", bestxi_budget=0.99),
        CapAgent(agent_id="cap_tier", cap_strategy="tier_based"),
        DynamicCapAgent(agent_id="dynamic_cap_bestx1_balanced", cap_strategy="bestxi_based", bestxi_budget=0.95),
        DynamicCapAgent(agent_id="dynamic_cap_bestx1_aggressive", cap_strategy="bestxi_based", bestxi_budget=0.99),
        DynamicCapAgent(agent_id="dynamic_cap_tier", cap_strategy="tier_based"),
        RLDeepAgent(agent_id="RLDEEPAGENT", mode = "training"),
        RandomAgent(agent_id="random_1")
    ]
random.shuffle(agents) 

initial_credits = 1000
slots = Slots()
listone = load_players_from_excel()
auction_type = "chiamata"
per_ruolo = True

# Create auction (but don't run yet)
auction = Auction(slots, agents, listone, initial_credits)

# Log auction and agent configuration
logger.info("\n⚙️ AUCTION CONFIGURATION")
logger.info("-" * 40)
logger.info(f"📊 Slots per team:")
logger.info(f"  GK: {slots.gk}")
logger.info(f"  DEF: {slots.def_}")
logger.info(f"  MID: {slots.mid}")
logger.info(f"  ATT: {slots.att}")
logger.info(f"💰 Initial credits per agent: {initial_credits}")
logger.info(f"👥 Number of agents: {len(agents)}")
logger.info(f"📋 Total players available: {len(listone)}")
logger.info(f"  Auction type: {auction_type}")
logger.info(f"  Per role: {per_ruolo}")    
logger.info("\n🤖 AGENT CONFIGURATIONS")
logger.info("-" * 40)
for agent in agents:
    if isinstance(agent, CapAgent):
        logger.info(f"  {agent.agent_id}: CapAgent (strategy: {agent.cap_strategy})")
    elif isinstance(agent, RandomAgent):
        logger.info(f"  {agent.agent_id}: RandomAgent")
    else:
        logger.info(f"  {agent.agent_id}: {type(agent).__name__}")


logger.info("🏁 AUCTION SIMULATION STARTED")
logger.info(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("="*60)

# Run auction with verbose logging now that logging is configured
auction.run_all(auction_type=auction_type, per_ruolo=per_ruolo, verbose=True)

logger.info("\n" + "="*60)


logger.info("🏁 AUCTION COMPLETED!")
logger.info(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("="*60)

# Log all team information instead of creating Excel files
logger.info("\n" + "="*80)
logger.info("📊 FINAL TEAMS BY AGENT")
logger.info("="*80)

for agent in auction.agents:
    logger.info(f"\n🏆 TEAM: {agent.agent_id.upper()}")
    logger.info("-" * 50)
    
    # Log squad players
    for player in agent.squad:
        logger.info(f"  {player.role:3} | {player.name:20} | {player.team:15} | Eval: {player.evaluation:3} | Cost: {player.final_cost:3}")
    
    # Calculate squad objectives
    total_eval = agent.squad.objective(standardized=False)
    total_std_eval = agent.squad.objective(standardized=True)
    bestxi_eval = agent.squad.objective(bestxi=True, standardized=False)
    bestxi_std_eval = agent.squad.objective(bestxi=True, standardized=True)
    
    # Log summary metrics
    logger.info("\n📈 SQUAD METRICS:")
    logger.info(f"  Total Squad Evaluation: {total_eval}")
    logger.info(f"  Total Squad Standardized: {total_std_eval:.3f}")
    logger.info(f"  Best XI Evaluation: {bestxi_eval}")
    logger.info(f"  Best XI Standardized: {bestxi_std_eval:.3f}")
    logger.info(f"  Credits Remaining: {agent.current_credits}")
    logger.info(f"  Total Credits Spent: {1000 - agent.current_credits}")
    logger.info("")
    logger.info("="*60)


for agent in agents:
    logger.info(f"\n🏆 TEAM: {agent.agent_id.upper()}")
    logger.info("-" * 50)
            # Calculate squad objectives
    total_eval = agent.squad.objective(standardized=False)
    total_std_eval = agent.squad.objective(standardized=True)
    bestxi_eval = agent.squad.objective(bestxi=True, standardized=False)
    bestxi_std_eval = agent.squad.objective(bestxi=True, standardized=True)
            # Log summary metrics
    logger.info("\n📈 SQUAD METRICS:")
    logger.info(f"  Total Squad Evaluation: {total_eval}")
    logger.info(f"  Total Squad Standardized: {total_std_eval:.3f}")
    logger.info(f"  Best XI Evaluation: {bestxi_eval}")
    logger.info(f"  Best XI Standardized: {bestxi_std_eval:.3f}")
    logger.info(f"  Credits Remaining: {agent.current_credits}")
    logger.info(f"  Total Credits Spent: {1000 - agent.current_credits}")
    logger.info("")

logger.info(f"\n📁 Complete log saved to: {log_filename}")