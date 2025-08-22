from src.models import Player, Squad, Slots
from src.agents.agent_class import Agent

agent = Agent("gigi")
players = [Player(name="Cristiano Ronaldo", team="Al Nassr", role="ATT", evaluation=95),
           Player(name="Lionel Messi", team="Inter Miami", role="ATT", evaluation=94),
           Player(name="Kevin De Bruyne", team="Manchester City", role="MID", evaluation=92)]

agent.squad.append(players)
# Get all remaining slots as a dictionary
all_remaining = agent.squad.get_remaining_slots(agent.slots)
print(all_remaining)
# Output: {'gk': 2, 'def': 5, 'mid': 6, 'att': 3, 'total': 16}

# Get remaining slots for specific positions
remaining_gk = agent.squad.get_remaining_slots(agent.slots, 'gk')
print(f"Remaining GK slots: {remaining_gk}")  # Output: 2

remaining_def = agent.squad.get_remaining_slots(agent.slots, 'def')
print(f"Remaining DEF slots: {remaining_def}")  # Output: 5

remaining_mid = agent.squad.get_remaining_slots(agent.slots, 'mid')
print(f"Remaining MID slots: {remaining_mid}")  # Output: 6

remaining_att = agent.squad.get_remaining_slots(agent.slots, 'att')
print(f"Remaining ATT slots: {remaining_att}")  # Output: 3

# Get total remaining slots
total_remaining = agent.squad.get_remaining_slots(agent.slots, 'total')
print(f"Total remaining slots: {total_remaining}")  # Output: 16


