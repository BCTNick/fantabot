#!/usr/bin/env python3
"""
Test script per verificare la compatibilità della GUI con i modelli
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

def test_data_loading():
    """Test loading players data"""
    try:
        from src.data_loader import load_players_from_excel
        
        print("🔍 Testing data loading...")
        players = load_players_from_excel()
        
        if players:
            player = players[0]
            print(f"✅ Loaded {len(players)} players")
            print(f"✅ First player: {player.name} ({player.role}) - Evaluation: {player.evaluation}")
            
            # Test all attributes we use in GUI
            assert hasattr(player, 'name'), "Player should have 'name' attribute"
            assert hasattr(player, 'role'), "Player should have 'role' attribute"
            assert hasattr(player, 'evaluation'), "Player should have 'evaluation' attribute"
            assert hasattr(player, 'team'), "Player should have 'team' attribute"
            
            print("✅ All player attributes are correct")
            return True
        else:
            print("❌ No players loaded")
            return False
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_agents():
    """Test agent creation"""
    try:
        from src.agents.agent_class import RandomAgent
        from src.agents.cap_based_agent import CapAgent
        
        print("\n🤖 Testing agents...")
        
        random_agent = RandomAgent("test_random")
        cap_agent = CapAgent("test_cap")
        
        print("✅ Agents created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error creating agents: {e}")
        return False

def test_auction():
    """Test auction creation"""
    try:
        from src.auction import Auction
        from src.models import Slots
        from src.agents.agent_class import RandomAgent
        from src.data_loader import load_players_from_excel
        
        print("\n🏛️ Testing auction...")
        
        # Create test components
        slots = Slots()
        agents = [RandomAgent("agent1"), RandomAgent("agent2")]
        players = load_players_from_excel()
        
        # Create auction
        auction = Auction(slots, agents, players, 1000)
        
        print("✅ Auction created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error creating auction: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running FantaBot GUI Compatibility Tests")
    print("=" * 50)
    
    tests = [
        test_data_loading,
        test_agents,
        test_auction
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    if all(results):
        print("🎉 All tests passed! GUI should work correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
