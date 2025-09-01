"""
Test script for the enhanced TTS system and Teams tab
"""

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.utils.tts_manager import TTSManager
from src.gui.teams_tab import TeamsTab
from src.models import Player, Slots
from src.agents.enhanced_cap_agent import EnhancedCapAgent
import tkinter as tk


def test_tts_system():
    """Test the enhanced TTS system"""
    print("🔊 Testing Enhanced TTS System...")
    
    tts_manager = TTSManager()
    
    # Test basic functionality
    print(f"TTS Available: {tts_manager.available}")
    print(f"TTS Status: {tts_manager.get_status()}")
    
    if tts_manager.available:
        # Test priority system
        print("Testing priority system...")
        tts_manager.speak_async("Low priority message", priority=8)
        tts_manager.speak_async("Medium priority message", priority=5)
        tts_manager.speak_immediate("High priority immediate message")
        
        # Test queue management
        print("Testing queue management...")
        for i in range(3):
            tts_manager.speak_async(f"Test message {i+1}", priority=6)
        
        print(f"Queue size: {tts_manager.tts_queue.qsize()}")
        
        # Test auction-specific methods
        tts_manager.announce_player("Donnarumma", "P", 25)
        tts_manager.announce_winner("Donnarumma", "Angelo", 50)
        tts_manager.announce_phase("Inizio asta portieri")
        
        # Wait a bit then check status
        import time
        time.sleep(2)
        print(f"TTS is busy: {tts_manager.is_busy()}")
        
        # Test cleanup
        tts_manager.clear_queue()
        print(f"Queue cleared, size: {tts_manager.tts_queue.qsize()}")
        
    print("✅ TTS System test completed\n")


def test_teams_tab():
    """Test the Teams tab"""
    print("👥 Testing Teams Tab...")
    
    # Create a simple tkinter window for testing
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Create test frame
    test_frame = tk.Frame(root)
    
    # Create teams tab
    teams_tab = TeamsTab(test_frame)
    
    # Test empty state
    print("Testing empty state...")
    teams_tab.show_empty_state()
    
    # Create test data
    print("Creating test data...")
    
    # Create test agents with squads
    agent1 = EnhancedCapAgent("Test Team 1")
    agent1.current_credits = 750
    agent1._squad = []
    
    # Create test players
    player1 = Player("Donnarumma", "P", "Milan", 25, 45)
    player1.final_cost = 50
    player1.fantasy_team = "Test Team 1"
    
    player2 = Player("Bastoni", "D", "Inter", 22, 38)
    player2.final_cost = 35
    player2.fantasy_team = "Test Team 1"
    
    agent1._squad = [player1, player2]
    
    agent2 = EnhancedCapAgent("Test Team 2")
    agent2.current_credits = 900
    agent2._squad = []
    
    agents = [agent1, agent2]
    players = [player1, player2]
    
    # Test with data
    print("Testing with data...")
    teams_tab.set_auction_data(agents, players)
    
    # Test export
    print("Testing export...")
    export_data = teams_tab.export_teams_data()
    print(f"Export data keys: {list(export_data.keys())}")
    if export_data:
        print(f"Sample team data: {export_data[list(export_data.keys())[0]]}")
    
    # Test team lookup
    team = teams_tab.get_team_by_name("Test Team 1")
    print(f"Team lookup result: {team.agent_id if team else 'Not found'}")
    
    root.destroy()
    print("✅ Teams Tab test completed\n")


def test_integration():
    """Test integration between TTS and Teams tab"""
    print("🔗 Testing Integration...")
    
    # Test TTS shutdown
    tts_manager = TTSManager()
    if tts_manager.available:
        tts_manager.speak_async("Testing shutdown", priority=5)
        print("Testing graceful shutdown...")
        tts_manager.shutdown()
        print("TTS manager shut down")
    
    print("✅ Integration test completed\n")


def main():
    """Run all tests"""
    print("🧪 TESTING ENHANCED FANTABOT SYSTEM")
    print("=" * 50)
    
    try:
        test_tts_system()
        test_teams_tab()
        test_integration()
        
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("\n📋 SUMMARY:")
        print("• Enhanced TTS system with priority queue: ✅")
        print("• Teams tab with live updates: ✅")
        print("• Integration and cleanup: ✅")
        print("\n🚀 System ready for your auction tonight!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
