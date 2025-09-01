"""
Test specifico per verificare la correzione del bug degli slots
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

def test_enhanced_auction_slots_fix():
    """Test che la correzione degli slots funzioni"""
    print("🔧 Testing Enhanced Auction slots fix...")
    
    try:
        from src.core.enhanced_auction import EnhancedAuction
        from src.models import Slots, Player
        from src.agents.agent_class import RandomAgent
        from src.agents.enhanced_cap_agent import EnhancedCapAgent
        
        # Create test data
        slots = Slots(gk=3, def_=8, mid=8, att=6)
        
        # Create test players
        test_players = [
            Player("Test GK", "Test Team", "GK", 80),
            Player("Test DEF", "Test Team", "DEF", 75),
            Player("Test MID", "Test Team", "MID", 85),
            Player("Test ATT", "Test Team", "ATT", 90)
        ]
        
        # Create test agents
        agents = [
            RandomAgent("Random1"),
            EnhancedCapAgent("Enhanced1", cap_strategy="adaptive")
        ]
        
        # Create auction
        auction = EnhancedAuction(
            slots=slots,
            agents=agents,
            players=test_players,
            initial_credits=1000
        )
        
        # Test che gli attributi siano corretti
        print(f"   ✅ auction.slots type: {type(auction.slots)}")
        print(f"   ✅ auction.slots_dict type: {type(auction.slots_dict)}")
        
        # Test accesso agli attributi
        print(f"   ✅ slots.gk: {auction.slots.gk}")
        print(f"   ✅ slots_dict['GK']: {auction.slots_dict['GK']}")
        
        # Test can_participate_in_bid
        test_agent = agents[0]
        can_participate = auction.can_participate_in_bid(
            test_agent, 50, "GK", auction.slots_dict
        )
        print(f"   ✅ can_participate_in_bid: {can_participate}")
        
        print("✅ Enhanced Auction slots fix working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Auction slots fix error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Enhanced Auction Slots Fix")
    print("=" * 50)
    
    success = test_enhanced_auction_slots_fix()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Slots fix test passed! Ready for auction!")
    else:
        print("❌ Slots fix test failed!")
    
    sys.exit(0 if success else 1)
