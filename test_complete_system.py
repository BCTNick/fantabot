"""
Test completo per simulare l'avvio di un'asta
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

def test_auction_startup():
    """Test completo dell'avvio asta"""
    print("🏆 Testing complete auction startup...")
    
    try:
        from src.core.enhanced_auction import EnhancedAuction
        from src.models import Slots
        from src.agents.enhanced_cap_agent import EnhancedCapAgent
        from src.agents.agent_class import RandomAgent
        from src.data_loader import load_players_from_excel
        from src.utils.logging_handler import AuctionLogger
        import queue
        
        # Setup logging
        log_queue = queue.Queue()
        logger = AuctionLogger(log_queue)
        
        print("   📊 Loading players...")
        players = load_players_from_excel()
        print(f"   ✅ Loaded {len(players)} players")
        
        print("   🤖 Creating agents...")
        agents = [
            EnhancedCapAgent("TestCap", cap_strategy="adaptive", aggression_level=0.5),
            RandomAgent("TestRandom")
        ]
        print(f"   ✅ Created {len(agents)} agents")
        
        print("   ⚙️  Creating auction...")
        slots = Slots(gk=3, def_=8, mid=8, att=6)
        auction = EnhancedAuction(
            slots=slots,
            agents=agents,
            players=players[:50],  # Use subset for testing
            initial_credits=1000
        )
        print("   ✅ Auction created successfully")
        
        print("   🔍 Testing auction properties...")
        print(f"      - slots type: {type(auction.slots)}")
        print(f"      - slots_dict type: {type(auction.slots_dict)}")
        print(f"      - can access slots.gk: {auction.slots.gk}")
        print(f"      - can access slots_dict['GK']: {auction.slots_dict['GK']}")
        
        print("   🧪 Testing bid validation...")
        test_agent = agents[0]
        test_player = players[0]
        can_bid = auction.can_participate_in_bid(test_agent, 50, test_player.role, auction.slots_dict)
        print(f"      - Agent can bid: {can_bid}")
        
        print("   📈 Testing auction summary...")
        summary = auction.get_auction_summary()
        print(f"      - Total players: {summary['total_players']}")
        print(f"      - Agents count: {len(summary['agents'])}")
        
        # Test a few rounds of auction without running full auction
        print("   🎯 Testing single player auction setup...")
        test_player = players[0]
        auction.current_player = test_player
        auction.current_price = 1
        auction.highest_bidder = None
        print(f"      - Current player: {auction.current_player.name}")
        print(f"      - Current price: {auction.current_price}")
        
        print("✅ Complete auction startup test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Auction startup error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_auction_integration():
    """Test integrazione GUI-Auction"""
    print("\n🖥️  Testing GUI-Auction integration...")
    
    try:
        import tkinter as tk
        from enhanced_auction_gui import EnhancedFantaAuctionGUI
        
        # Create minimal GUI test
        root = tk.Tk()
        root.withdraw()
        
        app = EnhancedFantaAuctionGUI(root)
        
        # Get configuration
        config = app.config_tab.get_configuration()
        print(f"   ✅ Config obtained: {config['valid']}")
        
        # Test that we could create an auction (without actually starting it)
        if config['valid']:
            from src.models import Slots
            from src.data_loader import load_players_from_excel
            
            players = load_players_from_excel()[:10]  # Small subset
            agents = app.create_agents(config["agents"][:2])  # Just 2 agents
            slots_config = config["settings"]["slots"]
            slots = Slots(**slots_config)
            
            print(f"   ✅ Could create {len(agents)} agents")
            print(f"   ✅ Slots configuration: {slots_config}")
        
        root.destroy()
        print("✅ GUI-Auction integration test successful!")
        return True
        
    except Exception as e:
        print(f"❌ GUI-Auction integration error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Complete Auction System Test")
    print("=" * 50)
    
    tests = [
        test_auction_startup,
        test_gui_auction_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"🏆 Final Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("")
        print("✅ System is ready for your auction tonight!")
        print("💡 Run: python launch_fantabot.py")
        print("🤖 Use Enhanced Cap Agent for best strategy!")
        print("🔊 Enable TTS for voice announcements!")
    else:
        print("⚠️  Some tests failed. Check errors above.")
    
    sys.exit(0 if passed == total else 1)
