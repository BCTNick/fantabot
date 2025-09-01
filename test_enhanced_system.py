"""
Test script per verificare il funzionamento del sistema migliorato
"""

import sys
import os
import traceback

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

def test_imports():
    """Test delle importazioni"""
    print("🔧 Testing imports...")
    
    try:
        # Test core imports
        from src.models import Player, Squad, Slots
        from src.data_loader import load_players_from_excel
        print("✅ Core models: OK")
        
        # Test agents
        from src.agents.agent_class import Agent, RandomAgent
        from src.agents.cap_based_agent import CapAgent
        from src.agents.enhanced_cap_agent import EnhancedCapAgent
        print("✅ Agents: OK")
        
        # Test utils
        from src.utils.logging_handler import AuctionLogger
        from src.utils.tts_manager import TTSManager
        from src.utils.validators import AuctionValidator
        from src.utils.file_manager import FileManager
        print("✅ Utils: OK")
        
        # Test core
        from src.core.enhanced_auction import EnhancedAuction
        print("✅ Core: OK")
        
        # Test GUI components (non-visual test)
        print("✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_data_loading():
    """Test caricamento dati"""
    print("\n📊 Testing data loading...")
    
    try:
        from src.data_loader import load_players_from_excel
        players = load_players_from_excel()
        
        if players:
            print(f"✅ Loaded {len(players)} players")
            
            # Test player structure
            first_player = players[0]
            print(f"   Sample player: {first_player.name} ({first_player.role}) - {first_player.evaluation}")
            
            # Test roles distribution
            roles = {}
            for player in players:
                roles[player.role] = roles.get(player.role, 0) + 1
            
            print(f"   Roles distribution: {roles}")
            return True
        else:
            print("❌ No players loaded")
            return False
            
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_enhanced_cap_agent():
    """Test Enhanced Cap Agent"""
    print("\n🤖 Testing Enhanced Cap Agent...")
    
    try:
        from src.agents.enhanced_cap_agent import EnhancedCapAgent
        from src.models import Slots
        from src.data_loader import load_players_from_excel
        
        # Create agent
        agent = EnhancedCapAgent(
            agent_id="TestBot",
            cap_strategy="adaptive",
            aggression_level=0.5
        )
        
        # Initialize with test data
        players = load_players_from_excel()[:50]  # Use subset for testing
        slots = Slots(gk=3, def_=8, mid=8, att=6)
        agent.initialize(players, slots, 1000, 4)
        
        print(f"✅ Agent initialized with {len(agent.players_caps)} player caps")
        
        # Test strategy summary
        summary = agent.get_strategy_summary()
        print(f"   Strategy: {summary['strategy']}")
        print(f"   Aggression: {summary['aggression_level']}")
        print(f"   Target players: {summary['target_players']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Cap Agent error: {e}")
        traceback.print_exc()
        return False

def test_validators():
    """Test validation system"""
    print("\n✅ Testing validators...")
    
    try:
        from src.utils.validators import AuctionValidator
        from src.models import Slots
        
        # Test agent config validation
        agents_config = [
            {"name": "Test1", "type": "Human Agent"},
            {"name": "Test2", "type": "Cap Agent"}
        ]
        
        is_valid, error = AuctionValidator.validate_agent_config(agents_config)
        print(f"   Agent config validation: {'✅ Valid' if is_valid else f'❌ {error}'}")
        
        # Test auction settings validation
        slots = Slots(gk=3, def_=8, mid=8, att=6)
        is_valid, error = AuctionValidator.validate_auction_settings(1000, slots)
        print(f"   Settings validation: {'✅ Valid' if is_valid else f'❌ {error}'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validators error: {e}")
        return False

def test_tts_manager():
    """Test TTS Manager"""
    print("\n🔊 Testing TTS Manager...")
    
    try:
        from src.utils.tts_manager import TTSManager
        
        tts = TTSManager()
        print(f"   TTS Available: {'✅ Yes' if tts.available else '❌ No'}")
        
        if tts.available:
            # Test message cleaning
            test_message = "💰 Angelo offre 50 crediti"
            clean_msg = tts.clean_message_for_tts(test_message, "Angelo")
            print(f"   Message cleaning: '{clean_msg}'")
        
        return True
        
    except Exception as e:
        print(f"❌ TTS Manager error: {e}")
        return False

def run_all_tests():
    """Esegue tutti i test"""
    print("🚀 Starting FantaBot Enhanced System Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_loading,
        test_enhanced_cap_agent,
        test_validators,
        test_tts_manager
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
    print(f"🏆 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for use.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
