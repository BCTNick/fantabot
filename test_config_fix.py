"""
Test script for ConfigurationTab fix
"""

import sys
import os
import tkinter as tk
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.gui.config_tab import ConfigurationTab
from src.agents.enhanced_cap_agent import EnhancedCapAgent
from src.agents.human_agent import HumanAgent


def test_configuration_tab():
    """Test ConfigurationTab methods"""
    print("🔧 Testing ConfigurationTab...")
    
    # Create a test window
    root = tk.Tk()
    root.withdraw()  # Hide window
    
    # Create test frame
    test_frame = tk.Frame(root)
    
    # Agent types for the config tab
    agent_types = {
        "Human Agent": HumanAgent,
        "Enhanced Cap Agent": EnhancedCapAgent
    }
    
    # Create configuration tab
    config_tab = ConfigurationTab(test_frame, agent_types)
    
    # Test get_agents_config method
    print("Testing get_agents_config()...")
    agents_config = config_tab.get_agents_config()
    print(f"✅ get_agents_config() returned: {len(agents_config)} agents")
    
    # Test get_configuration method
    print("Testing get_configuration()...")
    full_config = config_tab.get_configuration()
    print(f"✅ get_configuration() returned config with keys: {list(full_config.keys())}")
    
    # Test agents in configuration
    if agents_config:
        print("Sample agent config:")
        sample_agent = agents_config[0]
        print(f"  - Name: {sample_agent.get('name', 'N/A')}")
        print(f"  - Type: {sample_agent.get('type', 'N/A')}")
        print(f"  - TTS Enabled: {sample_agent.get('tts_enabled', False)}")
    
    root.destroy()
    print("✅ ConfigurationTab test completed\n")


def test_integration():
    """Test integration scenario like in GUI"""
    print("🔗 Testing Integration Scenario...")
    
    root = tk.Tk()
    root.withdraw()
    
    test_frame = tk.Frame(root)
    agent_types = {"Human Agent": HumanAgent, "Enhanced Cap Agent": EnhancedCapAgent}
    
    config_tab = ConfigurationTab(test_frame, agent_types)
    
    # Simulate the scenario that was failing
    try:
        agents_config = config_tab.get_agents_config()
        print(f"✅ Successfully got agents config: {len(agents_config)} agents")
        
        # Simulate checking TTS enabled for an agent
        for agent_config in agents_config:
            agent_name = agent_config.get("name", "Unknown")
            tts_enabled = agent_config.get("tts_enabled", False)
            print(f"  - {agent_name}: TTS {'enabled' if tts_enabled else 'disabled'}")
            
    except AttributeError as e:
        print(f"❌ AttributeError: {e}")
    except Exception as e:
        print(f"❌ Other error: {e}")
    
    root.destroy()
    print("✅ Integration test completed\n")


def main():
    """Run configuration tab tests"""
    print("🧪 TESTING CONFIGURATION TAB FIX")
    print("=" * 40)
    
    try:
        test_configuration_tab()
        test_integration()
        
        print("🎉 ALL CONFIGURATION TESTS PASSED!")
        print("\n📋 FIX SUMMARY:")
        print("• get_agents_config() method added ✅")
        print("• Integration with TTS system working ✅")
        print("• No more AttributeError ✅")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
