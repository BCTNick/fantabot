"""
Test rapido per verificare che la GUI si avvii senza errori
"""

import sys
import os
import tkinter as tk
from unittest.mock import patch

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

def test_gui_startup():
    """Test che la GUI si avvii senza errori"""
    print("🖥️  Testing GUI startup...")
    
    try:
        # Create a temporary root to test imports and initialization
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        from enhanced_auction_gui import EnhancedFantaAuctionGUI
        
        # Initialize the GUI (but don't show it)
        app = EnhancedFantaAuctionGUI(root)
        
        # Test some basic operations
        config = app.config_tab.get_configuration()
        print(f"   ✅ Config valid: {config['valid']}")
        print(f"   ✅ Default agents: {len(config['agents'])}")
        
        # Test setting slots
        app.auction_tab.set_auction_slots({"GK": 3, "DEF": 8, "MID": 8, "ATT": 6})
        print(f"   ✅ Slots set correctly")
        
        # Clean up
        root.destroy()
        
        print("✅ GUI startup test successful!")
        return True
        
    except Exception as e:
        print(f"❌ GUI startup error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing GUI Startup")
    print("=" * 40)
    
    success = test_gui_startup()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 GUI startup test passed!")
        print("💡 You can now run: python launch_fantabot.py")
    else:
        print("❌ GUI startup test failed!")
    
    sys.exit(0 if success else 1)
