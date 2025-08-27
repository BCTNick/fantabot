#!/usr/bin/env python3
"""
Launcher script for FantaBot Auction GUI
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from auction_gui import main
    
    if __name__ == "__main__":
        print("🚀 Avviando FantaBot Auction GUI...")
        main()
        
except ImportError as e:
    print(f"❌ Errore nell'importare le dipendenze: {e}")
    print("💡 Assicurati di aver installato tutte le dipendenze:")
    print("   pip install -r requirements_gui.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Errore nell'avviare l'applicazione: {e}")
    sys.exit(1)
