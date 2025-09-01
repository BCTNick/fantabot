#!/usr/bin/env python3
"""
Launcher per FantaBot Enhanced - Sistema Avanzato per Aste Fantacalcio
"""

import sys
import os
import subprocess
from pathlib import Path

def check_dependencies():
    """Controlla le dipendenze necessarie"""
    required_packages = [
        'tkinter',
        'pandas', 
        'openpyxl',
        'pyttsx3'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            elif package == 'pandas':
                import pandas
            elif package == 'openpyxl':
                import openpyxl
            elif package == 'pyttsx3':
                import pyttsx3
        except ImportError:
            missing.append(package)
    
    return missing

def install_dependencies():
    """Installa le dipendenze mancanti"""
    print("📦 Installing missing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            'pandas>=1.3.0', 'openpyxl>=3.0.7', 'pyttsx3>=2.90'
        ])
        return True
    except subprocess.CalledProcessError:
        return False

def check_data_file():
    """Controlla se esiste il file dati"""
    data_file = Path("data/players_list.xlsx")
    return data_file.exists()

def main():
    """Funzione principale di avvio"""
    print("🏆 FantaBot Enhanced - Sistema Avanzato Aste Fantacalcio")
    print("=" * 60)
    
    # Check se siamo nella directory corretta
    if not Path("enhanced_auction_gui.py").exists():
        print("❌ Errore: Esegui questo script dalla directory principale del progetto")
        sys.exit(1)
    
    # Check dipendenze
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"⚠️  Dipendenze mancanti: {', '.join(missing_deps)}")
        
        if input("Installare automaticamente? (y/n): ").lower() == 'y':
            if install_dependencies():
                print("✅ Dipendenze installate con successo!")
            else:
                print("❌ Errore nell'installazione delle dipendenze")
                print("Installa manualmente con: pip install -r requirements_gui.txt")
                sys.exit(1)
        else:
            print("Installa le dipendenze e riprova.")
            sys.exit(1)
    
    # Check file dati
    if not check_data_file():
        print("⚠️  File dati non trovato: data/players_list.xlsx")
        print("   Assicurati di avere il file Excel con i giocatori nella cartella data/")
        
        if input("Continuare comunque? (y/n): ").lower() != 'y':
            sys.exit(1)
    
    # Avvia l'applicazione
    print("\n🚀 Avvio FantaBot Enhanced...")
    print("   💡 Suggerimento: Usa l'Enhanced Cap Agent per la migliore strategia!")
    print("   🔊 TTS disponibile per annunci vocali")
    print("   💾 Ricorda di salvare la configurazione per usi futuri")
    print("\n" + "=" * 60)
    
    try:
        # Import e avvio GUI
        from enhanced_auction_gui import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"❌ Errore nell'importazione: {e}")
        print("Verifica che tutti i file siano presenti nella cartella src/")
    except Exception as e:
        print(f"❌ Errore nell'avvio dell'applicazione: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
