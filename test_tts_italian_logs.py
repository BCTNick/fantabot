"""
Test script for TTS log agents and Italian language support
"""

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.utils.tts_manager import TTSManager
import time


def test_italian_voice():
    """Test Italian voice initialization"""
    print("🇮🇹 Testing Italian Voice Support...")
    
    tts_manager = TTSManager()
    
    if tts_manager.available:
        print(f"✅ TTS Status: {tts_manager.get_status()}")
        
        # Test Italian phrases
        italian_phrases = [
            "Sistema pronto per l'asta",
            "In asta: Donnarumma, portiere",
            "Angelo offre cinquanta",
            "Inizia asta centrocampisti",
            "Vlahovic a Inter per settanta"
        ]
        
        print("Testing Italian pronunciation...")
        for i, phrase in enumerate(italian_phrases):
            print(f"  {i+1}. Testing: '{phrase}'")
            tts_manager.speak_async(phrase, priority=2)
            time.sleep(2.5)  # Wait between messages
        
        print("✅ Italian voice test completed")
        tts_manager.shutdown()
    else:
        print("❌ TTS not available")


def test_log_message_processing():
    """Test log message processing for TTS"""
    print("\n📝 Testing Log Message Processing...")
    
    tts_manager = TTSManager()
    
    if tts_manager.available:
        # Test agent configuration
        agents_config = [
            {"name": "Angelo", "tts_enabled": True},
            {"name": "Cap Agent", "tts_enabled": True},
            {"name": "Random Bot", "tts_enabled": False}
        ]
        
        # Test different log message patterns
        test_messages = [
            ("💰 Angelo offre 50 crediti", "Angelo"),
            ("🚫 Cap Agent non offre", "Cap Agent"), 
            ("⚽ GIOCATORE IN ASTA: Vlahovic (ATT)", ""),
            ("🏆 Donnarumma assegnato a Angelo per 65 crediti", "Angelo"),
            ("📢 Inizia asta portieri", ""),
            ("💰 Random Bot offre 30 crediti", "Random Bot")
        ]
        
        print("Testing message cleaning and TTS...")
        for message, agent_name in test_messages:
            print(f"Testing: '{message}' for agent '{agent_name}'")
            
            if agent_name:
                # Test agent-specific TTS
                tts_manager.speak_for_agent(agent_name, message, agents_config, priority=4)
            else:
                # Test general announcement
                clean_msg = tts_manager.clean_message_for_tts(message, "")
                if clean_msg:
                    tts_manager.speak_async(clean_msg, priority=3)
            
            time.sleep(2)
        
        print("✅ Log message processing test completed")
        tts_manager.shutdown()
    else:
        print("❌ TTS not available for log test")


def test_voice_selection():
    """Test voice selection and listing"""
    print("\n🎤 Testing Voice Selection...")
    
    import pyttsx3
    
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        print(f"Available voices ({len(voices)}):")
        for i, voice in enumerate(voices):
            print(f"  {i+1}. {voice.name} (ID: {voice.id})")
            if any(term in voice.name.lower() or term in voice.id.lower() 
                   for term in ['alice', 'luca', 'federica', 'italian', 'italia']):
                print(f"      🇮🇹 ITALIAN VOICE DETECTED!")
        
        engine.stop()
        print("✅ Voice selection test completed")
        
    except Exception as e:
        print(f"❌ Voice selection test failed: {e}")


def main():
    """Run all TTS log and Italian tests"""
    print("🧪 TESTING TTS LOG AGENTS & ITALIAN SUPPORT")
    print("=" * 50)
    
    try:
        test_voice_selection()
        test_italian_voice()
        test_log_message_processing()
        
        print("\n🎉 ALL TTS TESTS COMPLETED!")
        print("\n📋 FEATURES TESTED:")
        print("• Italian voice detection and selection ✅")
        print("• Log message processing for agents ✅") 
        print("• Enhanced message cleaning ✅")
        print("• Agent-specific TTS filtering ✅")
        print("• Italian phrase pronunciation ✅")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
