"""
Test semplice per debug del TTS
"""

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.utils.tts_manager import TTSManager
import time


def simple_debug_test():
    """Test semplice per capire cosa succede"""
    print("🔍 SIMPLE DEBUG TEST")
    
    tts_manager = TTSManager()
    
    if tts_manager.available:
        print("✅ TTS available")
        
        # Test 1: Un solo messaggio
        print("\n--- Test 1: Single message ---")
        tts_manager.speak_async("Primo messaggio", priority=5)
        
        # Aspettiamo e monitoriamo
        for i in range(20):  # 10 secondi max
            status = tts_manager.get_status()
            print(f"  {i:2d}s: Speaking={status['speaking']}, Queue={status['queue_size']}, Worker={status['worker_active']}")
            
            if not status['speaking'] and status['queue_size'] == 0:
                print("  ✅ First message completed")
                break
            time.sleep(0.5)
        
        # Test 2: Due messaggi
        print("\n--- Test 2: Two messages ---")
        tts_manager.speak_async("Secondo messaggio", priority=5)
        tts_manager.speak_async("Terzo messaggio", priority=5)
        
        # Aspettiamo e monitoriamo
        for i in range(30):  # 15 secondi max
            status = tts_manager.get_status()
            print(f"  {i:2d}s: Speaking={status['speaking']}, Queue={status['queue_size']}, Worker={status['worker_active']}")
            
            if not status['speaking'] and status['queue_size'] == 0:
                print("  ✅ All messages completed")
                break
            time.sleep(0.5)
        
        # Test 3: Priorità diverse
        print("\n--- Test 3: Different priorities ---")
        tts_manager.speak_async("Priorità bassa", priority=8)
        tts_manager.speak_async("Priorità alta", priority=2)
        
        # Aspettiamo e monitoriamo
        for i in range(30):  # 15 secondi max
            status = tts_manager.get_status()
            print(f"  {i:2d}s: Speaking={status['speaking']}, Queue={status['queue_size']}, Worker={status['worker_active']}")
            
            if not status['speaking'] and status['queue_size'] == 0:
                print("  ✅ Priority messages completed")
                break
            time.sleep(0.5)
        
        tts_manager.shutdown()
        print("✅ Debug test completed")
    else:
        print("❌ TTS not available")


if __name__ == "__main__":
    simple_debug_test()
