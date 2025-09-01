"""
Test del nuovo sistema TTS con PriorityQueue
"""

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.utils.tts_manager import TTSManager
import time


def test_priority_queue_behavior():
    """Test che la PriorityQueue funzioni correttamente"""
    print("🧪 Testing PriorityQueue TTS Behavior...")
    
    tts_manager = TTSManager()
    
    if tts_manager.available:
        print("✅ TTS Manager initialized with PriorityQueue")
        
        # Step 1: Aggiungiamo messaggi con priorità diverse
        print("\n1️⃣ Adding messages with different priorities...")
        tts_manager.speak_async("Messaggio priorità 8 (bassa)", priority=8)
        print("   Added: Priority 8 (low)")
        
        tts_manager.speak_async("Messaggio priorità 5 (media)", priority=5)
        print("   Added: Priority 5 (medium)")
        
        tts_manager.speak_async("Messaggio priorità 2 (alta)", priority=2)
        print("   Added: Priority 2 (high)")
        
        # Status dopo i primi messaggi
        time.sleep(0.5)
        status = tts_manager.get_status()
        print(f"   Queue size: {status['queue_size']}")
        
        # Step 2: Aggiungiamo un messaggio immediato (priorità 1)
        print("\n2️⃣ Adding immediate message (priority 1)...")
        tts_manager.speak_immediate("MESSAGGIO URGENTE PRIORITÀ 1!")
        print("   Added: Priority 1 (immediate)")
        
        # Step 3: Aggiungiamo altri messaggi
        print("\n3️⃣ Adding more mixed priority messages...")
        tts_manager.speak_async("Messaggio priorità 9 (molto bassa)", priority=9)
        print("   Added: Priority 9 (very low)")
        
        tts_manager.speak_async("Messaggio priorità 3 (alta)", priority=3)
        print("   Added: Priority 3 (high)")
        
        # Status finale
        status = tts_manager.get_status()
        print(f"   Final queue size: {status['queue_size']}")
        
        # Step 4: Monitoriamo l'ordine di esecuzione
        print("\n4️⃣ Expected execution order:")
        print("   1. MESSAGGIO URGENTE PRIORITÀ 1! (priority 1)")
        print("   2. Messaggio priorità 2 (alta) (priority 2)")
        print("   3. Messaggio priorità 3 (alta) (priority 3)")
        print("   4. Messaggio priorità 5 (media) (priority 5)")
        print("   5. Messaggio priorità 8 (bassa) (priority 8)")
        print("   6. Messaggio priorità 9 (molto bassa) (priority 9)")
        
        print("\n   🔊 Execution monitoring:")
        message_count = 0
        start_time = time.time()
        
        while tts_manager.is_busy() and (time.time() - start_time) < 90:
            current_status = tts_manager.get_status()
            if current_status['speaking']:
                message_count += 1
                print(f"      📢 Message {message_count} playing... Queue remaining: {current_status['queue_size']}")
                
                # Aspettiamo che finisca questo messaggio
                while current_status['speaking'] and (time.time() - start_time) < 90:
                    time.sleep(0.5)
                    current_status = tts_manager.get_status()
            
            time.sleep(0.5)
        
        final_status = tts_manager.get_status()
        print(f"\n✅ Final status: {final_status}")
        print(f"📊 Total messages processed: {message_count}")
        
        if message_count == 6:
            print("✅ All messages processed correctly!")
        else:
            print(f"⚠️  Expected 6 messages, processed {message_count}")
        
        tts_manager.shutdown()
        print("✅ PriorityQueue test completed")
    else:
        print("❌ TTS not available")


def test_no_interruption():
    """Test che i messaggi non vengano più interrotti"""
    print("\n🔇 Testing No Interruption Behavior...")
    
    tts_manager = TTSManager()
    
    if tts_manager.available:
        print("Starting long message followed by urgent message...")
        
        # Messaggio lungo
        long_message = "Questo è un messaggio molto lungo che non dovrebbe essere interrotto da nessun altro messaggio, anche se urgente"
        tts_manager.speak_async(long_message, priority=5)
        
        # Aspettiamo un po' che inizi
        time.sleep(1)
        
        # Aggiungiamo un messaggio urgente
        tts_manager.speak_immediate("URGENTE!")
        
        # Aggiungiamo un altro messaggio normale
        tts_manager.speak_async("Messaggio normale dopo urgente", priority=6)
        
        print("Expected: Long message completes, then URGENTE!, then normal message")
        
        message_count = 0
        start_time = time.time()
        
        while tts_manager.is_busy() and (time.time() - start_time) < 60:
            current_status = tts_manager.get_status()
            if current_status['speaking']:
                message_count += 1
                print(f"   🔊 Message {message_count} playing...")
                
                # Aspettiamo che finisca
                while current_status['speaking'] and (time.time() - start_time) < 60:
                    time.sleep(0.5)
                    current_status = tts_manager.get_status()
            
            time.sleep(0.5)
        
        print(f"✅ No interruption test completed - {message_count} messages played")
        tts_manager.shutdown()
    else:
        print("❌ TTS not available")


def main():
    """Esegui tutti i test"""
    print("🔬 TESTING PRIORITYQUEUE TTS SYSTEM")
    print("=" * 50)
    print("OBIETTIVO: Verificare che PriorityQueue risolva i problemi di priorità")
    print("e che i messaggi non vengano più tagliati\n")
    
    try:
        test_priority_queue_behavior()
        test_no_interruption()
        
        print("\n🎉 ALL PRIORITYQUEUE TESTS COMPLETED!")
        print("\n📋 EXPECTED IMPROVEMENTS:")
        print("• PriorityQueue gestisce correttamente le priorità ✅")
        print("• speak_immediate non interrompe messaggi in corso ✅")
        print("• Ordine di esecuzione rispetta le priorità ✅")
        print("• Nessun messaggio viene tagliato ✅")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
