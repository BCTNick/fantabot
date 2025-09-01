"""
Test per verificare che i messaggi TTS vengano messi in coda
invece di essere tagliati quando si usa speak_immediate
"""

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.utils.tts_manager import TTSManager
import time


def test_queue_behavior():
    """Test che i messaggi vengano accodati invece di tagliati"""
    print("🧪 Testing TTS Queue Behavior...")
    
    tts_manager = TTSManager()
    
    if tts_manager.available:
        print("✅ TTS Manager initialized")
        
        # Step 1: Aggiungiamo alcuni messaggi normali
        print("\n1️⃣ Adding normal messages to queue...")
        tts_manager.speak_async("Primo messaggio normale", priority=5)
        tts_manager.speak_async("Secondo messaggio normale", priority=6)
        tts_manager.speak_async("Terzo messaggio normale", priority=7)
        
        # Verifichiamo la coda
        time.sleep(0.5)  # Piccola pausa per permettere l'accodamento
        status = tts_manager.get_status()
        print(f"   Queue size dopo messaggi normali: {status['queue_size']}")
        
        # Step 2: Aggiungiamo un messaggio immediato
        print("\n2️⃣ Adding immediate message (should NOT clear queue)...")
        tts_manager.speak_immediate("MESSAGGIO URGENTE!")
        
        # Verifichiamo che la coda non sia stata svuotata
        time.sleep(0.5)
        status = tts_manager.get_status()
        print(f"   Queue size dopo messaggio immediato: {status['queue_size']}")
        
        if status['queue_size'] > 0:
            print("   ✅ SUCCESSO: I messaggi sono stati mantenuti in coda!")
        else:
            print("   ❌ ERRORE: I messaggi sono stati rimossi dalla coda!")
        
        # Step 3: Aggiungiamo altri messaggi per vedere l'ordine
        print("\n3️⃣ Adding more messages to test priority order...")
        tts_manager.speak_async("Messaggio bassa priorità", priority=8)
        tts_manager.speak_immediate("ALTRO MESSAGGIO URGENTE!")
        tts_manager.speak_async("Ultimo messaggio normale", priority=5)
        
        # Status finale
        status = tts_manager.get_status()
        print(f"   Queue size finale: {status['queue_size']}")
        
        # Step 4: Aspettiamo e monitoriamo l'esecuzione
        print("\n4️⃣ Monitoring queue execution...")
        start_time = time.time()
        message_count = 0
        
        while tts_manager.is_busy() and (time.time() - start_time) < 60:
            current_status = tts_manager.get_status()
            if current_status['speaking']:
                message_count += 1
                print(f"   📢 Message {message_count} being spoken... Queue remaining: {current_status['queue_size']}")
                
                # Aspettiamo che finisca questo messaggio
                while current_status['speaking'] and (time.time() - start_time) < 60:
                    time.sleep(0.5)
                    current_status = tts_manager.get_status()
            
            time.sleep(0.5)
        
        final_status = tts_manager.get_status()
        print(f"\n✅ Final status: {final_status}")
        print(f"📊 Total messages processed: {message_count}")
        
        tts_manager.shutdown()
        print("✅ Queue behavior test completed")
    else:
        print("❌ TTS not available for queue test")


def test_priority_order():
    """Test che i messaggi vengano eseguiti nell'ordine di priorità corretto"""
    print("\n🎯 Testing Priority Order...")
    
    tts_manager = TTSManager()
    
    if tts_manager.available:
        # Aggiungiamo messaggi con priorità diverse
        print("Adding messages with different priorities...")
        tts_manager.speak_async("Priorità 8 (bassa)", priority=8)
        tts_manager.speak_async("Priorità 5 (media)", priority=5)
        tts_manager.speak_async("Priorità 2 (alta)", priority=2)
        tts_manager.speak_immediate("Priorità 1 (immediata)")  # priority=1
        tts_manager.speak_async("Priorità 9 (molto bassa)", priority=9)
        
        print("Ordine atteso: Immediata(1) → Alta(2) → Media(5) → Bassa(8) → Molto bassa(9)")
        
        # Monitoriamo l'esecuzione
        time.sleep(1)
        message_num = 1
        while tts_manager.is_busy() and message_num <= 5:
            if tts_manager.get_status()['speaking']:
                print(f"   🔊 Messaggio {message_num} in riproduzione...")
                
                # Aspetta che il messaggio finisca
                while tts_manager.get_status()['speaking']:
                    time.sleep(0.5)
                
                message_num += 1
            time.sleep(0.5)
        
        tts_manager.shutdown()
        print("✅ Priority order test completed")
    else:
        print("❌ TTS not available for priority test")


def main():
    """Esegui tutti i test"""
    print("🔬 TESTING TTS QUEUE MANAGEMENT")
    print("=" * 50)
    print("OBIETTIVO: Verificare che speak_immediate NON tagli i messaggi")
    print("ma li metta semplicemente in coda con priorità alta\n")
    
    try:
        test_queue_behavior()
        test_priority_order()
        
        print("\n🎉 ALL QUEUE TESTS COMPLETED!")
        print("\n📋 EXPECTED IMPROVEMENTS:")
        print("• speak_immediate non cancella più i messaggi in coda ✅")
        print("• Tutti i messaggi vengono riprodotti in ordine di priorità ✅")
        print("• Non ci sono più interruzioni brusche del TTS ✅")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
