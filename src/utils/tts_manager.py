"""
Enhanced Text-to-Speech utilities with queue management
"""

import threading
import re
import time
import queue
from typing import Optional, List, Dict
import pyttsx3


class TTSManager:
    """Speech manager with queue management to prevent overlapping"""
    
    def __init__(self):
        self.engine = None
        self.available = False
        self.tts_queue = queue.PriorityQueue()  # Use PriorityQueue for proper priority handling
        self.is_speaking = False
        self.tts_thread = None
        self.stop_requested = False
        self.initialize_engine()
        
        if self.available:
            self.start_tts_worker()
    
    def initialize_engine(self):
        """Initialize TTS engine with enhanced settings"""
        try:
            # Initialize with driver specification for better compatibility
            try:
                self.engine = pyttsx3.init('nsss')  # macOS native speech system
            except:
                self.engine = pyttsx3.init()  # fallback to default
            
            # Enhanced voice settings for macOS
            voices = self.engine.getProperty('voices')
            if voices:
                # Find Italian voice first, then best system voice
                best_voice = None
                for voice in voices:
                    voice_name = voice.name.lower()
                    voice_id = getattr(voice, 'id', '').lower()
                    
                    # Prioritize Italian voices
                    if any(italian in voice_name or italian in voice_id 
                           for italian in ['alice', 'federica', 'luca', 'italian']):
                        best_voice = voice.id
                        print(f"🇮🇹 Found Italian voice: {voice.name}")
                        break
                    # Then prefer system voices that work well  
                    elif any(name in voice_name for name in ['samantha', 'alex', 'victoria', 'allison']):
                        best_voice = voice.id
                
                if best_voice:
                    self.engine.setProperty('voice', best_voice)
                    print("✅ Using Italian voice for TTS")
                else:
                    # Use first available voice as fallback
                    self.engine.setProperty('voice', voices[0].id)
            
            # Optimized settings to prevent corruption
            self.engine.setProperty('rate', 140)     # Slower for clarity
            self.engine.setProperty('volume', 0.7)   # Lower volume to prevent distortion
            
            # Test the engine with a simple phrase
            test_phrase = "Sistema pronto"
            self.engine.say(test_phrase)
            self.engine.runAndWait()
            
            self.available = True
            print("✅ TTS Engine initialized successfully")
            
        except Exception as e:
            self.engine = None
            self.available = False
            print(f"⚠️  TTS engine not available: {e}")
    
    def start_tts_worker(self):
        """Start the TTS worker thread"""
        if self.tts_thread and self.tts_thread.is_alive():
            return
        
        self.stop_requested = False
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()
    
    def _tts_worker(self):
        """Worker thread that processes TTS queue sequentially"""
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        while not self.stop_requested:
            try:
                # Wait for message with timeout
                message_data = self.tts_queue.get(timeout=1.0)
                
                if message_data is None:  # Shutdown signal
                    break
                
                priority, message = message_data
                
                if not message.strip():
                    continue
                
                # Set speaking flag
                self.is_speaking = True
                
                try:
                    # Ensure engine is ready
                    if not self.engine:
                        print("🔊 TTS engine not available, skipping message")
                        continue
                    
                    # DON'T stop the engine if already speaking - let it finish
                    # Only stop if there's a real problem
                    
                    # Brief pause to ensure engine is ready
                    time.sleep(0.1)
                    
                    # Speak the message
                    self.engine.say(message)
                    self.engine.runAndWait()
                    
                    # Pause between messages to prevent issues
                    time.sleep(0.4)
                    
                    # Reset consecutive errors on success
                    consecutive_errors = 0
                    
                except Exception as e:
                    consecutive_errors += 1
                    print(f"🔊 TTS Error (attempt {consecutive_errors}): {e}")
                    
                    # If too many consecutive errors, try to reinitialize
                    if consecutive_errors >= max_consecutive_errors:
                        print("🔊 Too many TTS errors, attempting to reinitialize...")
                        try:
                            self._reinitialize_engine()
                            consecutive_errors = 0
                        except Exception as reinit_error:
                            print(f"🔊 Failed to reinitialize TTS: {reinit_error}")
                            # Continue anyway, but with longer pause
                            time.sleep(1.0)
                    else:
                        # Short pause before retry
                        time.sleep(0.5)
                
                finally:
                    self.is_speaking = False
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"🔊 TTS Worker Error: {e}")
                self.is_speaking = False
                consecutive_errors += 1
                
                if consecutive_errors >= max_consecutive_errors:
                    print("🔊 Worker thread has too many errors, attempting recovery...")
                    time.sleep(2.0)
                    consecutive_errors = 0
    
    def speak_async(self, message: str, priority: int = 5):
        """
        Add message to TTS queue with priority
        Priority: 1 = highest (urgent), 10 = lowest (optional)
        """
        if not self.available or not message.strip():
            return
        
        # Health check before adding message
        if not self.health_check():
            return
        
        # Clear queue if it's getting too long (prevent backlog)
        if self.tts_queue.qsize() > 5:
            print("⚠️  TTS queue too long, clearing some old messages")
            # Clear a few messages to prevent infinite backlog
            temp_messages = []
            for _ in range(3):  # Keep only the 3 highest priority messages
                if not self.tts_queue.empty():
                    try:
                        temp_messages.append(self.tts_queue.get_nowait())
                    except queue.Empty:
                        break
            # Put back the highest priority messages
            for msg in temp_messages:
                self.tts_queue.put(msg)
        
        # Add to queue with priority as first element (lower number = higher priority)
        self.tts_queue.put((priority, message))
    
    def speak_immediate(self, message: str):
        """Speak message with highest priority without clearing queue"""
        if not self.available or not message.strip():
            return
        
        # Add message with highest priority (priority=1) to queue
        # This will be processed before lower priority messages
        self.speak_async(message, priority=1)
    
    def is_busy(self) -> bool:
        """Check if TTS is currently speaking or has messages in queue"""
        return self.is_speaking or not self.tts_queue.empty()
    
    def _reinitialize_engine(self):
        """Reinitialize TTS engine after errors"""
        print("🔧 Reinitializing TTS engine...")
        
        # Stop current engine
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
        
        # Brief pause
        time.sleep(0.5)
        
        # Reinitialize
        self.engine = None
        self.available = False
        
        try:
            # Initialize with driver specification for better compatibility
            try:
                self.engine = pyttsx3.init('nsss')  # macOS native speech system
            except:
                self.engine = pyttsx3.init()  # fallback to default
            
            # Enhanced voice settings for macOS
            voices = self.engine.getProperty('voices')
            if voices:
                # Find Italian voice first, then best system voice
                best_voice = None
                for voice in voices:
                    voice_name = voice.name.lower()
                    voice_id = getattr(voice, 'id', '').lower()
                    
                    # Prioritize Italian voices
                    if any(italian in voice_name or italian in voice_id 
                           for italian in ['alice', 'federica', 'luca', 'italian']):
                        best_voice = voice.id
                        print(f"🇮🇹 Found Italian voice: {voice.name}")
                        break
                    # Then prefer system voices that work well  
                    elif any(name in voice_name for name in ['samantha', 'alex', 'victoria', 'allison']):
                        best_voice = voice.id
                
                if best_voice:
                    self.engine.setProperty('voice', best_voice)
                    print("✅ Using Italian voice for TTS")
                else:
                    # Use first available voice as fallback
                    self.engine.setProperty('voice', voices[0].id)
            
            # Optimized settings to prevent corruption
            self.engine.setProperty('rate', 140)     # Slower for clarity
            self.engine.setProperty('volume', 0.7)   # Lower volume to prevent distortion
            
            self.available = True
            print("✅ TTS Engine reinitialized successfully")
            
        except Exception as e:
            self.engine = None
            self.available = False
            print(f"❌ Failed to reinitialize TTS engine: {e}")
    
    def clear_queue(self):
        """Clear all pending TTS messages"""
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
            except queue.Empty:
                break
    
    def shutdown(self):
        """Shutdown TTS manager"""
        self.stop_requested = True
        if self.tts_queue:
            self.tts_queue.put(None)  # Shutdown signal
        
        if self.tts_thread and self.tts_thread.is_alive():
            self.tts_thread.join(timeout=2.0)
    
    def clean_message_for_tts(self, message: str, agent_name: str) -> str:
        """Clean a log message for TTS output with enhanced Italian patterns"""
        # Remove emoji and special characters
        clean_msg = re.sub(r'[^\w\s\-:.,!?€]', '', message)
        
        # Extract relevant parts for different message types
        if "offre" in clean_msg.lower() and agent_name in clean_msg:
            # For bid messages like "💰 Angelo offre 50 crediti"
            match = re.search(rf'{re.escape(agent_name)}\s+offre\s+(\d+)\s+crediti', clean_msg)
            if match:
                amount = match.group(1)
                return f"{agent_name} offre {amount}"
        
        elif "non offre" in clean_msg.lower() and agent_name in clean_msg:
            # For no bid messages - skip these to reduce noise
            return ""  # Don't announce "no bid" to reduce clutter
        
        elif "assegnato a" in clean_msg.lower() and agent_name in clean_msg:
            # For assignment messages
            match = re.search(r'(\w+)\s+assegnato\s+a\s+' + re.escape(agent_name) + r'\s+per\s+(\d+)\s+crediti', clean_msg)
            if match:
                player_name = match.group(1)
                amount = match.group(2)
                return f"{player_name} a {agent_name} per {amount}"
        
        elif "GIOCATORE IN ASTA" in clean_msg.upper():
            # For new player announcements
            match = re.search(r'GIOCATORE IN ASTA:\s+(\w+)\s+\((\w+)\)', clean_msg, re.IGNORECASE)
            if match:
                player_name = match.group(1)
                role = match.group(2)
                # Translate role to Italian
                role_translation = {
                    "GK": "portiere", "P": "portiere",
                    "DEF": "difensore", "D": "difensore",
                    "MID": "centrocampista", "C": "centrocampista", 
                    "ATT": "attaccante", "A": "attaccante"
                }
                role_it = role_translation.get(role.upper(), role)
                return f"In asta: {player_name}, {role_it}"
        
        elif "inizia asta" in clean_msg.lower() or "fase" in clean_msg.lower():
            # For phase announcements
            if "portieri" in clean_msg.lower() or "GK" in clean_msg:
                return "Inizia asta portieri"
            elif "difensori" in clean_msg.lower() or "DEF" in clean_msg:
                return "Inizia asta difensori"
            elif "centrocampisti" in clean_msg.lower() or "MID" in clean_msg:
                return "Inizia asta centrocampisti"
            elif "attaccanti" in clean_msg.lower() or "ATT" in clean_msg:
                return "Inizia asta attaccanti"
        
        # If message contains agent name, return a short version
        if agent_name.lower() in clean_msg.lower() and len(clean_msg) < 100:
            # Clean up common patterns
            clean_msg = re.sub(r'^\d{2}:\d{2}:\d{2}', '', clean_msg).strip()  # Remove timestamps
            clean_msg = re.sub(r'\s+', ' ', clean_msg)  # Normalize spaces
            return clean_msg
        
        # If no specific pattern matches, return empty to avoid noise
        return ""
    
    def speak_for_agent(self, agent_name: str, message: str, agents_config: List[Dict], priority: int = 5):
        """Speak message if TTS is enabled for the agent"""
        if not self.available:
            return
        
        # Check if TTS is enabled for this agent
        agent_config = next((agent for agent in agents_config if agent["name"] == agent_name), None)
        if not agent_config or not agent_config.get("tts_enabled", False):
            return
        
        # Clean the message for TTS
        clean_message = self.clean_message_for_tts(message, agent_name)
        
        if clean_message:
            self.speak_async(clean_message, priority)
    
    def announce_bid(self, agent_name: str, amount: int, agents_config: List[Dict], priority: int = 4):
        """Announce a bid if TTS is enabled for the agent"""
        if not self.available or amount <= 0:
            return
        
        # Check if TTS is enabled for this agent
        agent_config = next((agent for agent in agents_config if agent["name"] == agent_name), None)
        
        if not agent_config or not agent_config.get("tts_enabled", False):
            return
        
        # Create clean bid message
        message = f"{agent_name} offre {amount}"
        self.speak_async(message, priority)
    
    def announce_no_bid(self, agent_name: str, agents_config: List[Dict]):
        """Skip no bid announcements to reduce noise"""
        # We intentionally don't announce "no bid" to keep TTS clean
        pass
    
    def announce_player(self, player_name: str, role: str, evaluation: int, priority: int = 2):
        """Announce player in auction with high priority"""
        if self.available:
            # Convert role to Italian if needed
            role_translation = {
                "GK": "portiere",
                "DEF": "difensore", 
                "MID": "centrocampista",
                "ATT": "attaccante"
            }
            role_it = role_translation.get(role, role)
            message = f"In asta: {player_name}, {role_it}"
            self.speak_async(message, priority)
    
    def announce_winner(self, player_name: str, winner: str, amount: int, priority: int = 1):
        """Announce auction winner with highest priority"""
        if self.available:
            message = f"{player_name} a {winner} per {amount}"
            self.speak_async(message, priority)
    
    def announce_phase(self, phase: str, priority: int = 1):
        """Announce auction phase (e.g., 'Inizio asta portieri')"""
        if self.available:
            self.speak_immediate(phase)
    
    def get_status(self) -> Dict:
        """Get TTS status information"""
        return {
            "available": self.available,
            "speaking": self.is_speaking,
            "queue_size": self.tts_queue.qsize() if self.tts_queue else 0,
            "worker_active": self.tts_thread.is_alive() if self.tts_thread else False,
            "engine_ready": self.engine is not None
        }
    
    def health_check(self) -> bool:
        """Check if TTS system is healthy"""
        if not self.available or not self.engine:
            return False
        
        # Check if worker thread is alive
        if not self.tts_thread or not self.tts_thread.is_alive():
            print("🔧 TTS worker thread not active, restarting...")
            self.start_tts_worker()
        
        # Check queue size
        if self.tts_queue.qsize() > 10:
            print("⚠️  TTS queue too large, clearing old messages")
            self._clear_low_priority_messages()
        
        return True
