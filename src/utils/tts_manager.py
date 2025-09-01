"""
Text-to-Speech utilities for the auction system
"""

import threading
import re
from typing import Optional, List, Dict
import pyttsx3


class TTSManager:
    """Text-to-Speech manager for auction events"""
    
    def __init__(self):
        self.engine = None
        self.available = False
        self.initialize_engine()
    
    def initialize_engine(self):
        """Initialize TTS engine"""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)  # Speed of speech
            self.engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
            self.available = True
        except Exception as e:
            self.engine = None
            self.available = False
            print(f"Warning: TTS engine not available - {e}")
    
    def speak_async(self, message: str):
        """Speak message asynchronously"""
        if not self.available or not message.strip():
            return
        
        def tts_worker():
            try:
                self.engine.say(message)
                self.engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")
        
        thread = threading.Thread(target=tts_worker, daemon=True)
        thread.start()
    
    def clean_message_for_tts(self, message: str, agent_name: str) -> str:
        """Clean a log message for TTS output"""
        # Remove emoji and special characters
        clean_msg = re.sub(r'[^\w\s\-:.,!?]', '', message)
        
        # Extract relevant parts for different message types
        if "offre" in clean_msg.lower() and agent_name in clean_msg:
            # For bid messages like "💰 Angelo offre 50 crediti"
            match = re.search(rf'{re.escape(agent_name)}\s+offre\s+(\d+)\s+crediti', clean_msg)
            if match:
                amount = match.group(1)
                return f"{agent_name} offre {amount} crediti"
        
        elif "non offre" in clean_msg.lower() and agent_name in clean_msg:
            # For no bid messages like "🚫 Angelo non offre"
            return f"{agent_name} non offre"
        
        elif "assegnato a" in clean_msg.lower() and agent_name in clean_msg:
            # For assignment messages
            match = re.search(r'(\w+)\s+assegnato\s+a\s+' + re.escape(agent_name) + r'\s+per\s+(\d+)\s+crediti', clean_msg)
            if match:
                player_name = match.group(1)
                amount = match.group(2)
                return f"{player_name} assegnato a {agent_name} per {amount} crediti"
        
        # If no specific pattern matches, return cleaned general message if it contains agent name
        if agent_name.lower() in clean_msg.lower():
            # Limit length for TTS
            clean_msg = clean_msg[:100]
            return clean_msg
        
        return ""
    
    def speak_for_agent(self, agent_name: str, message: str, agents_config: List[Dict]):
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
            self.speak_async(clean_message)
    
    def announce_player(self, player_name: str, role: str, evaluation: int):
        """Announce player in auction"""
        if self.available:
            message = f"In asta: {player_name}, {role}, valutazione {evaluation}"
            self.speak_async(message)
    
    def announce_winner(self, player_name: str, winner: str, amount: int):
        """Announce auction winner"""
        if self.available:
            message = f"{player_name} assegnato a {winner} per {amount} crediti"
            self.speak_async(message)
