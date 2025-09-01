# 🏆 FantaBot Pro - Sistema Avanzato per Aste Fantacalcio

## 📋 Panoramica

FantaBot Pro è un sistema completo e avanzato per gestire aste di fantacalcio con agenti intelligenti che simulano comportamenti realistici di partecipanti umani. Il sistema include un'interfaccia grafica moderna, agenti AI sofisticati e strumenti di analisi avanzati.

## ✨ Caratteristiche Principali

### 🤖 **Agenti Intelligenti**
- **Human Agent**: Controllo manuale tramite interfaccia grafica
- **Cap Agent**: Strategia basata su valutazioni e budget dinamici
- **Enhanced Cap Agent**: Versione avanzata con adattamento in tempo reale
- **Dynamic Cap Agent**: Agente che si adatta alle condizioni di mercato
- **Random Agent**: Comportamento casuale per test
- **RL Deep Agent**: Agente con intelligenza artificiale (reinforcement learning)

### 🎯 **Interfaccia Utente Avanzata**
- **Tab Configurazione**: Setup completo con presets e validazione
- **Tab Asta Live**: Visualizzazione in tempo reale con statistiche
- **Tab Risultati**: Analisi dettagliata post-asta
- **Tab Logs**: Sistema di logging completo
- **Sistema TTS**: Annunci vocali personalizzabili

### 📊 **Funzionalità Avanzate**
- Salvataggio/caricamento configurazioni
- Esportazione risultati in Excel
- Validazione input in tempo reale
- Sistema di backup automatico
- Analisi performance agenti
- Statistiche mercato in tempo reale

## 🚀 Installazione e Setup

### Prerequisiti
```bash
# Python 3.8 o superiore
python --version

# Dipendenze richieste
pip install -r requirements_gui.txt
```

### Dipendenze Principali
```
tkinter (incluso in Python)
pandas>=1.3.0
openpyxl>=3.0.7
pyttsx3>=2.90
```

### Setup Rapido
1. **Clona il repository**
   ```bash
   git clone <repository-url>
   cd fantabot
   ```

2. **Installa dipendenze**
   ```bash
   pip install -r requirements_gui.txt
   ```

3. **Prepara i dati giocatori**
   - Posiziona il file Excel con i giocatori in `data/players_list.xlsx`
   - Formato richiesto: `name`, `team`, `role`, `evaluation`

4. **Avvia l'applicazione**
   ```bash
   # Versione avanzata (consigliata)
   python enhanced_auction_gui.py
   
   # Versione originale
   python auction_gui.py
   ```

## 📖 Guida all'Uso

### 1. **Configurazione dell'Asta**
- **Impostazioni Base**: Definisci crediti iniziali e formazione (GK, DEF, MID, ATT)
- **Agenti Partecipanti**: Aggiungi giocatori umani e bot con strategie diverse
- **Tipo Asta**: Scegli tra "chiamata" e "classica"
- **Salva/Carica**: Mantieni configurazioni per aste future

### 2. **Gestione Asta Live**
- **Monitoraggio Real-time**: Visualizza giocatore in asta, offerte e statistiche
- **Input Umano**: Sistema intuitivo per offerte manuali
- **TTS Personalizzato**: Annunci vocali per ogni agente
- **Controlli Asta**: Pausa, riprendi, ferma quando necessario

### 3. **Analisi Risultati**
- **Esportazione Excel**: Report dettagliati per squadra
- **Statistiche Performance**: Analisi efficacia strategie
- **Comparazione Agenti**: Confronto risultati tra diversi bot
- **Log Completi**: Storia dettagliata di ogni offerta

## 🔧 Struttura del Progetto

```
fantabot/
├── 📁 src/
│   ├── 📁 core/
│   │   └── enhanced_auction.py      # Motore asta migliorato
│   ├── 📁 agents/
│   │   ├── agent_class.py           # Classe base agenti
│   │   ├── cap_based_agent.py       # Agente strategico
│   │   ├── enhanced_cap_agent.py    # Agente avanzato
│   │   └── human_agent.py           # Agente umano
│   ├── 📁 gui/
│   │   ├── config_tab.py            # Tab configurazione
│   │   └── enhanced_auction_tab.py  # Tab asta migliorato
│   ├── 📁 utils/
│   │   ├── logging_handler.py       # Sistema logging
│   │   ├── tts_manager.py          # Gestione TTS
│   │   ├── validators.py           # Validazioni
│   │   └── file_manager.py         # Gestione file
│   ├── models.py                   # Modelli dati
│   ├── auction.py                  # Sistema asta originale
│   └── data_loader.py              # Caricamento dati
├── 📁 data/
│   └── players_list.xlsx           # Database giocatori
├── 📁 logs/                        # File di log e backup
├── enhanced_auction_gui.py         # GUI principale migliorata
├── auction_gui.py                  # GUI originale
└── requirements_gui.txt            # Dipendenze
```

## 🎮 Strategie degli Agenti

### **Cap Agent Classico**
- Calcola un "cap" (prezzo massimo) per ogni giocatore
- Basato su valutazione e budget disponibile
- Strategia conservativa e prevedibile

### **Enhanced Cap Agent** ⭐
- **Strategia Adattiva**: Si adatta alle condizioni di mercato
- **Priorità Ruoli**: Investe di più su centrocampo e attacco
- **Target Players**: Identifica giocatori chiave da acquisire
- **Inflazione Mercato**: Monitora e si adatta ai prezzi di mercato
- **Gestione Budget**: Ottimizza spesa in base a crediti rimanenti

### **Dynamic Cap Agent**
- Modifica le strategie durante l'asta
- Reagisce al comportamento degli altri partecipanti
- Equilibrio tra aggressività e conservazione

## 🔍 Funzionalità Avanzate

### **Sistema di Validazione**
- Controllo configurazioni prima dell'avvio
- Validazione offerte in tempo reale
- Prevenzione errori di input

### **TTS Intelligente**
- Annunci personalizzati per agente
- Filtri per ridurre rumore
- Controllo volume e velocità

### **Analisi Performance**
- Tracking efficacia strategie
- Confronto risultati tra aste
- Identificazione pattern vincenti

### **Gestione File Avanzata**
- Backup automatici
- Esportazione multi-formato
- Import/export configurazioni

## 📈 Consigli per l'Uso Ottimale

### **Per Aste Casalinghe (4-6 partecipanti)**
```python
# Configurazione consigliata
Crediti: 1000€
Agenti: 2-3 Umani + 1-2 Enhanced Cap Agent
TTS: Abilitato per tutti
Strategia: "adaptive" con aggression_level: 0.5
```

### **Per Aste Grandi (8+ partecipanti)**
```python
# Configurazione consigliata  
Crediti: 1200€
Agenti: Mix di tutti i tipi
TTS: Solo per agenti chiave
Strategia: "bestxi_focused" con aggression_level: 0.7
```

### **Per Test e Simulazioni**
```python
# Configurazione test
Crediti: 500€
Agenti: 1 Umano + 1 Enhanced Cap + 1 Random
TTS: Disabilitato
Strategia: "value_hunting"
```

## 🛠️ Personalizzazione

### **Modificare Strategie Agenti**
```python
# In enhanced_cap_agent.py
agent = EnhancedCapAgent(
    agent_id="Il Mio Bot",
    cap_strategy="adaptive",     # adaptive, bestxi_focused, value_hunting
    aggression_level=0.8,        # 0.0-1.0 (conservativo-aggressivo)
    adaptability=True            # Abilita adattamento real-time
)
```

### **Personalizzare Priorità Ruoli**
```python
# Modifica position_priorities in EnhancedCapAgent
self.position_priorities = {
    "GK": 0.8,    # Bassa priorità portieri
    "DEF": 1.0,   # Normale per difensori  
    "MID": 1.3,   # Alta priorità centrocampo
    "ATT": 1.4    # Massima priorità attaccanti
}
```

## 🐛 Risoluzione Problemi

### **TTS Non Funziona**
```bash
# Installa/reinstalla pyttsx3
pip uninstall pyttsx3
pip install pyttsx3==2.90

# Su macOS potrebbe servire:
brew install espeak
```

### **Errori Excel**
```bash
# Installa openpyxl
pip install openpyxl==3.0.10

# Verifica formato file Excel (deve essere .xlsx)
```

### **Performance Lente**
- Riduci numero agenti attivi
- Disabilita TTS se non necessario
- Chiudi altre applicazioni pesanti

## 📞 Supporto e Contributi

### **Segnalazione Bug**
- Crea issue su GitHub con log dettagliati
- Includi configurazione utilizzata
- Specifica sistema operativo

### **Richieste Funzionalità**
- Proponi miglioramenti via issue
- Fornisci casi d'uso specifici
- Considera implementazione custom

### **Contributi Sviluppo**
- Fork del repository
- Branch per ogni feature
- Test approfonditi prima del merge
- Documentazione aggiornata

---

## 🏆 **Buona fortuna con la tua asta fantacalcio!**

*FantaBot Pro - Porta la tua asta al livello successivo! 🚀*
