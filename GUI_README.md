# FantaBot - GUI per Asta di Fantacalcio

Una interfaccia grafica moderna per gestire aste di fantacalcio simulate con agenti intelligenti.

## 🚀 Avvio Rapido

```bash
# Installa le dipendenze
pip install -r requirements_gui.txt

# Avvia la GUI
python run_gui.py
```

## 📋 Funzionalità

### Tab Configurazione
- **Impostazioni Asta**: Configura crediti iniziali, numero di slot per ruolo, tipo di asta
- **Gestione Agenti**: Aggiungi/rimuovi agenti partecipanti con diversi tipi:
  - **Human Agent**: Agenti umani (per test o partecipazione manuale)
  - **Random Agent**: Agenti che fanno offerte casuali
  - **Cap Agent**: Agenti basati su strategia di budget
  - **Dynamic Cap Agent**: Agenti con strategia dinamica
  - **RL Deep Agent**: Agenti con intelligenza artificiale

### Tab Asta
- **Controlli**: Avvia/ferma l'asta
- **Layout Diviso**: 
  - **Sinistra (60%)**: Informazioni asta e log in tempo reale
  - **Destra (40%)**: Pannello agenti umani (attivo solo quando necessario)
- **Interfaccia Dinamica**: Il pannello destro si attiva automaticamente quando gli agenti umani devono fare offerte
- **Stato**: Visualizzazione dello stato corrente dell'asta

### Tab Log
- **Visualizzazione**: Log dettagliati di tutte le operazioni dell'asta
- **Gestione**: Pulisci log, salva log su file
- **Tempo reale**: Aggiornamento automatico durante l'asta

## 🎮 Come Usare

### 1. Configurazione
1. Apri il tab "Configurazione"
2. Imposta i parametri dell'asta:
   - Crediti iniziali (default: 1000)
   - Numero di slot per ruolo (GK: 3, DEF: 8, MID: 8, ATT: 6)
   - Tipo asta: "chiamata" o "classica"
   - Modalità "per ruolo"
3. Configura gli agenti:
   - Inserisci nome e tipo di agente
   - Clicca "Aggiungi Agente"
   - Rimuovi agenti selezionandoli e cliccando "Rimuovi Agente Selezionato"

### 2. Avvio Asta
1. Passa al tab "Asta"
2. Clicca "Avvia Asta"
3. **Interfaccia Dinamica**:
   - Lato sinistro: mostra sempre le informazioni dell'asta in corso
   - Lato destro: di default mostra un messaggio informativo
4. **Partecipazione Agenti Umani**:
   - Per ogni giocatore, gli agenti automatici faranno le loro offerte prima
   - Il pannello destro si attiva automaticamente quando è il turno degli agenti umani
   - Appare "Chi vuole offrire?" con tutte le informazioni necessarie
   - Seleziona l'agente e imposta l'offerta direttamente nel pannello
   - Nessun limite di tempo - prendete tutto il tempo necessario per decidere
   - Dopo la decisione, il pannello si nasconde automaticamente
5. Monitora tutto il progresso in tempo reale
6. Usa "Ferma Asta" se necessario

### 3. Monitoraggio
1. Passa al tab "Log" per vedere tutti i dettagli
2. Usa "Salva Log" per esportare i risultati
3. Usa "Pulisci Log" per pulire la visualizzazione

## 🤖 Tipi di Agenti

### Human Agent ⭐ NOVITÀ
- **Interfaccia Integrata**: Gli agenti umani partecipano tramite un **pannello integrato** nella GUI principale
- **Attivazione Automatica**: Il pannello "Chi vuole offrire?" appare automaticamente quando necessario
- **Layout Intelligente**: 
  - Informazioni giocatore corrente
  - Stato dell'asta attuale
  - Selezione agente con controlli offerta integrati
  - Nessun limite di tempo - decidete con calma!
- **Esperienza Fluida**: Nessuna finestra popup, tutto integrato nell'interfaccia principale
- **Controlli Avanzati**: Validazione automatica e limiti dinamici per ogni agente

### Random Agent
- Fa offerte casuali
- Buono per test di base
- Comportamento imprevedibile

### Cap Agent
- Strategia basata su budget fisso
- Gestisce le offerte in base ai crediti disponibili
- Comportamento più realistico

### Dynamic Cap Agent
- Strategia adattiva
- Modifica il comportamento in base alla situazione dell'asta
- Più sofisticato del Cap Agent base

### RL Deep Agent
- Intelligenza artificiale con reinforcement learning
- Impara dalle aste precedenti
- Comportamento più avanzato e strategico

## 🔧 Configurazione Avanzata

### Personalizzazione Slot
Modifica i valori predefiniti per adattarsi alle tue regole di lega:
- **Portieri (GK)**: Numero di portieri per squadra
- **Difensori (DEF)**: Numero di difensori per squadra  
- **Centrocampisti (MID)**: Numero di centrocampisti per squadra
- **Attaccanti (ATT)**: Numero di attaccanti per squadra

### Tipi di Asta
- **Chiamata**: Asta tradizionale con chiamata di giocatori
- **Classica**: Asta con ordine predefinito

### Modalità Per Ruolo
- **Attiva**: L'asta procede ruolo per ruolo
- **Disattiva**: L'asta mescola tutti i giocatori

## 📁 File Generati

La GUI genera automaticamente:
- **Log dell'asta**: File di log dettagliati in `logs/`
- **Export log**: File salvati manualmente dall'utente

## 🔍 Risoluzione Problemi

### Errore "Servono almeno 2 agenti"
- Aggiungi almeno 2 agenti nella configurazione

### Errore di importazione
- Verifica che tutte le dipendenze siano installate: `pip install -r requirements_gui.txt`

### GUI non si avvia
- Controlla che Python supporti tkinter: `python -c "import tkinter"`
- Su Linux potrebbe servire: `sudo apt-get install python3-tk`

### Asta non parte
- Verifica che il file `data/players_list.xlsx` esista
- Controlla che gli agenti siano configurati correttamente

## 🛠️ Sviluppo

Per modificare o estendere la GUI:

```python
# File principale della GUI
auction_gui.py

# Script di avvio
run_gui.py

# Dipendenze
requirements_gui.txt
```

### Aggiungere Nuovi Tipi di Agenti
1. Crea la classe agente in `src/agents/`
2. Importa nel file `auction_gui.py`
3. Aggiungi al dizionario `agent_types`

### Personalizzare l'Interfaccia
La GUI usa tkinter con ttk per un aspetto moderno. Modifica `setup_ui()` e i metodi correlati per personalizzare l'interfaccia.

## 📊 Dati e Configurazione

- **Giocatori**: Caricati da `data/players_list.xlsx`
- **Log**: Salvati in `logs/` con timestamp automatico
- **Configurazione**: Salvata nella sessione corrente (non persistente)

## 🎯 Prossimi Sviluppi

- [ ] Salvataggio/caricamento configurazioni
- [ ] Grafici in tempo reale delle statistiche
- [ ] Export risultati in Excel
- [ ] Modalità multiplayer con interfaccia web
- [ ] Analisi post-asta automatica
- [ ] Configurazione agenti avanzata
- [ ] Temi personalizzabili per l'interfaccia
- [ ] Cronologia offerte per agente umano
- [ ] Chat tra agenti umani durante l'asta
- [ ] Notifiche audio per turni agenti umani

## 🎮 Funzionalità Avanzate Agenti Umani

### Pannello Integrato Dinamico
- **Attivazione Automatica**: Il pannello destro si attiva solo quando gli agenti umani devono decidere
- **Layout Ottimizzato**: 60% per informazioni asta, 40% per controlli agenti umani
- **Integrazione Perfetta**: Nessuna finestra popup che distrae dall'esperienza
- **Visibilità Completa**: Tutto il contesto visibile contemporaneamente

### Controlli Intelligenti
- **Selezione Agente**: Radio button per scegliere quale agente vuole offrire
- **Controllo Offerta**: Spinbox con limiti automatici per ogni agente
- **Validazione Istantanea**: Solo offerte valide sono permesse
- **Nessuna Fretta**: Decidete con calma, senza pressione temporale

### Flusso Ottimizzato
1. **Agenti automatici** fanno offerte → Lato destro mostra placeholder
2. **Attivazione pannello** → Lato destro mostra controlli agenti umani  
3. **Decisione agenti** → Selezione e conferma nel pannello integrato
4. **Risultato** → Il pannello si nasconde e torna al placeholder
5. **Giocatore successivo** → Ripetizione del ciclo

### Esperienza Utente
- **Niente Popup**: Tutto avviene nell'interfaccia principale
- **Controllo Completo**: Vedi asta e controlli contemporaneamente
- **Senza Stress**: Nessun timer che mette pressione
- **Coordinamento**: Agenti umani possono discutere con calma guardando la stessa schermata
