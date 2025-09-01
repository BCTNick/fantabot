#!/bin/bash

# Script per avviare FantaBot

echo "🚀 Avvio FantaBot..."

# Controlla se siamo nella directory corretta
if [ ! -f "api_server.py" ]; then
    echo "❌ Errore: Esegui questo script dalla directory root del progetto"
    exit 1
fi

echo "📋 Controllo dipendenze Python..."

# Controlla se Python è installato
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 non trovato. Installa Python 3 per continuare."
    exit 1
fi

# Controlla se le dipendenze Python sono installate
if [ ! -d "src" ]; then
    echo "❌ Directory src non trovata"
    exit 1
fi

echo "🌐 Avvio server API in background..."

# Avvia il server API in background
python3 api_server.py &
API_PID=$!

echo "🔧 Server API avviato con PID: $API_PID"
echo "📡 API disponibile su: http://localhost:8081/api"
echo "🎯 Frontend disponibile su: http://localhost:8081"

# Attendi alcuni secondi per l'avvio del server
sleep 3

echo "✅ FantaBot è ora in esecuzione!"
echo ""
echo "📖 Come usare:"
echo "   1. Apri il browser su http://localhost:8081"
echo "   2. Crea una nuova asta"
echo "   3. Configura i partecipanti"
echo "   4. Inizia l'asta!"
echo ""
echo "⏹️  Per fermare il server, premi Ctrl+C"

# Mantieni lo script in esecuzione
wait $API_PID
