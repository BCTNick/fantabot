#!/bin/bash

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}    🚀 FANTABOT - Avvio Backend e Frontend${NC}"
echo -e "${CYAN}============================================${NC}"
echo

echo -e "${GREEN}Controllo prerequisiti...${NC}"
echo

# Controlla se Python è installato
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Errore: Python non è installato o non è nel PATH${NC}"
    echo -e "${YELLOW}   Installa Python con: brew install python (macOS) o apt install python3 (Ubuntu)${NC}"
    exit 1
fi

# Usa python3 se disponibile, altrimenti python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
else
    PYTHON_CMD="python"
    PIP_CMD="pip"
fi

# Controlla se Node.js è installato
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Errore: Node.js non è installato o non è nel PATH${NC}"
    echo -e "${YELLOW}   Installa Node.js da https://nodejs.org o con: brew install node (macOS)${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python ($PYTHON_CMD) e Node.js trovati${NC}"
echo

# Controlla se il virtual environment esiste
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment non trovato, creazione in corso...${NC}"
    $PYTHON_CMD -m venv .venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Errore nella creazione del virtual environment${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Virtual environment creato${NC}"
fi

# Attiva il virtual environment e installa dipendenze Python
echo -e "${BLUE}📦 Installazione dipendenze Python...${NC}"
source .venv/bin/activate

# Controlla se requirements_web.txt esiste
if [ -f "requirements_web.txt" ]; then
    $PIP_CMD install -r requirements_web.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Errore nell'installazione delle dipendenze Python${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠️  File requirements_web.txt non trovato, skip installazione dipendenze Python${NC}"
fi

# Controlla se node_modules esiste nel frontend
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${BLUE}📦 Installazione dipendenze Frontend...${NC}"
    cd frontend
    npm install
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Errore nell'installazione delle dipendenze del frontend${NC}"
        exit 1
    fi
    cd ..
    echo -e "${GREEN}✅ Dipendenze frontend installate${NC}"
fi

echo
echo -e "${GREEN}🎯 Avvio servizi in parallelo...${NC}"
echo
echo -e "${YELLOW}📍 Backend sarà disponibile su: http://localhost:8080${NC}"
echo -e "${YELLOW}📍 Frontend sarà disponibile su: http://localhost:5173${NC}"
echo
echo -e "${BLUE}💡 Premi Ctrl+C per fermare entrambi i servizi${NC}"
echo

# Funzione per cleanup quando si esce
cleanup() {
    echo
    echo -e "${YELLOW}🛑 Fermando i servizi...${NC}"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}✅ Servizi fermati${NC}"
    exit 0
}

# Intercetta Ctrl+C
trap cleanup SIGINT

# Avvia il backend in background
echo -e "${PURPLE}🐍 Avvio Backend...${NC}"
source .venv/bin/activate && $PYTHON_CMD app.py &
BACKEND_PID=$!

# Aspetta un momento per dare tempo al backend di avviarsi
sleep 3

# Controlla se il backend è ancora in esecuzione
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}❌ Errore: Il backend non si è avviato correttamente${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Backend avviato (PID: $BACKEND_PID)${NC}"

# Avvia il frontend in background
echo -e "${PURPLE}⚛️  Avvio Frontend...${NC}"
cd frontend && npm run dev &
FRONTEND_PID=$!
cd ..

# Aspetta un momento per dare tempo al frontend di avviarsi
sleep 5

# Controlla se il frontend è ancora in esecuzione
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${RED}❌ Errore: Il frontend non si è avviato correttamente${NC}"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo -e "${GREEN}✅ Frontend avviato (PID: $FRONTEND_PID)${NC}"
echo

echo -e "${GREEN}🎉 FantaBot è ora in esecuzione!${NC}"
echo
echo -e "${YELLOW}📝 Note:${NC}"
echo -e "   - Backend PID: $BACKEND_PID"
echo -e "   - Frontend PID: $FRONTEND_PID"
echo -e "   - Premi Ctrl+C per fermare entrambi i servizi"
echo

# Determina il comando per aprire il browser
if command -v open &> /dev/null; then
    # macOS
    echo -e "${BLUE}🌐 Aprendo il browser su macOS...${NC}"
    sleep 2
    open http://localhost:5173
elif command -v xdg-open &> /dev/null; then
    # Linux
    echo -e "${BLUE}🌐 Aprendo il browser su Linux...${NC}"
    sleep 2
    xdg-open http://localhost:5173
else
    echo -e "${YELLOW}🌐 Apri manualmente il browser su: http://localhost:5173${NC}"
fi

echo
echo -e "${CYAN}🔄 Servizi in esecuzione... Premi Ctrl+C per fermare${NC}"

# Aspetta che i processi finiscano o che l'utente prema Ctrl+C
wait $BACKEND_PID $FRONTEND_PID
