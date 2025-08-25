@echo off
echo ============================================
echo    🚀 FANTABOT - Avvio Backend e Frontend
echo ============================================
echo.

REM Colori per output
for /F %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"
set "GREEN=%ESC%[32m"
set "BLUE=%ESC%[34m"
set "YELLOW=%ESC%[33m"
set "RED=%ESC%[31m"
set "RESET=%ESC%[0m"

echo %GREEN%Controllo prerequisiti...%RESET%
echo.

REM Controlla se Python è installato
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%❌ Errore: Python non è installato o non è nel PATH%RESET%
    echo %YELLOW%   Installa Python da https://python.org%RESET%
    pause
    exit /b 1
)

REM Controlla se Node.js è installato
node --version >nul 2>&1
if errorlevel 1 (
    echo %RED%❌ Errore: Node.js non è installato o non è nel PATH%RESET%
    echo %YELLOW%   Installa Node.js da https://nodejs.org%RESET%
    pause
    exit /b 1
)

echo %GREEN%✅ Python e Node.js trovati%RESET%
echo.

REM Controlla se il virtual environment esiste
if not exist ".venv\" (
    echo %YELLOW%⚠️  Virtual environment non trovato, creazione in corso...%RESET%
    python -m venv .venv
    if errorlevel 1 (
        echo %RED%❌ Errore nella creazione del virtual environment%RESET%
        pause
        exit /b 1
    )
    echo %GREEN%✅ Virtual environment creato%RESET%
)

REM Attiva il virtual environment e installa dipendenze Python
echo %BLUE%📦 Installazione dipendenze Python...%RESET%
call .venv\Scripts\activate.bat
pip install -r requirements_web.txt
if errorlevel 1 (
    echo %RED%❌ Errore nell'installazione delle dipendenze Python%RESET%
    pause
    exit /b 1
)

REM Controlla se node_modules esiste nel frontend
if not exist "frontend\node_modules\" (
    echo %BLUE%📦 Installazione dipendenze Frontend...%RESET%
    cd frontend
    npm install
    if errorlevel 1 (
        echo %RED%❌ Errore nell'installazione delle dipendenze del frontend%RESET%
        pause
        exit /b 1
    )
    cd ..
    echo %GREEN%✅ Dipendenze frontend installate%RESET%
)

echo.
echo %GREEN%🎯 Avvio servizi in parallelo...%RESET%
echo.
echo %YELLOW%📍 Backend sarà disponibile su: http://localhost:8080%RESET%
echo %YELLOW%📍 Frontend sarà disponibile su: http://localhost:5173%RESET%
echo.
echo %BLUE%💡 Premi Ctrl+C per fermare entrambi i servizi%RESET%
echo.

REM Avvia il backend in una nuova finestra
start "🐍 FantaBot Backend" cmd /k "call .venv\Scripts\activate.bat && python app.py"

REM Aspetta un secondo per dare tempo al backend di avviarsi
timeout /t 2 /nobreak >nul

REM Avvia il frontend in una nuova finestra
start "⚛️ FantaBot Frontend" cmd /k "cd frontend && npm run dev"

echo %GREEN%✅ Servizi avviati!%RESET%
echo.
echo %YELLOW%📝 Note:%RESET%
echo   - Due nuove finestre del terminale si sono aperte
echo   - Backend: Finestra "🐍 FantaBot Backend"
echo   - Frontend: Finestra "⚛️ FantaBot Frontend"
echo   - Chiudi le finestre per fermare i servizi
echo.
echo %BLUE%🌐 Aprendo il browser...%RESET%

REM Aspetta che i servizi si avviino
timeout /t 5 /nobreak >nul

REM Apri il browser al frontend
start http://localhost:5173

echo.
echo %GREEN%🎉 FantaBot è ora in esecuzione!%RESET%
echo.
pause
