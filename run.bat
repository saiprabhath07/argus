@echo off
REM ARGUS - Run script for Windows
echo 🛰️  ARGUS - Aerial Geospatial Reconstruction
echo ==========================================

REM Check if venv exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Installing dependencies...
.venv\Scripts\python -m pip install --upgrade pip -q
.venv\Scripts\pip install -r requirements.txt -q

echo ✅ Dependencies installed
echo 🚀 Starting Streamlit app...
echo    Open http://localhost:8501 in your browser
echo.

.venv\Scripts\python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
pause
