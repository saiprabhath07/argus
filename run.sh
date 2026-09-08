#!/bin/bash
# ARGUS - Run script for Linux/Mac
set -e

echo "🛰️  ARGUS - Aerial Geospatial Reconstruction"
echo "=========================================="

# Check if venv exists, create if not
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate and install deps
echo "Installing dependencies..."
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt -q

echo "✅ Dependencies installed"
echo "🚀 Starting Streamlit app..."
echo "   Open http://localhost:8501 in your browser"
echo ""

.venv/bin/python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
