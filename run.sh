#!/bin/bash
echo "🌾 Agricultural Crop Classification System"
echo "=========================================="
echo

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install requirements if needed
if [ ! -d ".venv/lib/python*/site-packages/gradio" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the app
echo "Starting the app..."
echo "Open http://localhost:7860 in your browser"
echo
python app.py
