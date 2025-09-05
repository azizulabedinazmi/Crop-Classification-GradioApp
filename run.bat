@echo off
echo 🌾 Agricultural Crop Classification System
echo ==========================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate

REM Install requirements if needed
if not exist ".venv\Lib\site-packages\gradio" (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Run the app
echo Starting the app...
echo Open http://localhost:7860 in your browser
echo.
python app.py

pause
