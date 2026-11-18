@echo off
setlocal
cd /d "%~dp0"
set "KERAS_HOME=%~dp0.keras"
set "MPLCONFIGDIR=%~dp0.matplotlib"
set "TF_ENABLE_ONEDNN_OPTS=0"

if exist ".venv311\Scripts\python.exe" (
    ".venv311\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
) else (
    python -m streamlit run app.py
)

if errorlevel 1 (
    echo.
    echo Could not start SignSpeak. Install the project dependencies first.
    pause
)
