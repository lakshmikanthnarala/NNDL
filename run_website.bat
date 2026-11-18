@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
if exist ".venv311\Scripts\python.exe" set "PYTHON=.venv311\Scripts\python.exe"

"%PYTHON%" server.py
if errorlevel 1 (
    echo.
    echo Could not start SignSpeak. Install the dependencies in requirements.txt.
    pause
)
