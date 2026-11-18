@echo off
setlocal
cd /d "%~dp0"
python -m http.server 8000 --directory public
if errorlevel 1 (
    echo.
    echo Could not start SignSpeak. Install Python or run npm run dev.
    pause
)
