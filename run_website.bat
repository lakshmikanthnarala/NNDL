@echo off
setlocal
cd /d "%~dp0"
if not exist "node_modules" (
    echo Install Node dependencies first with: npm install
    pause
    exit /b 1
)
npm run dev
