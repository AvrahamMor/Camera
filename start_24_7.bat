@echo off
chcp 65001 >nul
title VisionDine AI - 24/7 Restaurant Server
cd /d "%~dp0"

if not exist "logs" mkdir logs

echo ================================================================================
echo   VisionDine AI - 24/7 Watchdog Server Mode
echo   Server will start automatically and restart after 5s if it ever exits.
echo ================================================================================

:loop
echo [%date% %time%] Starting VisionDine AI server... >> logs\server.log
echo [%date% %time%] Starting VisionDine AI server...
python run.py --no-browser >> logs\server.log 2>&1

echo [%date% %time%] Server exited with code %errorlevel%. Restarting in 5 seconds... >> logs\server.log
echo [%date% %time%] Server stopped. Restarting in 5 seconds... (Press Ctrl+C to stop)
timeout /t 5 /nobreak >nul
goto loop
