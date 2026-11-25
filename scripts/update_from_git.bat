@echo off
chcp 65001 >nul
cd /d "%~dp0\.."

echo [1/3] Stopping running VisionDine Python process...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 /nobreak >nul

echo [2/3] Pulling latest code from GitHub...
git pull origin main

echo [3/3] Checking Python dependencies and starting 24/7 server...
python -m pip install -r requirements.txt --quiet
start /min "" "%~dp0\..\start_24_7.bat"

echo Done! Server updated and restarted in background.
pause
