
@echo off
title SignBridge Launcher

cd /d "%~dp0"

call "%~dp0venv\Scripts\activate.bat"

cd /d "%~dp0backend"

start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:5000/"

python app.py

pause
