@echo off
title CIS-7 Cosmic Intelligence System
color 0A

echo.
echo  ========================================
echo   CIS-7 COSMIC INTELLIGENCE SYSTEM
echo  ========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Install from https://python.org
    pause
    exit
)

:: Move to api folder
cd /d "%~dp0api"

:: Install dependencies
echo  [1/3] Installing dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo  [ERROR] Failed to install requirements.
    pause
    exit
)

:: Check required files exist
echo  [2/3] Checking required files...

if not exist "..\ml\Model\best_svc_Model.pkl" (
    echo.
    echo  [ERROR] Missing: ml\Model\best_svc_Model.pkl
    echo  Copy it from your GitHub repo into ml\Model\
    echo.
    pause
    exit
)

if not exist "..\ml\Model\num_pipeline.pkl" (
    echo.
    echo  [ERROR] Missing: ml\Model\num_pipeline.pkl
    echo  Copy it from your GitHub repo into ml\Model\
    echo.
    pause
    exit
)

if not exist "..\ml\Dataset\thermoracleTrain.csv" (
    echo.
    echo  [ERROR] Missing: ml\Dataset\thermoracleTrain.csv
    echo  Copy it from your GitHub repo into ml\Dataset\
    echo.
    pause
    exit
)

echo  [3/3] All files found. Starting server...
echo.
echo  ----------------------------------------
echo   Open your browser at:
echo   http://localhost:8000
echo  ----------------------------------------
echo.
echo  Press Ctrl+C to stop the server.
echo.

:: Start server
uvicorn main:app --reload --port 8000

pause
