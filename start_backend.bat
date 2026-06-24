@echo off
:: ============================================================
::  AWIS — Start Backend (Windows)
::  Requires Python 3.10+ and packages from requirements.txt
:: ============================================================
title AWIS Backend

echo.
echo  AWIS Backend — FastAPI + Uvicorn
echo  ==================================
echo.

:: Change to the directory containing this script
cd /d "%~dp0"

:: Check Python is available
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo  [ERROR] Python not found. Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

:: Check if virtual environment exists; create if not
if not exist ".venv\Scripts\activate.bat" (
    echo  Creating virtual environment...
    python -m venv .venv
    echo  Installing dependencies...
    .venv\Scripts\pip install -r requirements.txt
)

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Load .env if it exists
if exist ".env" (
    echo  Loading .env ...
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        if not "%%A"=="" if not "%%A:~0,1%"=="#" set "%%A=%%B"
    )
)

:: Default values
if "%PORT%"==""    set PORT=8000
if "%HOST%"==""    set HOST=0.0.0.0
if "%WORKERS%"=="" set WORKERS=1

echo  Starting FastAPI on http://%HOST%:%PORT%
echo  Swagger UI: http://localhost:%PORT%/docs
echo  Press Ctrl+C to stop.
echo.

uvicorn main:app --host %HOST% --port %PORT% --workers %WORKERS% --reload
