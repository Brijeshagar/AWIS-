@echo off
:: ============================================================
::  AWIS — Start Frontend (Windows)
::  Requires Node.js 18+ and npm
:: ============================================================
title AWIS Frontend

echo.
echo  AWIS Frontend — React + Vite Dev Server
echo  =========================================
echo.

:: Change to the frontend directory
cd /d "%~dp0frontend"

:: Check Node is available
where node >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo  [ERROR] Node.js not found. Please install Node.js 18+ from https://nodejs.org
    pause
    exit /b 1
)

:: Install dependencies if node_modules is missing
if not exist "node_modules" (
    echo  Installing npm dependencies...
    npm install
)

echo  Starting Vite dev server...
echo  Open: http://localhost:5173
echo  Press Ctrl+C to stop.
echo.

npm run dev
