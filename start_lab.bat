@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set PORT=8766
set BASE=http://127.0.0.1:%PORT%
set PY=.venv\Scripts\pythonw.exe

if /i "%~1"=="stop" goto :stop_all

rem ============ 1. ensure lab server is running ============
netstat -ano | findstr /r ":%PORT% .*LISTENING" >nul
if errorlevel 1 (
    echo [1/3] Starting fly2048 lab server - background, no window...
    start "fly2048-lab" /min "%PY%" fly2048_lab.py
    call :wait_port %PORT%
    if errorlevel 1 (
        echo [ERROR] Port %PORT% not ready within 60s.
        echo Run manually to see the error: .venv\Scripts\python.exe fly2048_lab.py
        pause
        exit /b 1
    )
    echo       Server started.
) else (
    echo [1/3] Lab server already running.
)

rem ============ 2. training status: ask to start if not running ============
powershell -NoProfile -Command "if ((Invoke-RestMethod -Uri '%BASE%/train/status').running) { exit 0 } else { exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo [2/3] No training is running.
    set /p DO_TRAIN=Start 300-episode training now? [Y/N, Enter=N]:
    if /i "!DO_TRAIN!"=="Y" (
        echo       Starting training - 300 episodes...
        powershell -NoProfile -Command "Invoke-RestMethod -Method Post -Uri '%BASE%/train/start' -ContentType 'application/json' -Body '{\"episodes\":300,\"random_baseline\":false}' | Out-Null"
        echo       Training started in background. Watch progress in the console page.
    )
) else (
    echo [2/3] Training is running. Watch progress in the console page.
)

rem ============ 3. open browser ============
echo [3/3] Opening browser...
start "" "%BASE%/"
start "" "%BASE%/console"
echo.
echo Done. Lab:    %BASE%/
echo        Console: %BASE%/console
exit /b 0

rem ============ wait for port (max 60 s) ============
:wait_port
set /a tries=0
:wait_loop
timeout /t 1 /nobreak >nul
set /a tries+=1
netstat -ano | findstr /r ":%~1 .*LISTENING" >nul
if not errorlevel 1 exit /b 0
if %tries% lss 60 goto :wait_loop
exit /b 1

rem ============ stop everything ============
:stop_all
echo Stopping training and lab server...
powershell -NoProfile -Command "try { Invoke-RestMethod -Method Post -Uri '%BASE%/train/stop' | Out-Null } catch { }" >nul 2>&1
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r ":%PORT% .*LISTENING"') do taskkill /f /pid %%p >nul 2>&1
echo Stopped.
exit /b 0