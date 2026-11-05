@echo off
rem Twin Depot launcher for Windows: double-click for a menu, or "twin demo", "twin real", "twin ports", ...
setlocal
set "PY="
if exist "%~dp0.twin-python" set /p PY=<"%~dp0.twin-python"
if defined PY if not exist "%PY%" set "PY="
rem no environment yet: any Python works for "twin setup" (conda base, the py launcher, python on PATH)
for %%D in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniforge3" "%PROGRAMDATA%\miniconda3" "%PROGRAMDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3") do (
    if not defined PY if exist "%%~D\python.exe" set "PY=%%~D\python.exe"
)
if not defined PY where py >nul 2>&1 && set "PY=py"
if not defined PY where python >nul 2>&1 && set "PY=python"
if not defined PY (
    echo [twin] Python not found. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)
set PYTHONUTF8=1
"%PY%" "%~dp0twin.py" %*
