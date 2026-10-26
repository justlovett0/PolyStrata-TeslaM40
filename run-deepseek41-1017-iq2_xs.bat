@echo off
setlocal
title PolyStrata DeepSeek V4.1 Flash 1017 IQ2_XS
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\prepare_deepseek41_launch.ps1" -ConfigPath "%~dp0strata-deepseek41-1017-iq2_xs.json"
if errorlevel 1 (pause & exit /b 1)
set "STRATA_PYTHON=C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
set "PYTHONPATH=%~dp0.tools\pydeps;%~dp0.venv\Lib\site-packages"
"%STRATA_PYTHON%" "%~dp0serve\server.py" --engine strata --config "%~dp0strata-deepseek41-1017-iq2_xs.json" --port 8088 --open
if errorlevel 1 pause
