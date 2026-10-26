@echo off
setlocal
title PolyStrata Qwen3.5 122B-A10B Q4_K_M
cd /d "%~dp0"
set "STRATA_PYTHON=C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not exist "%STRATA_PYTHON%" set "STRATA_PYTHON=%~dp0.venv\Scripts\python.exe"
set "PYTHONPATH=%~dp0.tools\pydeps;%~dp0.venv\Lib\site-packages"
"%STRATA_PYTHON%" "%~dp0serve\server.py" --engine strata --config "%~dp0strata-qwen35-122b-q4_k_m.json" --port 8086 --open
if errorlevel 1 pause
