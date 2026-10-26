@echo off
setlocal
title PolyStrata LFM2.5 24B-A2B UD-Q4_K_XL
cd /d "%~dp0"
set "STRATA_PYTHON=C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not exist "%STRATA_PYTHON%" set "STRATA_PYTHON=%~dp0.venv\Scripts\python.exe"
set "PYTHONPATH=%~dp0.tools\pydeps;%~dp0.venv\Lib\site-packages"
"%STRATA_PYTHON%" "%~dp0serve\server.py" --engine strata --config "%~dp0strata-lfm25-24b-q4_k_xl.json" --port 8087 --open
if errorlevel 1 pause
