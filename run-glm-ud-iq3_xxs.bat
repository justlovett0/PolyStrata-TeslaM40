@echo off
title PolyStrata GLM-5.3-Flash-UD-IQ3_XXS
cd /d "D:\polystrata-maxwell\PolyStrata-0.1.44.1"
"D:\polystrata-maxwell\PolyStrata-0.1.44.1\.venv\Scripts\python.exe" "D:\polystrata-maxwell\PolyStrata-0.1.44.1\serve\server.py" "--engine" "strata" "--config" "D:\polystrata-maxwell\PolyStrata-0.1.44.1\strata-glm-ud-iq3_xxs.json" "--port" "8081" "--open"
if errorlevel 1 pause
