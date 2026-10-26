@echo off
setlocal
call "C:\Program1\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.39 >nul || exit /b 1
set "CMAKE=%~dp0.venv\Lib\site-packages\cmake\data\bin\cmake.exe"
"%CMAKE%" --build "%~dp0build-maxwell" --target strata-dsv41 -j 12 || exit /b 1
