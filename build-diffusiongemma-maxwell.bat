@echo off
setlocal
call "C:\Program1\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.39 >nul || exit /b 1
set "SRC=D:\polystrata-maxwell\llama-latest"
set "BLD=D:\polystrata-maxwell\llama-latest\build-diffusion-sm52"
set "CMAKE=%~dp0.venv\Lib\site-packages\cmake\data\bin\cmake.exe"
set "NINJA=%~dp0.venv\Scripts\ninja.exe"
set "PATCHED_SERVER=%~dp0tools\diffusiongemma\diffusion-gemma-visual-server.cpp"
set "SERVER_SOURCE=%SRC%\examples\diffusion-gemma-server\diffusion-gemma-visual-server.cpp"
set "OUT=%~dp0engine-diffusiongemma-maxwell"
if not exist "%PATCHED_SERVER%" (echo Missing PolyStrata protocol source: %PATCHED_SERVER%& exit /b 1)
copy /y "%PATCHED_SERVER%" "%SERVER_SOURCE%" >nul || exit /b 1
"%CMAKE%" -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" -S "%SRC%" -B "%BLD%" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=OFF -DGGML_CUDA_FA=OFF -DGGML_CUDA_NO_VMM=ON -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=ON -DCMAKE_CUDA_ARCHITECTURES=52 "-DCMAKE_C_COMPILER=C:\Program1\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64\cl.exe" "-DCMAKE_CXX_COMPILER=C:\Program1\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64\cl.exe" "-DCMAKE_LINKER=C:\Program1\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64\link.exe" "-DCMAKE_CUDA_COMPILER=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\nvcc.exe" "-DCUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0" || exit /b 1
"%CMAKE%" --build "%BLD%" --target llama-diffusion-gemma-visual-server -j 12 || exit /b 1
if not exist "%OUT%" mkdir "%OUT%"
copy /y "%BLD%\bin\*.dll" "%OUT%\" >nul || exit /b 1
copy /y "%BLD%\bin\llama-diffusion-gemma-visual-server.exe" "%OUT%\" >nul || exit /b 1
echo DiffusionGemma SM52 engine ready in %OUT%
