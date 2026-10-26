@echo off

call "C:\Program1\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.39 >nul

D:\polystrata-maxwell\PolyStrata-0.1.44.1\.venv\Lib\site-packages\cmake\data\bin\cmake.exe -G Ninja -DCMAKE_MAKE_PROGRAM=D:\polystrata-maxwell\PolyStrata-0.1.44.1\.venv\Scripts\ninja.exe -S D:\polystrata-maxwell\PolyStrata-0.1.44.1 -B D:\polystrata-maxwell\PolyStrata-0.1.44.1\build-maxwell -DCMAKE_BUILD_TYPE=Release "-DCMAKE_C_COMPILER=C:\Program1\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64\cl.exe" "-DCMAKE_CXX_COMPILER=C:\Program1\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64\cl.exe" "-DCMAKE_LINKER=C:\Program1\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64\link.exe" -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=52 "-DCMAKE_CUDA_COMPILER=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\nvcc.exe" "-DCUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0" -DSTRATA_GGML_DIR=D:\polystrata-maxwell\PolyStrata-0.1.44.1\third_party\llama.cpp -DSTRATA_EXPERIMENTAL_SM52=ON -DSTRATA_BUILD_DSV41=ON -DCMAKE_CUDA_FLAGS=--use-local-env -DSTRATA_ISA_FLOOR=avx || exit /b 1

D:\polystrata-maxwell\PolyStrata-0.1.44.1\.venv\Lib\site-packages\cmake\data\bin\cmake.exe --build D:\polystrata-maxwell\PolyStrata-0.1.44.1\build-maxwell --target strata strata-poly strata-dsv41 -j 16 && exit /b 0

echo   (the build stopped - trying it once more)

D:\polystrata-maxwell\PolyStrata-0.1.44.1\.venv\Lib\site-packages\cmake\data\bin\cmake.exe --build D:\polystrata-maxwell\PolyStrata-0.1.44.1\build-maxwell --target strata strata-poly strata-dsv41 -j 16 || exit /b 1




