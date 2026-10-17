@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if not exist D:\affan_ai_engine_2_mini\build mkdir D:\affan_ai_engine_2_mini\build
cd /d D:\affan_ai_engine_2_mini\engine_cpp
cl /nologo /std:c++17 /O2 /arch:AVX2 /EHsc /fp:precise /W3 /D_CRT_SECURE_NO_WARNINGS ^
   affa_engine.cpp affa_cuda_backend.cpp affa_d3d12_backend.cpp affa_cli.cpp ^
   /Fo:D:\affan_ai_engine_2_mini\build\ /Fe:D:\affan_ai_engine_2_mini\build\affa_cli.exe
