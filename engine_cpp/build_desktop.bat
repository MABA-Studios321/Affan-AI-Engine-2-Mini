@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if not exist D:\affan_ai_engine_2_mini\dist mkdir D:\affan_ai_engine_2_mini\dist
if not exist D:\affan_ai_engine_2_mini\build mkdir D:\affan_ai_engine_2_mini\build
cd /d D:\affan_ai_engine_2_mini\engine_cpp

echo Compiling Affan AI Engine 2 Mini (Standalone Native Windows Executable with CUDA + DirectML/D3D12 + AVX2)...
cl /nologo /std:c++17 /O2 /arch:AVX2 /EHsc /fp:precise /W3 /D_CRT_SECURE_NO_WARNINGS /DUNICODE /D_UNICODE ^
   affa_engine.cpp affa_cuda_backend.cpp affa_d3d12_backend.cpp affa_server.cpp main_desktop.cpp ^
   ws2_32.lib comdlg32.lib shell32.lib user32.lib advapi32.lib ^
   /Fo:D:\affan_ai_engine_2_mini\build\ "/Fe:D:\affan_ai_engine_2_mini\dist\Affan AI Engine 2 Mini.exe" /link /SUBSYSTEM:WINDOWS

if %ERRORLEVEL% equ 0 (
    echo SUCCESS: D:\affan_ai_engine_2_mini\dist\Affan AI Engine 2 Mini.exe built successfully!
) else (
    echo FAILED to build EXE.
)


