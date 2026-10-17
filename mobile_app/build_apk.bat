@echo off
setlocal enabledelayedexpansion

set ANDROID_SDK=C:\Users\AFFAN\AppData\Local\Android\Sdk
set NDK=%ANDROID_SDK%\ndk\27.2.12479018
set BUILD_TOOLS=%ANDROID_SDK%\build-tools\34.0.0
set PLATFORM=%ANDROID_SDK%\platforms\android-34\android.jar
set JAVA_HOME=C:\Program Files\Android\Android Studio\jbr
set PATH=%JAVA_HOME%\bin;%BUILD_TOOLS%;%PATH%

set PROJ_DIR=D:\affan_ai_engine_2_mini\mobile_app\android_project
set OUT_DIR=D:\affan_ai_engine_2_mini\dist
if not exist %OUT_DIR% mkdir %OUT_DIR%

set BUILD_TMP=D:\affan_ai_engine_2_mini\mobile_app\build_tmp
if exist %BUILD_TMP% rd /s /q %BUILD_TMP%
mkdir %BUILD_TMP%\obj
mkdir %BUILD_TMP%\dex
mkdir %BUILD_TMP%\res_compiled

set CLANG_ARM64=%NDK%\toolchains\llvm\prebuilt\windows-x86_64\bin\aarch64-linux-android26-clang++.cmd
set CLANG_ARMV7=%NDK%\toolchains\llvm\prebuilt\windows-x86_64\bin\armv7a-linux-androideabi26-clang++.cmd

echo ========================================================
echo [1/5] Compiling Native C++ Ternary Engine (.so) for ARM64 and ARMv7...
echo ========================================================
mkdir %BUILD_TMP%\lib\arm64-v8a
call "%CLANG_ARM64%" -O3 -ffast-math -funroll-loops -march=armv8.2-a+dotprod+fp16 -std=c++17 -fPIC -shared ^
    -static-libstdc++ ^
    -Wl,--no-undefined ^
    -I"D:\affan_ai_engine_2_mini\engine_cpp" ^
    "D:\affan_ai_engine_2_mini\engine_cpp\affa_engine.cpp" ^
    "D:\affan_ai_engine_2_mini\engine_cpp\affa_server.cpp" ^
    "%PROJ_DIR%\app\src\main\jni\affa_jni.cpp" ^
    -llog -landroid ^
    -o "%BUILD_TMP%\lib\arm64-v8a\libaffa_engine.so"

if %ERRORLEVEL% neq 0 (
    echo ARM64 Compilation Failed!
    exit /b 1
)

mkdir %BUILD_TMP%\lib\armeabi-v7a
call "%CLANG_ARMV7%" -O3 -std=c++17 -fPIC -shared ^
    -static-libstdc++ ^
    -Wl,--no-undefined ^
    -I"D:\affan_ai_engine_2_mini\engine_cpp" ^
    "D:\affan_ai_engine_2_mini\engine_cpp\affa_engine.cpp" ^
    "D:\affan_ai_engine_2_mini\engine_cpp\affa_server.cpp" ^
    "%PROJ_DIR%\app\src\main\jni\affa_jni.cpp" ^
    -llog -landroid ^
    -o "%BUILD_TMP%\lib\armeabi-v7a\libaffa_engine.so"

echo ========================================================
echo [2/5] Compiling Android Resources with AAPT2...
echo ========================================================
"%BUILD_TOOLS%\aapt2.exe" compile --dir "%PROJ_DIR%\app\src\main\res" -o "%BUILD_TMP%\res_compiled\res.zip"
"%BUILD_TOOLS%\aapt2.exe" link -o "%BUILD_TMP%\base.apk" ^
    -I "%PLATFORM%" ^
    "%BUILD_TMP%\res_compiled\res.zip" ^
    --manifest "%PROJ_DIR%\app\src\main\AndroidManifest.xml" ^
    --min-sdk-version 26 ^
    --target-sdk-version 34 ^
    --java "%BUILD_TMP%\gen" ^
    --auto-add-overlay

echo ========================================================
echo [3/5] Compiling Java Sources to Bytecode...
echo ========================================================
"%JAVA_HOME%\bin\javac.exe" -source 1.8 -target 1.8 ^
    -bootclasspath "%PLATFORM%" ^
    -cp "%BUILD_TMP%\gen" ^
    -d "%BUILD_TMP%\obj" ^
    "%PROJ_DIR%\app\src\main\java\com\maba\affanai\MainActivity.java"

echo ========================================================
echo [4/5] Converting Bytecode to DEX (D8)...
echo ========================================================
"%JAVA_HOME%\bin\jar.exe" -cf "%BUILD_TMP%\classes.jar" -C "%BUILD_TMP%\obj" .
"%JAVA_HOME%\bin\java.exe" -cp "%BUILD_TOOLS%\lib\d8.jar" com.android.tools.r8.D8 ^
    --release --min-api 26 --output "%BUILD_TMP%\dex" "%BUILD_TMP%\classes.jar"

echo ========================================================
echo [5/5] Packaging, Aligning, and Signing APK...
echo ========================================================
copy "%BUILD_TMP%\base.apk" "%BUILD_TMP%\unaligned.apk" >nul

"%JAVA_HOME%\bin\jar.exe" -uf "%BUILD_TMP%\unaligned.apk" -C "%BUILD_TMP%\dex" classes.dex
"%JAVA_HOME%\bin\jar.exe" -uf "%BUILD_TMP%\unaligned.apk" -C "%BUILD_TMP%" lib
mkdir "%BUILD_TMP%\assets"
copy /y "D:\affan_ai_engine_2_mini\models\affan_ai_engine_2_mini.affa" "%BUILD_TMP%\assets\" >nul
"%JAVA_HOME%\bin\jar.exe" -u0f "%BUILD_TMP%\unaligned.apk" -C "%BUILD_TMP%" assets

"%BUILD_TOOLS%\zipalign.exe" -p -f 4 "%BUILD_TMP%\unaligned.apk" "%BUILD_TMP%\aligned.apk"

set KEYSTORE=D:\affan_ai_engine_2_mini\mobile_app\debug.keystore
if not exist "%KEYSTORE%" (
    "%JAVA_HOME%\bin\keytool.exe" -genkeypair -v -keystore "%KEYSTORE%" ^
        -storepass android -alias androiddebugkey -keypass android -keyalg RSA -keysize 2048 -validity 10000 ^
        -dname "CN=MABA Studios, OU=Affan AI, O=MABA, L=Karachi, S=Sindh, C=PK" >nul 2>&1
)

"%JAVA_HOME%\bin\java.exe" -jar "%BUILD_TOOLS%\lib\apksigner.jar" sign ^
    --ks "%KEYSTORE%" --ks-pass pass:android ^
    --out "%OUT_DIR%\Affan AI Engine 2 Mini.apk" "%BUILD_TMP%\aligned.apk"

if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================================
    echo SUCCESS! Standalone APK Generated:
    echo %OUT_DIR%\Affan AI Engine 2 Mini.apk
    echo ========================================================
) else (
    echo APK Signing Failed!
    exit /b 1
)
