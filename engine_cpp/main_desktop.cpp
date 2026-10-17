// Affan AI Engine Studio - Native Windows Desktop Entrypoint
// Launches the embedded high-speed C++ engine server and opens the minimalist UI.
#include "affa_server.h"
#include <windows.h>
#include <shellapi.h>
#include <string>
#include <thread>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    (void)hInstance; (void)hPrevInstance; (void)lpCmdLine; (void)nCmdShow;

    // Resolve models folder (D:\affan_ai_engine_2_mini\models or relative to exe)
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(NULL, exePath, MAX_PATH);
    fs::path exeDir = fs::path(exePath).parent_path();
    
    fs::path modelsDir = "D:\\affan_ai_engine_2_mini\\models";
    if (!fs::exists(modelsDir)) {
        modelsDir = exeDir / "models";
        std::error_code ec;
        fs::create_directories(modelsDir, ec);
    }

    AffaServerConfig cfg;
    cfg.platform = "windows";
    cfg.models_dir = modelsDir.u8string();
    cfg.extra_models_dir = (exeDir / "models").u8string();
    unsigned int hw = std::thread::hardware_concurrency();
    cfg.default_threads = (hw > 2u) ? hw : 2u;
    cfg.default_ctx = 2048;
    cfg.native_file_dialog = true;

    int port = affa_server_start(cfg, 37821);
    if (port <= 0) {
        MessageBoxW(NULL, L"Failed to start Affan AI Engine local server.", L"Affan AI Engine 2 Mini", MB_ICONERROR | MB_OK);
        return 1;
    }

    // Auto-load built-in model immediately
    std::string err;
    if (__argc > 1) {
        affa_server_load_path(__argv[1], err);
    } else {
        fs::path defaultModel = modelsDir / "affan_ai_engine_2_mini.affa";
        if (fs::exists(defaultModel)) {
            affa_server_load_path(defaultModel.u8string(), err);
        } else {
            for (const auto& entry : fs::directory_iterator(modelsDir)) {
                if (entry.path().extension() == ".affa") {
                    affa_server_load_path(entry.path().u8string(), err);
                    break;
                }
            }
        }
    }

    // Launch default web browser / webview window pointing to local server
    std::string url = "http://127.0.0.1:" + std::to_string(port);
    ShellExecuteA(NULL, "open", url.c_str(), NULL, NULL, SW_SHOWNORMAL);

    // Keep server process running until user closes application or idle timeout
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        long long idleMs = affa_server_ms_since_ping();
        // If the web window has been closed for more than 15 seconds after having connected
        if (idleMs > 15000) {
            break;
        }
    }

    affa_server_stop();
    return 0;
}
