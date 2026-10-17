// Affan AI Engine Studio - embedded local HTTP server (serves the UI + chat API)
#pragma once
#include "affa_engine.h"
#include <string>

struct AffaServerConfig {
    std::string platform = "windows";      // "windows" | "android"
    std::string models_dir;                // where uploaded / listed models live
    std::string extra_models_dir;          // optional second folder to scan
    int default_threads = 4;
    int default_ctx = 2048;
    bool native_file_dialog = false;       // Windows only
};

// returns the bound port (>0) or -1.  Binds 127.0.0.1 only.
int  affa_server_start(const AffaServerConfig& cfg, int preferred_port);
void affa_server_stop();
bool affa_server_load_path(const std::string& path, std::string& err);
bool affa_server_load_fd(int fd, const std::string& display_name, std::string& err);
// milliseconds since the UI last sent a heartbeat (-1 if it never did)
long long affa_server_ms_since_ping();
