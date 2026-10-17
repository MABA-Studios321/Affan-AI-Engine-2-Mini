#include "affa_server.h"
#include <cstdio>
#include <thread>
#include <chrono>

int main() {
    AffaServerConfig cfg;
    cfg.platform = "windows";
    cfg.models_dir = "D:\\affan_ai_engine_2_mini\\models";
    cfg.default_threads = 4;
    cfg.default_ctx = 2048;
    int port = affa_server_start(cfg, 37822);
    printf("Server started on port %d\n", port);
    std::string err;
    if (!affa_server_load_path("D:\\affan_ai_engine_2_mini\\models\\affan_ai_engine_2_mini.affa", err)) {
        printf("Load failed: %s\n", err.c_str());
        return 1;
    }
    printf("Model loaded successfully on port %d! Sleeping for 60 seconds...\n", port);
    std::this_thread::sleep_for(std::chrono::seconds(60));
    affa_server_stop();
    return 0;
}
