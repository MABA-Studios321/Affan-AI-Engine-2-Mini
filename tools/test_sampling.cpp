#include "affa_engine.h"
#include <cstdio>
#include <vector>
#include <thread>

static int on_tok(const char* t, int n, void*) {
    fwrite(t, 1, n, stdout);
    return 0;
}

static void run_query(AffaModel* m, const char* label, const char* q) {
    printf("=== %s: '%s' ===\n", label, q);
    std::thread([&]() {
        const char* r[] = {"user"}; const char* c[] = {q};
        AffaParams p; affa_default_params(m, &p); p.temperature = 0.6f; p.top_p = 0.85f; p.rep_penalty = 1.1f; p.max_new_tokens = 180;
        AffaStats st;
        affa_chat(m, 1, r, c, nullptr, &p, on_tok, nullptr, &st);
        printf("\n[TTFT: %.1f ms, TPS: %.1f]\n\n", st.ttft_ms, st.tps);
    }).join();
}

int main() {
    char err[512] = {0};
    AffaModel* m = affa_load_file("D:\\affan_ai_engine_2_mini\\models\\affan_ai_engine_2_mini.affa", 4, 2048, err, sizeof err);
    if (!m) { printf("Load error: %s\n", err); return 1; }
    printf("GPU Device: '%s'\n\n", affa_gpu_device(m));

    run_query(m, "Test 1 (Personality)", "who was osman");
    run_query(m, "Test 2 (Personality)", "who was osman ghazi");
    run_query(m, "Test 3 (Personality)", "who is albert einstein");
    run_query(m, "Test 4 (Self Identity)", "who are you");
    run_query(m, "Test 5 (Creator Identity)", "who created you");

    affa_free(m);
    return 0;
}
