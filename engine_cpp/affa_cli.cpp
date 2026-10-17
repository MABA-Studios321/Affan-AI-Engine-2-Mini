// affa_cli - command line driver / test tool for the Affan AI Engine
#include "affa_engine.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int on_tok(const char* t, int n, void*) { fwrite(t, 1, n, stdout); fflush(stdout); return 0; }

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: affa_cli model.affa <tok TEXT | logits IDS_CSV OUT.bin | bench [P G] | chat TEXT> [-t threads] [-c ctx] [-b auto|cuda|dml|cpu]\n");
        return 1;
    }
    int threads = 0, ctx = 2048;
    std::string backend = "auto";
    for (int i = 3; i + 1 < argc; i++) {
        if (!strcmp(argv[i], "-t")) threads = atoi(argv[i + 1]);
        if (!strcmp(argv[i], "-c")) ctx = atoi(argv[i + 1]);
        if (!strcmp(argv[i], "-b")) backend = argv[i + 1];
    }
    char err[512] = {0};
    AffaModel* m = affa_load_file(argv[1], threads, ctx, err, sizeof err);
    if (!m) { fprintf(stderr, "load failed: %s\n", err); return 2; }
    if (backend != "auto") {
        if (affa_set_backend(m, backend.c_str()) != 0) {
            fprintf(stderr, "failed to select backend: %s\n", backend.c_str());
            affa_free(m);
            return 3;
        }
    }
    std::string cmd = argv[2];
    if (cmd == "tokf") {
        FILE* f = fopen(argv[3], "rb"); std::string txt; char buf[4096]; size_t k;
        while ((k = fread(buf, 1, sizeof buf, f)) > 0) txt.append(buf, k); fclose(f);
        std::vector<int> ids(65536); int n = affa_tokenize(m, txt.c_str(), ids.data(), (int)ids.size());
        for (int i = 0; i < n; i++) printf("%d%s", ids[i], i + 1 < n ? "," : "\n");
    } else if (cmd == "tok") {
        std::vector<int> ids(65536); int n = affa_tokenize(m, argv[3], ids.data(), (int)ids.size());
        for (int i = 0; i < n; i++) printf("%d%s", ids[i], i + 1 < n ? "," : "\n");
    } else if (cmd == "logits") {
        std::vector<int> ids; char* s = _strdup(argv[3]);
        for (char* p = strtok(s, ","); p; p = strtok(nullptr, ",")) ids.push_back(atoi(p));
        std::vector<float> out((size_t)ids.size() * 200000);
        int V = affa_forward_logits(m, ids.data(), (int)ids.size(), out.data(), 1);
        if (V < 0) { fprintf(stderr, "forward failed\n"); return 3; }
        FILE* f = fopen(argv[4], "wb"); fwrite(out.data(), 4, (size_t)ids.size() * V, f); fclose(f);
        printf("%d positions x %d vocab written\n", (int)ids.size(), V);
    } else if (cmd == "bench") {
        int P = 64, G = 64; if (argc > 4 && argv[3][0] != '-') { P = atoi(argv[3]); G = atoi(argv[4]); }
        double r[3]; affa_bench(m, 8, 4, r);   // warm-up
        affa_bench(m, P, G, r);
        printf("%s\n", affa_info_json(m));
        printf("prefill %d tok: %.1f ms (%.1f tok/s)   decode %d tok: %.2f tok/s\n", P, r[2], r[0], G, r[1]);
    } else if (cmd == "chat") {
        const char* roles[1] = {"user"}; const char* cont[1] = {argv[3]};
        AffaParams p; affa_default_params(m, &p); p.max_new_tokens = 200; p.temperature = 0; AffaStats st;
        affa_chat(m, 1, roles, cont, nullptr, &p, on_tok, nullptr, &st);
        printf("\n[backend %s | prompt %d tok, ttft %.0f ms, %d tok, %.1f tok/s]\n", affa_get_backend(m), st.prompt_tokens, st.ttft_ms, st.gen_tokens, st.tps);
    }
    affa_free(m);
    return 0;
}
