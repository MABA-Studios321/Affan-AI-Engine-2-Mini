// Affan AI Engine - public C API.   (c) MABA Studios / Muhammad Affan Bin Adnan
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(AFFA_BUILD_DLL)
#define AFFA_API __declspec(dllexport)
#else
#define AFFA_API
#endif

typedef struct AffaModel AffaModel;

typedef struct {
    float temperature;    // 0 = greedy
    float top_p;
    int   top_k;
    float rep_penalty;    // 1.0 = off
    int   max_new_tokens;
    int   seed;           // <=0 : random
    int   think;          // 0 = bypass reasoning, 1 = deep reasoning (<think>)
} AffaParams;

typedef struct {
    int    prompt_tokens, reused_tokens, gen_tokens;
    double ttft_ms;       // request start -> first generated token
    double prefill_ms;
    double decode_ms;
    double tps;           // decode tokens / second
} AffaStats;

// return non-zero from the callback to stop generation. text is valid UTF-8 (not NUL terminated).
typedef int (*AffaTokenCb)(const char* text, int len, void* user);

AFFA_API AffaModel* affa_load_file(const char* utf8_path, int n_threads, int n_ctx, char* err, int errlen);
AFFA_API AffaModel* affa_load_fd(int fd, int n_threads, int n_ctx, char* err, int errlen);   // POSIX
AFFA_API void       affa_free(AffaModel*);
AFFA_API const char* affa_info_json(AffaModel*);
AFFA_API const char* affa_gpu_device(AffaModel*);
AFFA_API const char* affa_get_backend(AffaModel*);
// mode: "auto" (CUDA -> DirectML/D3D12 -> CPU AVX2), "cuda", "dml" (or "d3d12"), "cpu"
AFFA_API int        affa_set_backend(AffaModel*, const char* mode);
AFFA_API void       affa_default_params(AffaModel*, AffaParams*);
AFFA_API void       affa_set_threads(AffaModel*, int n);
AFFA_API void       affa_stop(AffaModel*);
AFFA_API void       affa_reset(AffaModel*);

// roles: "system" | "user" | "assistant".  system may be NULL -> model default; "" -> none.
AFFA_API int affa_chat(AffaModel*, int n_msgs, const char** roles, const char** contents,
                       const char* system, const AffaParams*, AffaTokenCb cb, void* user, AffaStats* stats);

AFFA_API int affa_tokenize(AffaModel*, const char* text, int* out, int max_out);
AFFA_API int affa_detokenize(AffaModel*, const int* ids, int n, char* out, int max_out);
// Test hook: runs ids from an empty cache. all_positions!=0 -> out is [n*vocab], else [vocab] (last token).
AFFA_API int affa_forward_logits(AffaModel*, const int* ids, int n, float* out, int all_positions);
// Benchmark with synthetic tokens. out[0]=prefill tok/s, out[1]=decode tok/s, out[2]=prefill ms
AFFA_API int affa_bench(AffaModel*, int n_prompt, int n_gen, double* out3);

#ifdef __cplusplus
}
#endif
