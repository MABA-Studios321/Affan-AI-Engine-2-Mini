#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int H;
    int I;
    int L;
    int nh;
    int nkv;
    int hd;
    int vocab;
    int n_ctx;
    float eps;
} AffaCudaCfg;

typedef struct {
    const float* ln1;
    const float* ln2;
    const float* qn;
    const float* kn;
    const uint8_t* w[7]; // q, k, v, o, g, u, d
    float s[7];          // scales
} AffaCudaLayer;

int   affa_cuda_device_name(char* out, int n);
void* affa_cuda_create(const AffaCudaCfg* cfg, const int8_t* embed, const float* erscale, const float* fnorm,
                       const AffaCudaLayer* layers, const float* rcos, const float* rsin, char* err, int errlen);
void  affa_cuda_forward(void* h, const int* ids, int n, int pos0, int mode, float* out);
void  affa_cuda_sync(void* h);
void  affa_cuda_free(void* h);

#ifdef __cplusplus
}
#endif
