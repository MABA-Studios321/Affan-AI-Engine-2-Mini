#pragma once
#include "affa_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

// Universal DirectML / DirectX 12 GPU Compute Backend (RX 580 / AMD / Intel Arc & Iris / NVIDIA)
// Dynamically loads d3d12.dll and dxgi.dll at runtime so zero SDK installation is required.
int   affa_d3d12_device_name(char* out, int n);
void* affa_d3d12_create(const AffaCudaCfg* cfg, const int8_t* embed, const float* erscale, const float* fnorm,
                        const AffaCudaLayer* layers, const float* rcos, const float* rsin, char* err, int errlen);
void  affa_d3d12_forward(void* h, const int* ids, int n, int pos0, int mode, float* out);
void  affa_d3d12_sync(void* h);
void  affa_d3d12_free(void* h);

#ifdef __cplusplus
}
#endif
