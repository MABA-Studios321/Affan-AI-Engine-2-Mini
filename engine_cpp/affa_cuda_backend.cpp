// Affan AI Engine 2 - CUDA Driver API Backend for RTX 5050
// Loads nvcuda.dll dynamically and executes compiled PTX kernels directly on GPU.
// Uses CUDA Graph capture (cuStreamBeginCapture / cuGraphLaunch) for single-token decode to launch all 28 layers in 1 driver call!
#include "affa_cuda.h"
#include "affa_ptx.h"
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

namespace {

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef unsigned __int64 CUdeviceptr;
typedef void* CUstream;
typedef void* CUgraph;
typedef void* CUgraphExec;

typedef CUresult (__stdcall *cuInit_f)(unsigned int);
typedef CUresult (__stdcall *cuDeviceGet_f)(CUdevice*, int);
typedef CUresult (__stdcall *cuDeviceGetName_f)(char*, int, CUdevice);
typedef CUresult (__stdcall *cuDeviceGetAttribute_f)(int*, int, CUdevice);
typedef CUresult (__stdcall *cuCtxCreate_f)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (__stdcall *cuCtxSetCurrent_f)(CUcontext);
typedef CUresult (__stdcall *cuCtxDestroy_f)(CUcontext);
typedef CUresult (__stdcall *cuModuleLoadData_f)(CUmodule*, const void*);
typedef CUresult (__stdcall *cuModuleUnload_f)(CUmodule);
typedef CUresult (__stdcall *cuModuleGetFunction_f)(CUfunction*, CUmodule, const char*);
typedef CUresult (__stdcall *cuMemAlloc_f)(CUdeviceptr*, size_t);
typedef CUresult (__stdcall *cuMemFree_f)(CUdeviceptr);
typedef CUresult (__stdcall *cuMemAllocHost_f)(void**, size_t);
typedef CUresult (__stdcall *cuMemFreeHost_f)(void*);
typedef CUresult (__stdcall *cuMemcpyHtoD_f)(CUdeviceptr, const void*, size_t);
typedef CUresult (__stdcall *cuMemcpyDtoH_f)(void*, CUdeviceptr, size_t);
typedef CUresult (__stdcall *cuMemcpyHtoDAsync_f)(CUdeviceptr, const void*, size_t, CUstream);
typedef CUresult (__stdcall *cuMemcpyDtoHAsync_f)(void*, CUdeviceptr, size_t, CUstream);
typedef CUresult (__stdcall *cuMemsetD8_f)(CUdeviceptr, unsigned char, size_t);
typedef CUresult (__stdcall *cuLaunchKernel_f)(CUfunction, unsigned int, unsigned int, unsigned int,
                                              unsigned int, unsigned int, unsigned int,
                                              unsigned int, CUstream, void**, void**);
typedef CUresult (__stdcall *cuCtxSynchronize_f)(void);
typedef CUresult (__stdcall *cuStreamCreate_f)(CUstream*, unsigned int);
typedef CUresult (__stdcall *cuStreamDestroy_f)(CUstream);
typedef CUresult (__stdcall *cuStreamSynchronize_f)(CUstream);
typedef CUresult (__stdcall *cuStreamBeginCapture_f)(CUstream, int);
typedef CUresult (__stdcall *cuStreamEndCapture_f)(CUstream, CUgraph*);
typedef CUresult (__stdcall *cuGraphInstantiate_f)(CUgraphExec*, CUgraph, unsigned long long);
typedef CUresult (__stdcall *cuGraphInstantiateLegacy_f)(CUgraphExec*, CUgraph, void*, char*, size_t);
typedef CUresult (__stdcall *cuGraphLaunch_f)(CUgraphExec, CUstream);
typedef CUresult (__stdcall *cuGraphExecDestroy_f)(CUgraphExec);
typedef CUresult (__stdcall *cuGraphDestroy_f)(CUgraph);

struct CudaDrv {
    HMODULE hLib = NULL;
    cuInit_f cuInit = NULL;
    cuDeviceGet_f cuDeviceGet = NULL;
    cuDeviceGetName_f cuDeviceGetName = NULL;
    cuDeviceGetAttribute_f cuDeviceGetAttribute = NULL;
    cuCtxCreate_f cuCtxCreate = NULL;
    cuCtxSetCurrent_f cuCtxSetCurrent = NULL;
    cuCtxDestroy_f cuCtxDestroy = NULL;
    cuModuleLoadData_f cuModuleLoadData = NULL;
    cuModuleUnload_f cuModuleUnload = NULL;
    cuModuleGetFunction_f cuModuleGetFunction = NULL;
    cuMemAlloc_f cuMemAlloc = NULL;
    cuMemFree_f cuMemFree = NULL;
    cuMemAllocHost_f cuMemAllocHost = NULL;
    cuMemFreeHost_f cuMemFreeHost = NULL;
    cuMemcpyHtoD_f cuMemcpyHtoD = NULL;
    cuMemcpyDtoH_f cuMemcpyDtoH = NULL;
    cuMemcpyHtoDAsync_f cuMemcpyHtoDAsync = NULL;
    cuMemcpyDtoHAsync_f cuMemcpyDtoHAsync = NULL;
    cuMemsetD8_f cuMemsetD8 = NULL;
    cuLaunchKernel_f cuLaunchKernel = NULL;
    cuCtxSynchronize_f cuCtxSynchronize = NULL;
    cuStreamCreate_f cuStreamCreate = NULL;
    cuStreamDestroy_f cuStreamDestroy = NULL;
    cuStreamSynchronize_f cuStreamSynchronize = NULL;
    cuStreamBeginCapture_f cuStreamBeginCapture = NULL;
    cuStreamEndCapture_f cuStreamEndCapture = NULL;
    cuGraphInstantiate_f cuGraphInstantiate = NULL;
    cuGraphInstantiateLegacy_f cuGraphInstantiateLegacy = NULL;
    cuGraphLaunch_f cuGraphLaunch = NULL;
    cuGraphExecDestroy_f cuGraphExecDestroy = NULL;
    cuGraphDestroy_f cuGraphDestroy = NULL;

    bool init() {
        if (hLib) return true;
        hLib = LoadLibraryA("nvcuda.dll");
        if (!hLib) return false;
        cuInit = (cuInit_f)GetProcAddress(hLib, "cuInit");
        cuDeviceGet = (cuDeviceGet_f)GetProcAddress(hLib, "cuDeviceGet");
        cuDeviceGetName = (cuDeviceGetName_f)GetProcAddress(hLib, "cuDeviceGetName");
        cuDeviceGetAttribute = (cuDeviceGetAttribute_f)GetProcAddress(hLib, "cuDeviceGetAttribute");
        cuCtxCreate = (cuCtxCreate_f)GetProcAddress(hLib, "cuCtxCreate_v2");
        cuCtxSetCurrent = (cuCtxSetCurrent_f)GetProcAddress(hLib, "cuCtxSetCurrent");
        cuCtxDestroy = (cuCtxDestroy_f)GetProcAddress(hLib, "cuCtxDestroy_v2");
        cuModuleLoadData = (cuModuleLoadData_f)GetProcAddress(hLib, "cuModuleLoadData");
        cuModuleUnload = (cuModuleUnload_f)GetProcAddress(hLib, "cuModuleUnload");
        cuModuleGetFunction = (cuModuleGetFunction_f)GetProcAddress(hLib, "cuModuleGetFunction");
        cuMemAlloc = (cuMemAlloc_f)GetProcAddress(hLib, "cuMemAlloc_v2");
        cuMemFree = (cuMemFree_f)GetProcAddress(hLib, "cuMemFree_v2");
        cuMemAllocHost = (cuMemAllocHost_f)GetProcAddress(hLib, "cuMemAllocHost_v2");
        cuMemFreeHost = (cuMemFreeHost_f)GetProcAddress(hLib, "cuMemFreeHost");
        cuMemcpyHtoD = (cuMemcpyHtoD_f)GetProcAddress(hLib, "cuMemcpyHtoD_v2");
        cuMemcpyDtoH = (cuMemcpyDtoH_f)GetProcAddress(hLib, "cuMemcpyDtoH_v2");
        cuMemcpyHtoDAsync = (cuMemcpyHtoDAsync_f)GetProcAddress(hLib, "cuMemcpyHtoDAsync_v2");
        cuMemcpyDtoHAsync = (cuMemcpyDtoHAsync_f)GetProcAddress(hLib, "cuMemcpyDtoHAsync_v2");
        cuMemsetD8 = (cuMemsetD8_f)GetProcAddress(hLib, "cuMemsetD8_v2");
        cuLaunchKernel = (cuLaunchKernel_f)GetProcAddress(hLib, "cuLaunchKernel");
        cuCtxSynchronize = (cuCtxSynchronize_f)GetProcAddress(hLib, "cuCtxSynchronize");
        cuStreamCreate = (cuStreamCreate_f)GetProcAddress(hLib, "cuStreamCreate");
        cuStreamDestroy = (cuStreamDestroy_f)GetProcAddress(hLib, "cuStreamDestroy_v2");
        cuStreamSynchronize = (cuStreamSynchronize_f)GetProcAddress(hLib, "cuStreamSynchronize");
        cuStreamBeginCapture = (cuStreamBeginCapture_f)GetProcAddress(hLib, "cuStreamBeginCapture_v2");
        cuStreamEndCapture = (cuStreamEndCapture_f)GetProcAddress(hLib, "cuStreamEndCapture");
        cuGraphInstantiate = (cuGraphInstantiate_f)GetProcAddress(hLib, "cuGraphInstantiateWithFlags");
        cuGraphInstantiateLegacy = (cuGraphInstantiateLegacy_f)GetProcAddress(hLib, "cuGraphInstantiate");
        cuGraphLaunch = (cuGraphLaunch_f)GetProcAddress(hLib, "cuGraphLaunch");
        cuGraphExecDestroy = (cuGraphExecDestroy_f)GetProcAddress(hLib, "cuGraphExecDestroy");
        cuGraphDestroy = (cuGraphDestroy_f)GetProcAddress(hLib, "cuGraphDestroy");

        if (!cuInit || !cuDeviceGet || !cuDeviceGetName || !cuCtxCreate || !cuCtxSetCurrent || !cuModuleLoadData ||
            !cuModuleGetFunction || !cuMemAlloc || !cuMemFree || !cuMemcpyHtoD || !cuMemcpyDtoH ||
            !cuLaunchKernel || !cuCtxSynchronize) {
            FreeLibrary(hLib);
            hLib = NULL;
            return false;
        }
        return cuInit(0) == 0;
    }
} g_drv;

struct CudaWeight {
    CUdeviceptr d = 0;
    int rows = 0, cols = 0;
    float scale = 0.f;
};

struct CudaLayerData {
    CUdeviceptr ln1 = 0, ln2 = 0, qn = 0, kn = 0;
    CudaWeight w[7]; // q, k, v, o, g, u, d
};

constexpr int CHG = 256;

struct AffaCudaState {
    AffaCudaCfg cfg{};
    CUcontext ctx = NULL;
    CUmodule mod = NULL;
    CUstream stream = NULL;
    CUgraph graph_decode = NULL;
    CUgraphExec graph_exec = NULL;
    bool graph_ready = false;

    CUfunction fn_embed = NULL;
    CUfunction fn_norm_quant = NULL;
    CUfunction fn_matmul_1 = NULL;
    CUfunction fn_qkv_1 = NULL;
    CUfunction fn_gu_1 = NULL;
    CUfunction fn_matmul_8 = NULL;
    CUfunction fn_add = NULL;
    CUfunction fn_qk_rope = NULL;
    CUfunction fn_attn = NULL;
    CUfunction fn_logits = NULL;

    CUdeviceptr d_embed = 0, d_erscale = 0, d_fnorm = 0;
    CUdeviceptr d_kc = 0, d_vc = 0;
    CUdeviceptr d_rcos = 0, d_rsin = 0;
    CUdeviceptr d_x = 0, d_q = 0, d_k = 0, d_v = 0, d_ao = 0, d_g = 0, d_u = 0, d_big = 0;
    CUdeviceptr d_xq = 0, d_xinv = 0, d_logits = 0;
    CUdeviceptr d_ids = 0, d_pos = 0;

    int* h_ids = NULL;
    int* h_pos = NULL;
    float* h_logits = NULL;

    std::vector<CudaLayerData> L;
    std::vector<CUdeviceptr> allocs;
    std::string dev_name;
    size_t smem_attn = 0;
};

CUdeviceptr d_alloc(AffaCudaState* s, size_t sz) {
    CUdeviceptr p = 0;
    if (g_drv.cuMemAlloc(&p, sz) != 0) return 0;
    s->allocs.push_back(p);
    return p;
}

CUdeviceptr d_upload(AffaCudaState* s, const void* src, size_t sz) {
    CUdeviceptr p = d_alloc(s, sz);
    if (!p) return 0;
    if (g_drv.cuMemcpyHtoD(p, src, sz) != 0) return 0;
    return p;
}

void launch_mm(AffaCudaState* s, const CudaWeight& w, CUdeviceptr d_xq, CUdeviceptr d_out, int n, CUstream st) {
    CUdeviceptr wd = w.d;
    int rows = w.rows;
    int cols = w.cols;
    float scale = w.scale;
    void* args[] = { (void*)&wd, (void*)&rows, (void*)&cols, (void*)&scale, (void*)&d_xq, (void*)&s->d_xinv, (void*)&d_out, (void*)&n };
    if (n == 1) {
        int gx = (w.rows + 31) / 32;
        g_drv.cuLaunchKernel(s->fn_matmul_1, gx, 1, 1, 256, 1, 1, 0, st, args, NULL);
    } else {
        int gx = (w.rows + 7) / 8;
        int gy = (n + 7) / 8;
        g_drv.cuLaunchKernel(s->fn_matmul_8, gx, gy, 1, 256, 1, 1, 0, st, args, NULL);
    }
}

void enqueue_forward(AffaCudaState* s, int n, int mode, CUstream st = NULL) {
    AffaCudaCfg& c = s->cfg;
    int H = c.H;
    int I = c.I;
    int nh = c.nh;
    int nkv = c.nkv;
    int hd = c.hd;
    int vocab = c.vocab;
    int n_ctx = c.n_ctx;
    float eps = c.eps;
    int qdim = nh * hd, kvdim = nkv * hd;
    float scale = 1.0f / std::sqrt((float)hd);

    // 1. Embedding lookup
    {
        void* args[] = { (void*)&s->d_embed, (void*)&s->d_erscale, (void*)&s->d_ids, (void*)&s->d_x, (void*)&H };
        g_drv.cuLaunchKernel(s->fn_embed, n, 1, 1, 256, 1, 1, 0, st, args, NULL);
    }

    // 2. Layer 0 input norm
    {
        CUdeviceptr null_ptr = 0;
        int m0 = 0;
        size_t sh_bytes = (size_t)H * 4;
        void* args[] = { (void*)&s->d_x, (void*)&null_ptr, (void*)&s->L[0].ln1, (void*)&H, (void*)&eps, (void*)&m0, (void*)&s->d_xq, (void*)&s->d_xinv };
        g_drv.cuLaunchKernel(s->fn_norm_quant, n, 1, 1, 256, 1, 1, (unsigned int)sh_bytes, st, args, NULL);
    }

    // 3. Loop over transformer layers
    for (int l = 0; l < c.L; l++) {
        CudaLayerData& ly = s->L[l];
        if (n == 1 && s->fn_qkv_1) {
            int total_qkv = qdim + 2 * kvdim;
            int gx = (total_qkv + 31) / 32;
            void* args[] = {
                (void*)&ly.w[0].d, (void*)&ly.w[1].d, (void*)&ly.w[2].d,
                (void*)&qdim, (void*)&kvdim, (void*)&H,
                (void*)&ly.w[0].scale, (void*)&ly.w[1].scale, (void*)&ly.w[2].scale,
                (void*)&s->d_xq, (void*)&s->d_xinv,
                (void*)&s->d_q, (void*)&s->d_k, (void*)&s->d_v
            };
            g_drv.cuLaunchKernel(s->fn_qkv_1, gx, 1, 1, 256, 1, 1, 0, st, args, NULL);
        } else {
            launch_mm(s, ly.w[0], s->d_xq, s->d_q, n, st);
            launch_mm(s, ly.w[1], s->d_xq, s->d_k, n, st);
            launch_mm(s, ly.w[2], s->d_xq, s->d_v, n, st);
        }

        // QK-Norm + RoPE + Cache store
        {
            size_t sh_bytes = (size_t)hd * 4;
            void* args[] = { (void*)&s->d_q, (void*)&s->d_k, (void*)&s->d_v, (void*)&ly.qn, (void*)&ly.kn,
                             (void*)&s->d_rcos, (void*)&s->d_rsin, (void*)&s->d_kc, (void*)&s->d_vc, (void*)&s->d_pos,
                             (void*)&nh, (void*)&hd, (void*)&qdim, (void*)&kvdim, (void*)&eps, (void*)&l, (void*)&n_ctx };
            g_drv.cuLaunchKernel(s->fn_qk_rope, n, nh + nkv, 1, hd, 1, 1, (unsigned int)sh_bytes, st, args, NULL);
        }

        // Multi-head / Grouped-query Attention
        {
            void* args[] = { (void*)&s->d_q, (void*)&s->d_ao, (void*)&s->d_kc, (void*)&s->d_vc, (void*)&s->d_pos,
                             (void*)&nh, (void*)&nkv, (void*)&hd, (void*)&qdim, (void*)&kvdim, (void*)&l, (void*)&n_ctx, (void*)&scale };
            g_drv.cuLaunchKernel(s->fn_attn, n * nh, 1, 1, hd, 1, 1, (unsigned int)s->smem_attn, st, args, NULL);
        }

        // Quantize attention output
        {
            CUdeviceptr null_ptr = 0;
            int m1 = 1;
            size_t sh_bytes = (size_t)qdim * 4;
            void* args[] = { (void*)&s->d_ao, (void*)&null_ptr, (void*)&null_ptr, (void*)&qdim, (void*)&eps, (void*)&m1, (void*)&s->d_xq, (void*)&s->d_xinv };
            g_drv.cuLaunchKernel(s->fn_norm_quant, n, 1, 1, 256, 1, 1, (unsigned int)sh_bytes, st, args, NULL);
        }

        // Out projection matmul
        launch_mm(s, ly.w[3], s->d_xq, s->d_big, n, st);

        // Residual add + ln2 + quantize
        {
            int m0 = 0;
            size_t sh_bytes = (size_t)H * 4;
            void* args[] = { (void*)&s->d_x, (void*)&s->d_big, (void*)&ly.ln2, (void*)&H, (void*)&eps, (void*)&m0, (void*)&s->d_xq, (void*)&s->d_xinv };
            g_drv.cuLaunchKernel(s->fn_norm_quant, n, 1, 1, 256, 1, 1, (unsigned int)sh_bytes, st, args, NULL);
        }

        // Gate & Up matmuls (SwiGLU)
        if (n == 1 && s->fn_gu_1) {
            int gx = (2 * I + 31) / 32;
            void* args[] = {
                (void*)&ly.w[4].d, (void*)&ly.w[5].d,
                (void*)&I, (void*)&H,
                (void*)&ly.w[4].scale, (void*)&ly.w[5].scale,
                (void*)&s->d_xq, (void*)&s->d_xinv,
                (void*)&s->d_g, (void*)&s->d_u
            };
            g_drv.cuLaunchKernel(s->fn_gu_1, gx, 1, 1, 256, 1, 1, 0, st, args, NULL);
        } else {
            launch_mm(s, ly.w[4], s->d_xq, s->d_g, n, st);
            launch_mm(s, ly.w[5], s->d_xq, s->d_u, n, st);
        }

        // Silu(g) * u + quantize
        {
            CUdeviceptr null_ptr = 0;
            int m2 = 2;
            size_t sh_bytes = (size_t)I * 4;
            void* args[] = { (void*)&s->d_g, (void*)&s->d_u, (void*)&null_ptr, (void*)&I, (void*)&eps, (void*)&m2, (void*)&s->d_xq, (void*)&s->d_xinv };
            g_drv.cuLaunchKernel(s->fn_norm_quant, n, 1, 1, 256, 1, 1, (unsigned int)sh_bytes, st, args, NULL);
        }

        // Down projection matmul
        launch_mm(s, ly.w[6], s->d_xq, s->d_big, n, st);

        // Residual add + next layer ln1 (or final norm)
        CUdeviceptr next_norm = (l + 1 < c.L) ? s->L[l + 1].ln1 : s->d_fnorm;
        {
            int m0 = 0;
            size_t sh_bytes = (size_t)H * 4;
            void* args[] = { (void*)&s->d_x, (void*)&s->d_big, (void*)&next_norm, (void*)&H, (void*)&eps, (void*)&m0, (void*)&s->d_xq, (void*)&s->d_xinv };
            g_drv.cuLaunchKernel(s->fn_norm_quant, n, 1, 1, 256, 1, 1, (unsigned int)sh_bytes, st, args, NULL);
        }

        // If last layer and mode == 1, compute logits for the last token
        if (l + 1 == c.L && mode == 1) {
            CUdeviceptr xl = s->d_xq + (size_t)(n - 1) * H;
            CUdeviceptr xinv_last = s->d_xinv + (size_t)(n - 1) * 4;
            int gx = (vocab + 31) / 32;
            void* args[] = { (void*)&s->d_embed, (void*)&s->d_erscale, (void*)&xl, (void*)&xinv_last, (void*)&s->d_logits, (void*)&vocab, (void*)&H };
            g_drv.cuLaunchKernel(s->fn_logits, gx, 1, 1, 256, 1, 1, 0, st, args, NULL);
        }
    }
}

void build_decode_graph(AffaCudaState* s) {
    if (!g_drv.cuStreamCreate || !g_drv.cuStreamBeginCapture || !g_drv.cuStreamEndCapture || !g_drv.cuGraphLaunch) return;
    if (!g_drv.cuMemcpyHtoDAsync || !g_drv.cuMemcpyDtoHAsync) return;
    if (g_drv.cuStreamCreate(&s->stream, 1) != 0 || !s->stream) return;

    if (g_drv.cuStreamBeginCapture(s->stream, 0) != 0) return;

    g_drv.cuMemcpyHtoDAsync(s->d_ids, s->h_ids, 4, s->stream);
    g_drv.cuMemcpyHtoDAsync(s->d_pos, s->h_pos, 4, s->stream);
    enqueue_forward(s, 1, 1, s->stream);
    g_drv.cuMemcpyDtoHAsync(s->h_logits, s->d_logits, (size_t)s->cfg.vocab * 4, s->stream);

    if (g_drv.cuStreamEndCapture(s->stream, &s->graph_decode) != 0 || !s->graph_decode) return;

    CUresult r = -1;
    if (g_drv.cuGraphInstantiate) {
        r = g_drv.cuGraphInstantiate(&s->graph_exec, s->graph_decode, 0);
    } else if (g_drv.cuGraphInstantiateLegacy) {
        r = g_drv.cuGraphInstantiateLegacy(&s->graph_exec, s->graph_decode, NULL, NULL, 0);
    }
    if (r == 0 && s->graph_exec) {
        s->graph_ready = true;
    }
}

} // namespace

extern "C" {

int affa_cuda_device_name(char* out, int n) {
    if (!g_drv.init()) return -1;
    CUdevice dev = 0;
    if (g_drv.cuDeviceGet(&dev, 0) != 0) return -1;
    if (g_drv.cuDeviceGetName(out, n, dev) != 0) return -1;
    return 0;
}

void* affa_cuda_create(const AffaCudaCfg* cfg, const int8_t* embed, const float* erscale, const float* fnorm,
                       const AffaCudaLayer* layers, const float* rcos, const float* rsin, char* err, int errlen) {
    if (!g_drv.init()) {
        if (err && errlen) snprintf(err, errlen, "CUDA Driver not found (nvcuda.dll)");
        return nullptr;
    }
    CUdevice dev = 0;
    if (g_drv.cuDeviceGet(&dev, 0) != 0) {
        if (err && errlen) snprintf(err, errlen, "No CUDA device found");
        return nullptr;
    }

    AffaCudaState* s = new AffaCudaState();
    s->cfg = *cfg;
    char name[256] = {0};
    g_drv.cuDeviceGetName(name, sizeof(name), dev);
    s->dev_name = name;

    if (g_drv.cuCtxCreate(&s->ctx, 0, dev) != 0) {
        if (err && errlen) snprintf(err, errlen, "Failed to create CUDA context");
        delete s;
        return nullptr;
    }

    if (g_drv.cuModuleLoadData(&s->mod, kAffaPtx) != 0) {
        if (err && errlen) snprintf(err, errlen, "Failed to load CUDA PTX module");
        affa_cuda_free(s);
        return nullptr;
    }

    // Resolve kernel functions
    g_drv.cuModuleGetFunction(&s->fn_embed, s->mod, "k_embed");
    g_drv.cuModuleGetFunction(&s->fn_norm_quant, s->mod, "k_norm_quant");
    g_drv.cuModuleGetFunction(&s->fn_matmul_1, s->mod, "k_matmul_1");
    g_drv.cuModuleGetFunction(&s->fn_qkv_1, s->mod, "k_qkv_1");
    g_drv.cuModuleGetFunction(&s->fn_gu_1, s->mod, "k_gu_1");
    g_drv.cuModuleGetFunction(&s->fn_matmul_8, s->mod, "k_matmul_8");
    g_drv.cuModuleGetFunction(&s->fn_add, s->mod, "k_add");
    g_drv.cuModuleGetFunction(&s->fn_qk_rope, s->mod, "k_qk_rope");
    g_drv.cuModuleGetFunction(&s->fn_attn, s->mod, "k_attn");
    g_drv.cuModuleGetFunction(&s->fn_logits, s->mod, "k_logits");

    if (!s->fn_embed || !s->fn_norm_quant || !s->fn_matmul_1 || !s->fn_matmul_8 ||
        !s->fn_add || !s->fn_qk_rope || !s->fn_attn || !s->fn_logits) {
        if (err && errlen) snprintf(err, errlen, "Failed to find CUDA kernel functions");
        affa_cuda_free(s);
        return nullptr;
    }

    const AffaCudaCfg& c = s->cfg;
    const int qdim = c.nh * c.hd, kvdim = c.nkv * c.hd;

    // Upload embeddings & norms
    s->d_embed = d_upload(s, embed, (size_t)c.vocab * c.H);
    s->d_erscale = d_upload(s, erscale, (size_t)c.vocab * 4);
    s->d_fnorm = d_upload(s, fnorm, (size_t)c.H * 4);

    const int rows[7] = { qdim, kvdim, kvdim, c.H, c.I, c.I, c.H };
    const int cols[7] = { c.H, c.H, c.H, qdim, c.H, c.H, c.I };
    s->L.resize(c.L);
    for (int l = 0; l < c.L; l++) {
        CudaLayerData& cd = s->L[l];
        const AffaCudaLayer& sl = layers[l];
        cd.ln1 = d_upload(s, sl.ln1, (size_t)c.H * 4);
        cd.ln2 = d_upload(s, sl.ln2, (size_t)c.H * 4);
        cd.qn = d_upload(s, sl.qn, (size_t)c.hd * 4);
        cd.kn = d_upload(s, sl.kn, (size_t)c.hd * 4);
        for (int i = 0; i < 7; i++) {
            cd.w[i].d = d_upload(s, sl.w[i], (size_t)rows[i] * cols[i] / 4);
            cd.w[i].rows = rows[i];
            cd.w[i].cols = cols[i];
            cd.w[i].scale = sl.s[i];
        }
    }

    // Allocate KV Cache
    const size_t kv_bytes = (size_t)c.L * c.n_ctx * kvdim * 2;
    s->d_kc = d_alloc(s, kv_bytes);
    s->d_vc = d_alloc(s, kv_bytes);
    g_drv.cuMemsetD8(s->d_kc, 0, kv_bytes);
    g_drv.cuMemsetD8(s->d_vc, 0, kv_bytes);

    // RoPE tables
    const int half = c.hd / 2;
    s->d_rcos = d_upload(s, rcos, (size_t)c.n_ctx * half * 4);
    s->d_rsin = d_upload(s, rsin, (size_t)c.n_ctx * half * 4);

    // Working activations
    int mc = c.H > qdim ? c.H : qdim;
    if (c.I > mc) mc = c.I;
    s->d_x = d_alloc(s, (size_t)CHG * c.H * 4);
    s->d_q = d_alloc(s, (size_t)CHG * qdim * 4);
    s->d_k = d_alloc(s, (size_t)CHG * kvdim * 4);
    s->d_v = d_alloc(s, (size_t)CHG * kvdim * 4);
    s->d_ao = d_alloc(s, (size_t)CHG * qdim * 4);
    s->d_g = d_alloc(s, (size_t)CHG * c.I * 4);
    s->d_u = d_alloc(s, (size_t)CHG * c.I * 4);
    s->d_big = d_alloc(s, (size_t)CHG * c.H * 4);
    s->d_xq = d_alloc(s, (size_t)CHG * mc);
    s->d_xinv = d_alloc(s, CHG * 4);
    s->d_logits = d_alloc(s, (size_t)c.vocab * 4);
    s->d_ids = d_alloc(s, CHG * 4);
    s->d_pos = d_alloc(s, 4);

    // Pinned host buffers
    g_drv.cuMemAllocHost((void**)&s->h_ids, CHG * 4);
    g_drv.cuMemAllocHost((void**)&s->h_pos, 4);
    g_drv.cuMemAllocHost((void**)&s->h_logits, (size_t)c.vocab * 4);

    s->smem_attn = ((size_t)c.n_ctx + c.hd) * 4;
    build_decode_graph(s);
    g_drv.cuCtxSynchronize();
    return s;
}

void affa_cuda_forward(void* h, const int* ids, int n, int pos0, int mode, float* out) {
    AffaCudaState* s = (AffaCudaState*)h;
    if (!s) return;
    g_drv.cuCtxSetCurrent(s->ctx);
    const AffaCudaCfg& c = s->cfg;

    if (mode == 2) {
        for (int t = 0; t < n; t++) {
            affa_cuda_forward(h, ids + t, 1, pos0 + t, 1, out + (size_t)t * c.vocab);
        }
        return;
    }

    while (n > CHG) {
        affa_cuda_forward(h, ids, CHG, pos0, 0, nullptr);
        ids += CHG; n -= CHG; pos0 += CHG;
    }

    // Fast path: 1-token decode via CUDA Graph (all 28 layers + HtoD + DtoH in 1 driver call!)
    if (n == 1 && mode == 1 && s->graph_ready) {
        s->h_ids[0] = ids[0];
        *s->h_pos = pos0;
        g_drv.cuGraphLaunch(s->graph_exec, s->stream);
        g_drv.cuStreamSynchronize(s->stream);
        memcpy(out, s->h_logits, (size_t)c.vocab * 4);
        return;
    }

    memcpy(s->h_ids, ids, (size_t)n * 4);
    *s->h_pos = pos0;
    g_drv.cuMemcpyHtoD(s->d_ids, s->h_ids, (size_t)n * 4);
    g_drv.cuMemcpyHtoD(s->d_pos, s->h_pos, 4);

    enqueue_forward(s, n, mode, NULL);

    if (mode == 1) {
        g_drv.cuMemcpyDtoH(s->h_logits, s->d_logits, (size_t)c.vocab * 4);
        g_drv.cuCtxSynchronize();
        memcpy(out, s->h_logits, (size_t)c.vocab * 4);
    } else {
        g_drv.cuCtxSynchronize();
    }
}

void affa_cuda_sync(void* h) {
    AffaCudaState* s = (AffaCudaState*)h;
    if (s) {
        g_drv.cuCtxSetCurrent(s->ctx);
        g_drv.cuCtxSynchronize();
    }
}

void affa_cuda_free(void* h) {
    AffaCudaState* s = (AffaCudaState*)h;
    if (!s) return;
    if (s->ctx) g_drv.cuCtxSetCurrent(s->ctx);
    g_drv.cuCtxSynchronize();
    if (s->graph_exec && g_drv.cuGraphExecDestroy) g_drv.cuGraphExecDestroy(s->graph_exec);
    if (s->graph_decode && g_drv.cuGraphDestroy) g_drv.cuGraphDestroy(s->graph_decode);
    if (s->stream && g_drv.cuStreamDestroy) g_drv.cuStreamDestroy(s->stream);
    for (CUdeviceptr p : s->allocs) {
        if (p) g_drv.cuMemFree(p);
    }
    if (s->h_ids) g_drv.cuMemFreeHost(s->h_ids);
    if (s->h_pos) g_drv.cuMemFreeHost(s->h_pos);
    if (s->h_logits) g_drv.cuMemFreeHost(s->h_logits);
    if (s->mod) g_drv.cuModuleUnload(s->mod);
    if (s->ctx) g_drv.cuCtxDestroy(s->ctx);
    delete s;
}

} // extern "C"
