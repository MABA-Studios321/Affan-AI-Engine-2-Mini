// Affan AI Engine 2 - CUDA backend (RTX / Blackwell / Ada / Ampere)
// (c) MABA Studios / Muhammad Affan Bin Adnan
//
// Same math as the CPU engine, executed on the GPU:
//   * ternary weights stay 2-bit packed in VRAM, decoded through a 256-entry LUT into int8x4 and
//     multiplied with int8 activations using __dp4a  (integer MAC, no float weight math)
//   * per-token absmax int8 activation quantisation, RMSNorm, QK-norm, RoPE, GQA attention, SwiGLU
//   * bf16 KV cache resident in VRAM, CUDA-graph captured decode step (one launch per token)
#include "affa_cuda.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

#define CKR(x, ret) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { if (err && errlen) snprintf(err, errlen, "CUDA: %s (%s:%d)", cudaGetErrorString(e_), __FILE__, __LINE__); return ret; } } while (0)

namespace {

constexpr int CHG = 256;     // max tokens per GPU forward call

struct WT { const uint8_t* d; int rows, cols; float scale; };
struct LayerD { const float *ln1, *ln2, *qn, *kn; WT w[7]; };   // q k v o g u d

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ float block_sum(float v, float* red) {
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float r = lane < nw ? red[lane] : 0.f;
    r = warp_sum(r);
    return r;
}
__device__ float block_max(float v, float* red) {
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float r = lane < nw ? red[lane] : -3.0e38f;
    r = warp_max(r);
    return r;
}

__global__ void k_embed(const int8_t* emb, const float* rs, const int* ids, float* x, int H) {
    int t = blockIdx.x; int id = ids[t]; float s = rs[id];
    const int8_t* er = emb + (size_t)id * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) x[(size_t)t * H + i] = (float)er[i] * s;
}

// mode 0: v = x (+add);  h = rmsnorm(v) * w          -> quantise      (x updated when add != null)
// mode 1: h = x                                       -> quantise
// mode 2: h = silu(x) * add                           -> quantise
__global__ void k_norm_quant(float* x, const float* add, const float* w, int cols, float eps, int mode,
                             int8_t* xq, float* xinv) {
    extern __shared__ float sh[];
    __shared__ float red[32];
    const int t = blockIdx.x;
    float* xr = x + (size_t)t * cols;
    const float* ar = add ? add + (size_t)t * cols : nullptr;
    float ss = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v;
        if (mode == 2) { float g = xr[i]; v = g / (1.0f + expf(-g)) * ar[i]; }
        else if (ar) { v = xr[i] + ar[i]; xr[i] = v; }
        else v = xr[i];
        sh[i] = v; ss += v * v;
    }
    float r = 1.0f;
    if (mode == 0) { ss = block_sum(ss, red); r = 1.0f / sqrtf(ss / (float)cols + eps); }
    float am = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float h = mode == 0 ? sh[i] * r * w[i] : sh[i];
        sh[i] = h; am = fmaxf(am, fabsf(h));
    }
    am = block_max(am, red);
    am = fmaxf(am, 1e-5f);
    const float s = 127.0f / am;
    int8_t* q = xq + (size_t)t * cols;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        int v = (int)rintf(sh[i] * s); v = v > 127 ? 127 : (v < -127 ? -127 : v);
        q[i] = (int8_t)v;
    }
    if (threadIdx.x == 0) xinv[t] = 1.0f / s;
}

template <int TT>
__global__ void k_matmul(const uint8_t* __restrict__ W, int rows, int cols, float wscale,
                         const int8_t* __restrict__ xq, const float* __restrict__ xinv,
                         float* __restrict__ out, int n) {
    __shared__ uint32_t lut[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        uint32_t v = 0;
        for (int k = 0; k < 4; k++) { int c = (i >> (2 * k)) & 3; int w = (c & 1) - ((c >> 1) & 1); v |= ((uint32_t)(w & 0xFF)) << (8 * k); }
        lut[i] = v;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + warp;
    if (r >= rows) return;
    const int tbase = blockIdx.y * TT;
    const uint32_t* wr = (const uint32_t*)(W + (size_t)r * (cols / 4));
    const int nw = cols / 16;
    int acc[TT];
#pragma unroll
    for (int i = 0; i < TT; i++) acc[i] = 0;
    for (int wi = lane; wi < nw; wi += 32) {
        uint32_t w = wr[wi];
        int d0 = (int)lut[w & 255], d1 = (int)lut[(w >> 8) & 255], d2 = (int)lut[(w >> 16) & 255], d3 = (int)lut[w >> 24];
#pragma unroll
        for (int tt = 0; tt < TT; tt++) {
            int t = min(tbase + tt, n - 1);
            int4 x4 = ((const int4*)(xq + (size_t)t * cols))[wi];
            acc[tt] = __dp4a(d0, x4.x, acc[tt]); acc[tt] = __dp4a(d1, x4.y, acc[tt]);
            acc[tt] = __dp4a(d2, x4.z, acc[tt]); acc[tt] = __dp4a(d3, x4.w, acc[tt]);
        }
    }
#pragma unroll
    for (int tt = 0; tt < TT; tt++) {
        int a = acc[tt];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        int t = tbase + tt;
        if (lane == 0 && t < n) out[(size_t)t * rows + r] = (float)a * (wscale * xinv[t]);
    }
}

__global__ void k_add(float* x, const float* y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// grid (n, nh+nkv), block hd
__global__ void k_qk_rope(float* q, float* k, const float* v, const float* qn, const float* kn,
                          const float* rcos, const float* rsin, __nv_bfloat16* kc, __nv_bfloat16* vc,
                          const int* pos0p, int nh, int hd, int qdim, int kvdim, float eps, int layer, int n_ctx) {
    extern __shared__ float y[];
    __shared__ float red[32];
    const int t = blockIdx.x, hh = blockIdx.y, d = threadIdx.x, half = hd / 2;
    const int pos = *pos0p + t;
    const bool isq = hh < nh;
    float* base = isq ? q + (size_t)t * qdim + hh * hd : k + (size_t)t * kvdim + (hh - nh) * hd;
    const float* w = isq ? qn : kn;
    float val = base[d];
    float ss = block_sum(val * val, red);
    float r = 1.0f / sqrtf(ss / (float)hd + eps);
    float yv = val * r * w[d];
    y[d] = yv;
    __syncthreads();
    int i = d < half ? d : d - half;
    float c = rcos[(size_t)pos * half + i], s = rsin[(size_t)pos * half + i];
    float o;
    if (d < half) o = yv * c - y[d + half] * s; else o = yv * c + y[d - half] * s;
    base[d] = o;
    if (!isq) {
        size_t off = ((size_t)layer * n_ctx + pos) * kvdim + (hh - nh) * hd + d;
        kc[off] = __float2bfloat16_rn(o);
        vc[off] = __float2bfloat16_rn(v[(size_t)t * kvdim + (hh - nh) * hd + d]);
    }
}

// grid n*nh, block hd.  dyn smem: (n_ctx + hd) floats
__global__ void k_attn(const float* q, float* ao, const __nv_bfloat16* kc, const __nv_bfloat16* vc, const int* pos0p,
                       int nh, int nkv, int hd, int qdim, int kvdim, int layer, int n_ctx, float scale) {
    extern __shared__ float sm[];
    __shared__ float red[32];
    float* sc = sm; float* qs = sm + n_ctx;
    const int t = blockIdx.x / nh, hh = blockIdx.x % nh;
    const int pos = *pos0p + t, kvh = hh / (nh / nkv);
    qs[threadIdx.x] = q[(size_t)t * qdim + hh * hd + threadIdx.x];
    __syncthreads();
    const __nv_bfloat16* Kb = kc + (size_t)layer * n_ctx * kvdim + kvh * hd;
    const __nv_bfloat16* Vb = vc + (size_t)layer * n_ctx * kvdim + kvh * hd;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarp = blockDim.x >> 5;
    for (int j = warp; j <= pos; j += nwarp) {
        float s = 0.f;
        for (int d = lane; d < hd; d += 32) s += qs[d] * __bfloat162float(Kb[(size_t)j * kvdim + d]);
        s = warp_sum(s);
        if (lane == 0) sc[j] = s * scale;
    }
    __syncthreads();
    float mx = -3.0e38f;
    for (int j = threadIdx.x; j <= pos; j += blockDim.x) mx = fmaxf(mx, sc[j]);
    mx = block_max(mx, red);
    float sum = 0.f;
    for (int j = threadIdx.x; j <= pos; j += blockDim.x) { float e = expf(sc[j] - mx); sc[j] = e; sum += e; }
    sum = block_sum(sum, red);
    const float inv = 1.0f / sum;
    __syncthreads();
    float acc = 0.f;
    const int d = threadIdx.x;
    for (int j = 0; j <= pos; j++) acc += (sc[j] * inv) * __bfloat162float(Vb[(size_t)j * kvdim + d]);
    ao[(size_t)t * qdim + hh * hd + d] = acc;
}

__global__ void k_logits(const int8_t* emb, const float* rs, const int8_t* xq, const float* xinv, float* out, int vocab, int H) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + warp;
    if (r >= vocab) return;
    const int4* er = (const int4*)(emb + (size_t)r * H);
    const int4* xv = (const int4*)xq;
    int acc = 0;
    for (int i = lane; i < H / 16; i += 32) {
        int4 e = er[i], x = xv[i];
        acc = __dp4a(e.x, x.x, acc); acc = __dp4a(e.y, x.y, acc); acc = __dp4a(e.z, x.z, acc); acc = __dp4a(e.w, x.w, acc);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) out[r] = (float)acc * rs[r] * xinv[0];
}

}  // namespace

struct AffaCuda {
    AffaCudaCfg c{};
    cudaStream_t st = nullptr;
    int8_t* embed = nullptr; float* erscale = nullptr; float* fnorm = nullptr;
    std::vector<LayerD> L;
    std::vector<void*> allocs;
    __nv_bfloat16 *kc = nullptr, *vc = nullptr;
    float *rcos = nullptr, *rsin = nullptr;
    float *x = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *ao = nullptr, *g = nullptr, *u = nullptr, *big = nullptr;
    int8_t* xq = nullptr; float* xinv = nullptr; float* logits = nullptr;
    int* d_ids = nullptr; int* d_pos = nullptr;
    int* h_ids = nullptr; int* h_pos = nullptr; float* h_logits = nullptr;
    cudaGraphExec_t gexec = nullptr; bool graph_tried = false;
    std::string devname;
    size_t smem_attn = 0;
};

static void* dmalloc(AffaCuda* g, size_t n) { void* p = nullptr; if (cudaMalloc(&p, n) != cudaSuccess) return nullptr; g->allocs.push_back(p); return p; }
static void* dupload(AffaCuda* g, const void* src, size_t n) {
    void* p = dmalloc(g, n); if (!p) return nullptr;
    if (cudaMemcpy(p, src, n, cudaMemcpyHostToDevice) != cudaSuccess) return nullptr; return p;
}

extern "C" {

int affa_cuda_device_name(char* out, int n) {
    int cnt = 0; if (cudaGetDeviceCount(&cnt) != cudaSuccess || cnt <= 0) return -1;
    cudaDeviceProp p; if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return -1;
    if (p.major < 6 || (p.major == 6 && p.minor < 1)) return -1;      // need __dp4a (sm_61+)
    snprintf(out, n, "%s", p.name); return 0;
}

void* affa_cuda_create(const AffaCudaCfg* cfg, const int8_t* embed, const float* erscale, const float* fnorm,
                       const AffaCudaLayer* layers, const float* rcos, const float* rsin, char* err, int errlen) {
    char nm[256]; if (affa_cuda_device_name(nm, sizeof nm) != 0) { if (err && errlen) snprintf(err, errlen, "no usable CUDA device"); return nullptr; }
    AffaCuda* g = new AffaCuda(); g->c = *cfg; g->devname = nm;
    const AffaCudaCfg& c = g->c;
    const int qdim = c.nh * c.hd, kvdim = c.nkv * c.hd;
    auto fail = [&](const char* m) { if (err && errlen) snprintf(err, errlen, "CUDA init failed: %s", m); affa_cuda_free(g); return (void*)nullptr; };
    if (c.hd > 256 || c.hd % 32) return fail("unsupported head_dim");
    if (cudaStreamCreateWithFlags(&g->st, cudaStreamNonBlocking) != cudaSuccess) return fail("stream");
    g->embed = (int8_t*)dupload(g, embed, (size_t)c.vocab * c.H);
    g->erscale = (float*)dupload(g, erscale, (size_t)c.vocab * 4);
    g->fnorm = (float*)dupload(g, fnorm, (size_t)c.H * 4);
    if (!g->embed || !g->erscale || !g->fnorm) return fail("out of VRAM (embedding)");
    const int rows[7] = {qdim, kvdim, kvdim, c.H, c.I, c.I, c.H};
    const int cols[7] = {c.H, c.H, c.H, qdim, c.H, c.H, c.I};
    g->L.resize(c.L);
    for (int l = 0; l < c.L; l++) {
        LayerD& d = g->L[l]; const AffaCudaLayer& s = layers[l];
        d.ln1 = (float*)dupload(g, s.ln1, (size_t)c.H * 4); d.ln2 = (float*)dupload(g, s.ln2, (size_t)c.H * 4);
        d.qn = (float*)dupload(g, s.qn, (size_t)c.hd * 4); d.kn = (float*)dupload(g, s.kn, (size_t)c.hd * 4);
        if (!d.ln1 || !d.ln2 || !d.qn || !d.kn) return fail("out of VRAM (norms)");
        for (int i = 0; i < 7; i++) {
            d.w[i].d = (const uint8_t*)dupload(g, s.w[i], (size_t)rows[i] * cols[i] / 4);
            d.w[i].rows = rows[i]; d.w[i].cols = cols[i]; d.w[i].scale = s.s[i];
            if (!d.w[i].d) return fail("out of VRAM (weights)");
        }
    }
    const size_t kvn = (size_t)c.L * c.n_ctx * kvdim;
    g->kc = (__nv_bfloat16*)dmalloc(g, kvn * 2); g->vc = (__nv_bfloat16*)dmalloc(g, kvn * 2);
    if (!g->kc || !g->vc) return fail("out of VRAM (KV cache)");
    cudaMemset(g->kc, 0, kvn * 2); cudaMemset(g->vc, 0, kvn * 2);
    const int half = c.hd / 2;
    g->rcos = (float*)dupload(g, rcos, (size_t)c.n_ctx * half * 4); g->rsin = (float*)dupload(g, rsin, (size_t)c.n_ctx * half * 4);
    int mc = c.H > qdim ? c.H : qdim; if (c.I > mc) mc = c.I;
    g->x = (float*)dmalloc(g, (size_t)CHG * c.H * 4); g->q = (float*)dmalloc(g, (size_t)CHG * qdim * 4);
    g->k = (float*)dmalloc(g, (size_t)CHG * kvdim * 4); g->v = (float*)dmalloc(g, (size_t)CHG * kvdim * 4);
    g->ao = (float*)dmalloc(g, (size_t)CHG * qdim * 4); g->g = (float*)dmalloc(g, (size_t)CHG * c.I * 4);
    g->u = (float*)dmalloc(g, (size_t)CHG * c.I * 4); g->big = (float*)dmalloc(g, (size_t)CHG * c.H * 4);
    g->xq = (int8_t*)dmalloc(g, (size_t)CHG * mc); g->xinv = (float*)dmalloc(g, CHG * 4);
    g->logits = (float*)dmalloc(g, (size_t)c.vocab * 4);
    g->d_ids = (int*)dmalloc(g, CHG * 4); g->d_pos = (int*)dmalloc(g, 4);
    if (!g->x || !g->q || !g->k || !g->v || !g->ao || !g->g || !g->u || !g->big || !g->xq || !g->xinv || !g->logits || !g->d_ids || !g->d_pos)
        return fail("out of VRAM (activations)");
    if (cudaMallocHost((void**)&g->h_ids, CHG * 4) != cudaSuccess || cudaMallocHost((void**)&g->h_pos, 4) != cudaSuccess ||
        cudaMallocHost((void**)&g->h_logits, (size_t)c.vocab * 4) != cudaSuccess) return fail("pinned memory");
    g->smem_attn = ((size_t)c.n_ctx + c.hd) * 4;
    if (g->smem_attn > 48 * 1024) {
        if (cudaFuncSetAttribute(k_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)g->smem_attn) != cudaSuccess) return fail("attention smem");
    }
    cudaDeviceSynchronize();
    return g;
}

static void launch_mm(AffaCuda* g, const WT& w, const int8_t* xq, float* out, int n) {
    dim3 blk(256);
    if (n == 1) { dim3 grd((w.rows + 7) / 8, 1); k_matmul<1><<<grd, blk, 0, g->st>>>(w.d, w.rows, w.cols, w.scale, xq, g->xinv, out, n); }
    else { dim3 grd((w.rows + 7) / 8, (n + 7) / 8); k_matmul<8><<<grd, blk, 0, g->st>>>(w.d, w.rows, w.cols, w.scale, xq, g->xinv, out, n); }
}

static void enqueue_forward(AffaCuda* g, int n, int mode, int logits_all_t) {
    const AffaCudaCfg& c = g->c; cudaStream_t s = g->st;
    const int qdim = c.nh * c.hd, kvdim = c.nkv * c.hd;
    const float scale = 1.0f / sqrtf((float)c.hd);
    k_embed<<<n, 256, 0, s>>>(g->embed, g->erscale, g->d_ids, g->x, c.H);
    // layer 0 input norm (no residual add)
    k_norm_quant<<<n, 256, (size_t)c.H * 4, s>>>(g->x, nullptr, g->L[0].ln1, c.H, c.eps, 0, g->xq, g->xinv);
    for (int l = 0; l < c.L; l++) {
        const LayerD& ly = g->L[l];
        launch_mm(g, ly.w[0], g->xq, g->q, n); launch_mm(g, ly.w[1], g->xq, g->k, n); launch_mm(g, ly.w[2], g->xq, g->v, n);
        k_qk_rope<<<dim3(n, c.nh + c.nkv), c.hd, (size_t)c.hd * 4, s>>>(g->q, g->k, g->v, ly.qn, ly.kn, g->rcos, g->rsin, g->kc, g->vc,
                                                                      g->d_pos, c.nh, c.hd, qdim, kvdim, c.eps, l, c.n_ctx);
        k_attn<<<n * c.nh, c.hd, g->smem_attn, s>>>(g->q, g->ao, g->kc, g->vc, g->d_pos, c.nh, c.nkv, c.hd, qdim, kvdim, l, c.n_ctx, scale);
        k_norm_quant<<<n, 256, (size_t)qdim * 4, s>>>(g->ao, nullptr, nullptr, qdim, c.eps, 1, g->xq, g->xinv);
        launch_mm(g, ly.w[3], g->xq, g->big, n);
        // x += big ; h = rms(x)*ln2 ; quantise
        k_norm_quant<<<n, 256, (size_t)c.H * 4, s>>>(g->x, g->big, ly.ln2, c.H, c.eps, 0, g->xq, g->xinv);
        launch_mm(g, ly.w[4], g->xq, g->g, n); launch_mm(g, ly.w[5], g->xq, g->u, n);
        k_norm_quant<<<n, 256, (size_t)c.I * 4, s>>>(g->g, g->u, nullptr, c.I, c.eps, 2, g->xq, g->xinv);
        launch_mm(g, ly.w[6], g->xq, g->big, n);
        // x += big ; next layer's ln1 (or final norm)
        const float* nw = (l + 1 < c.L) ? g->L[l + 1].ln1 : g->fnorm;
        k_norm_quant<<<n, 256, (size_t)c.H * 4, s>>>(g->x, g->big, nw, c.H, c.eps, 0, g->xq, g->xinv);
        if (l + 1 == c.L && mode == 1) {
            // last token only: re-normalise handled above for all rows; logits use row n-1
            const int8_t* xl = g->xq + (size_t)(n - 1) * c.H;
            k_logits<<<(c.vocab + 7) / 8, 256, 0, s>>>(g->embed, g->erscale, xl, g->xinv + (n - 1), g->logits, c.vocab, c.H);
        }
    }
    (void)logits_all_t;
}

void affa_cuda_forward(void* h, const int* ids, int n, int pos0, int mode, float* out) {
    AffaCuda* g = (AffaCuda*)h; const AffaCudaCfg& c = g->c;
    if (mode == 2) {
        // all-position logits (used by verification only): run tokens one at a time
        for (int t = 0; t < n; t++) affa_cuda_forward(h, ids + t, 1, pos0 + t, 1, out + (size_t)t * c.vocab);
        return;
    }
    while (n > CHG) { affa_cuda_forward(h, ids, CHG, pos0, 0, nullptr); ids += CHG; n -= CHG; pos0 += CHG; }
    memcpy(g->h_ids, ids, (size_t)n * 4); *g->h_pos = pos0;
    cudaMemcpyAsync(g->d_ids, g->h_ids, (size_t)n * 4, cudaMemcpyHostToDevice, g->st);
    cudaMemcpyAsync(g->d_pos, g->h_pos, 4, cudaMemcpyHostToDevice, g->st);
    if (n == 1 && mode == 1) {
        if (!g->gexec && !g->graph_tried) {
            g->graph_tried = true;
            cudaGraph_t gr = nullptr;
            if (cudaStreamBeginCapture(g->st, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
                enqueue_forward(g, 1, 1, 0);
                cudaMemcpyAsync(g->h_logits, g->logits, (size_t)c.vocab * 4, cudaMemcpyDeviceToHost, g->st);
                if (cudaStreamEndCapture(g->st, &gr) == cudaSuccess && gr) {
                    if (cudaGraphInstantiate(&g->gexec, gr, 0) != cudaSuccess) g->gexec = nullptr;
                    cudaGraphDestroy(gr);
                } else cudaGetLastError();
            } else cudaGetLastError();
            // the capture run itself did not execute; fall through to a normal run below
        }
        if (g->gexec) {
            cudaGraphLaunch(g->gexec, g->st);
        } else {
            enqueue_forward(g, 1, 1, 0);
            cudaMemcpyAsync(g->h_logits, g->logits, (size_t)c.vocab * 4, cudaMemcpyDeviceToHost, g->st);
        }
        cudaStreamSynchronize(g->st);
        memcpy(out, g->h_logits, (size_t)c.vocab * 4);
        return;
    }
    enqueue_forward(g, n, mode, 0);
    if (mode == 1) {
        cudaMemcpyAsync(g->h_logits, g->logits, (size_t)c.vocab * 4, cudaMemcpyDeviceToHost, g->st);
        cudaStreamSynchronize(g->st);
        memcpy(out, g->h_logits, (size_t)c.vocab * 4);
    }
}

void affa_cuda_sync(void* h) { cudaStreamSynchronize(((AffaCuda*)h)->st); }

void affa_cuda_free(void* h) {
    AffaCuda* g = (AffaCuda*)h; if (!g) return;
    if (g->st) cudaStreamSynchronize(g->st);
    if (g->gexec) cudaGraphExecDestroy(g->gexec);
    for (void* p : g->allocs) cudaFree(p);
    if (g->h_ids) cudaFreeHost(g->h_ids); if (g->h_pos) cudaFreeHost(g->h_pos); if (g->h_logits) cudaFreeHost(g->h_logits);
    if (g->st) cudaStreamDestroy(g->st);
    delete g;
}

}  // extern "C"
