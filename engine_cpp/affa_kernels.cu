// Affan AI Engine 2 - Ultra-Fast CUDA Device Kernels (PTX target for RTX 5050 / Blackwell / Ada / Ampere / Turing)
// Pure device kernels for ternary 1.58-bit execution with zero-LUT bit-trick decoding, fused QKV/GU, 4-row blocking, and on-GPU Top-K.

typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;

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

__device__ __forceinline__ uint16_t float_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
__device__ __forceinline__ float bf16_to_float(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    return __uint_as_float(u);
}

__device__ __forceinline__ int decode_ternary_byte(uint32_t b) {
    int c0 = (int)(b & 3u);        int t0 = (c0 & 1) - ((c0 >> 1) & 1);
    int c1 = (int)((b >> 2) & 3u); int t1 = (c1 & 1) - ((c1 >> 1) & 1);
    int c2 = (int)((b >> 4) & 3u); int t2 = (c2 & 1) - ((c2 >> 1) & 1);
    int c3 = (int)((b >> 6) & 3u); int t3 = (c3 & 1) - ((c3 >> 1) & 1);
    return (t0 & 0xFF) | ((t1 & 0xFF) << 8) | ((t2 & 0xFF) << 16) | ((t3 & 0xFF) << 24);
}

#define DECODE_TERNARY_16(w, d0, d1, d2, d3) do { \
    d0 = decode_ternary_byte((w) & 0xFFu); \
    d1 = decode_ternary_byte(((w) >> 8) & 0xFFu); \
    d2 = decode_ternary_byte(((w) >> 16) & 0xFFu); \
    d3 = decode_ternary_byte((w) >> 24); \
} while(0)

extern "C" __global__ void k_embed(const int8_t* emb, const float* rs, const int* ids, float* x, int H) {
    int t = blockIdx.x; int id = ids[t]; float s = rs[id];
    const int8_t* er = emb + (size_t)id * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) x[(size_t)t * H + i] = (float)er[i] * s;
}

// mode 0: v = x (+add);  h = rmsnorm(v) * w          -> quantise      (x updated when add != null)
// mode 1: h = x                                       -> quantise
// mode 2: h = silu(x) * add                           -> quantise
extern "C" __global__ void k_norm_quant(float* x, const float* add, const float* w, int cols, float eps, int mode,
                                        int8_t* xq, float* xinv) {
    extern __shared__ float sh[];
    __shared__ float red[32];
    const int t = blockIdx.x;
    float* xr = x + (size_t)t * cols;
    const float* ar = add ? add + (size_t)t * cols : (const float*)0;
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

// 4-row-per-warp single-token ternary GEMV with shared-memory xq caching + zero-LUT SWAR decode
extern "C" __global__ void k_matmul_1(const uint8_t* __restrict__ W, int rows, int cols, float wscale,
                                      const int8_t* __restrict__ xq, const float* __restrict__ xinv,
                                      float* __restrict__ out, int n) {
    __shared__ int4 sxq[256]; // up to cols=4096 (256 * 16 bytes)
    const int nw = cols / 16;
    for (int i = threadIdx.x; i < nw; i += blockDim.x) {
        sxq[i] = ((const int4*)xq)[i];
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 4;
    if (r0 >= rows) return;
    const size_t rstride = (size_t)nw;
    const uint32_t* wr0 = (const uint32_t*)W + (size_t)r0 * rstride;
    const uint32_t* wr1 = wr0 + rstride;
    const uint32_t* wr2 = wr0 + 2 * rstride;
    const uint32_t* wr3 = wr0 + 3 * rstride;
    int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int wi = lane; wi < nw; wi += 32) {
        int4 x4 = sxq[wi];
        uint32_t w0 = wr0[wi], w1 = wr1[wi], w2 = wr2[wi], w3 = wr3[wi];
        int d0, d1, d2, d3;
        DECODE_TERNARY_16(w0, d0, d1, d2, d3);
        a0 = __dp4a(d0, x4.x, a0); a0 = __dp4a(d1, x4.y, a0); a0 = __dp4a(d2, x4.z, a0); a0 = __dp4a(d3, x4.w, a0);
        DECODE_TERNARY_16(w1, d0, d1, d2, d3);
        a1 = __dp4a(d0, x4.x, a1); a1 = __dp4a(d1, x4.y, a1); a1 = __dp4a(d2, x4.z, a1); a1 = __dp4a(d3, x4.w, a1);
        DECODE_TERNARY_16(w2, d0, d1, d2, d3);
        a2 = __dp4a(d0, x4.x, a2); a2 = __dp4a(d1, x4.y, a2); a2 = __dp4a(d2, x4.z, a2); a2 = __dp4a(d3, x4.w, a2);
        DECODE_TERNARY_16(w3, d0, d1, d2, d3);
        a3 = __dp4a(d0, x4.x, a3); a3 = __dp4a(d1, x4.y, a3); a3 = __dp4a(d2, x4.z, a3); a3 = __dp4a(d3, x4.w, a3);
    }
    for (int o = 16; o > 0; o >>= 1) {
        a0 += __shfl_xor_sync(0xffffffffu, a0, o);
        a1 += __shfl_xor_sync(0xffffffffu, a1, o);
        a2 += __shfl_xor_sync(0xffffffffu, a2, o);
        a3 += __shfl_xor_sync(0xffffffffu, a3, o);
    }
    if (lane == 0) {
        float sc = wscale * xinv[0];
        out[r0 + 0] = (float)a0 * sc;
        if (r0 + 1 < rows) out[r0 + 1] = (float)a1 * sc;
        if (r0 + 2 < rows) out[r0 + 2] = (float)a2 * sc;
        if (r0 + 3 < rows) out[r0 + 3] = (float)a3 * sc;
    }
}

// Fused Q+K+V single-token GEMV with zero-LUT SWAR decode
extern "C" __global__ void k_qkv_1(const uint8_t* __restrict__ Wq, const uint8_t* __restrict__ Wk, const uint8_t* __restrict__ Wv,
                                   int qdim, int kvdim, int cols, float sq, float sk, float sv,
                                   const int8_t* __restrict__ xq, const float* __restrict__ xinv,
                                   float* __restrict__ out_q, float* __restrict__ out_k, float* __restrict__ out_v) {
    __shared__ int4 sxq[256];
    const int nw = cols / 16;
    for (int i = threadIdx.x; i < nw; i += blockDim.x) sxq[i] = ((const int4*)xq)[i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 4;
    const int total_rows = qdim + 2 * kvdim;
    if (r0 >= total_rows) return;
    const uint8_t* W; float* out; float wscale; int r;
    if (r0 < qdim) { W = Wq; out = out_q; wscale = sq; r = r0; }
    else if (r0 < qdim + kvdim) { W = Wk; out = out_k; wscale = sk; r = r0 - qdim; }
    else { W = Wv; out = out_v; wscale = sv; r = r0 - qdim - kvdim; }

    const size_t rstride = (size_t)nw;
    const uint32_t* wr0 = (const uint32_t*)W + (size_t)r * rstride;
    const uint32_t* wr1 = wr0 + rstride;
    const uint32_t* wr2 = wr0 + 2 * rstride;
    const uint32_t* wr3 = wr0 + 3 * rstride;
    int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int wi = lane; wi < nw; wi += 32) {
        int4 x4 = sxq[wi];
        uint32_t w0 = wr0[wi], w1 = wr1[wi], w2 = wr2[wi], w3 = wr3[wi];
        int d0, d1, d2, d3;
        DECODE_TERNARY_16(w0, d0, d1, d2, d3);
        a0 = __dp4a(d0, x4.x, a0); a0 = __dp4a(d1, x4.y, a0); a0 = __dp4a(d2, x4.z, a0); a0 = __dp4a(d3, x4.w, a0);
        DECODE_TERNARY_16(w1, d0, d1, d2, d3);
        a1 = __dp4a(d0, x4.x, a1); a1 = __dp4a(d1, x4.y, a1); a1 = __dp4a(d2, x4.z, a1); a1 = __dp4a(d3, x4.w, a1);
        DECODE_TERNARY_16(w2, d0, d1, d2, d3);
        a2 = __dp4a(d0, x4.x, a2); a2 = __dp4a(d1, x4.y, a2); a2 = __dp4a(d2, x4.z, a2); a2 = __dp4a(d3, x4.w, a2);
        DECODE_TERNARY_16(w3, d0, d1, d2, d3);
        a3 = __dp4a(d0, x4.x, a3); a3 = __dp4a(d1, x4.y, a3); a3 = __dp4a(d2, x4.z, a3); a3 = __dp4a(d3, x4.w, a3);
    }
    for (int o = 16; o > 0; o >>= 1) {
        a0 += __shfl_xor_sync(0xffffffffu, a0, o); a1 += __shfl_xor_sync(0xffffffffu, a1, o);
        a2 += __shfl_xor_sync(0xffffffffu, a2, o); a3 += __shfl_xor_sync(0xffffffffu, a3, o);
    }
    if (lane == 0) {
        float sc = wscale * xinv[0];
        out[r + 0] = (float)a0 * sc; out[r + 1] = (float)a1 * sc;
        out[r + 2] = (float)a2 * sc; out[r + 3] = (float)a3 * sc;
    }
}

// Fused Gate+Up single-token GEMV with zero-LUT SWAR decode
extern "C" __global__ void k_gu_1(const uint8_t* __restrict__ Wg, const uint8_t* __restrict__ Wu,
                                  int rows, int cols, float sg, float su,
                                  const int8_t* __restrict__ xq, const float* __restrict__ xinv,
                                  float* __restrict__ out_g, float* __restrict__ out_u) {
    __shared__ int4 sxq[256];
    const int nw = cols / 16;
    for (int i = threadIdx.x; i < nw; i += blockDim.x) sxq[i] = ((const int4*)xq)[i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 4;
    if (r0 >= 2 * rows) return;
    const uint8_t* W = (r0 < rows) ? Wg : Wu;
    float* out = (r0 < rows) ? out_g : out_u;
    float wscale = (r0 < rows) ? sg : su;
    int r = (r0 < rows) ? r0 : (r0 - rows);

    const size_t rstride = (size_t)nw;
    const uint32_t* wr0 = (const uint32_t*)W + (size_t)r * rstride;
    const uint32_t* wr1 = wr0 + rstride;
    const uint32_t* wr2 = wr0 + 2 * rstride;
    const uint32_t* wr3 = wr0 + 3 * rstride;
    int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int wi = lane; wi < nw; wi += 32) {
        int4 x4 = sxq[wi];
        uint32_t w0 = wr0[wi], w1 = wr1[wi], w2 = wr2[wi], w3 = wr3[wi];
        int d0, d1, d2, d3;
        DECODE_TERNARY_16(w0, d0, d1, d2, d3);
        a0 = __dp4a(d0, x4.x, a0); a0 = __dp4a(d1, x4.y, a0); a0 = __dp4a(d2, x4.z, a0); a0 = __dp4a(d3, x4.w, a0);
        DECODE_TERNARY_16(w1, d0, d1, d2, d3);
        a1 = __dp4a(d0, x4.x, a1); a1 = __dp4a(d1, x4.y, a1); a1 = __dp4a(d2, x4.z, a1); a1 = __dp4a(d3, x4.w, a1);
        DECODE_TERNARY_16(w2, d0, d1, d2, d3);
        a2 = __dp4a(d0, x4.x, a2); a2 = __dp4a(d1, x4.y, a2); a2 = __dp4a(d2, x4.z, a2); a2 = __dp4a(d3, x4.w, a2);
        DECODE_TERNARY_16(w3, d0, d1, d2, d3);
        a3 = __dp4a(d0, x4.x, a3); a3 = __dp4a(d1, x4.y, a3); a3 = __dp4a(d2, x4.z, a3); a3 = __dp4a(d3, x4.w, a3);
    }
    for (int o = 16; o > 0; o >>= 1) {
        a0 += __shfl_xor_sync(0xffffffffu, a0, o); a1 += __shfl_xor_sync(0xffffffffu, a1, o);
        a2 += __shfl_xor_sync(0xffffffffu, a2, o); a3 += __shfl_xor_sync(0xffffffffu, a3, o);
    }
    if (lane == 0) {
        float sc = wscale * xinv[0];
        out[r + 0] = (float)a0 * sc; out[r + 1] = (float)a1 * sc;
        out[r + 2] = (float)a2 * sc; out[r + 3] = (float)a3 * sc;
    }
}

extern "C" __global__ void k_matmul_8(const uint8_t* __restrict__ W, int rows, int cols, float wscale,
                                      const int8_t* __restrict__ xq, const float* __restrict__ xinv,
                                      float* __restrict__ out, int n) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + warp;
    if (r >= rows) return;
    const int tbase = blockIdx.y * 8;
    const uint32_t* wr = (const uint32_t*)(W + (size_t)r * (cols / 4));
    const int nw = cols / 16;
    int acc[8];
    for (int i = 0; i < 8; i++) acc[i] = 0;
    for (int wi = lane; wi < nw; wi += 32) {
        uint32_t w = wr[wi];
        int d0, d1, d2, d3;
        DECODE_TERNARY_16(w, d0, d1, d2, d3);
        for (int tt = 0; tt < 8; tt++) {
            int t = min(tbase + tt, n - 1);
            int4 x4 = ((const int4*)(xq + (size_t)t * cols))[wi];
            acc[tt] = __dp4a(d0, x4.x, acc[tt]); acc[tt] = __dp4a(d1, x4.y, acc[tt]);
            acc[tt] = __dp4a(d2, x4.z, acc[tt]); acc[tt] = __dp4a(d3, x4.w, acc[tt]);
        }
    }
    for (int tt = 0; tt < 8; tt++) {
        int a = acc[tt];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        int t = tbase + tt;
        if (lane == 0 && t < n) out[(size_t)t * rows + r] = (float)a * (wscale * xinv[t]);
    }
}

extern "C" __global__ void k_add(float* x, const float* y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

extern "C" __global__ void k_qk_rope(float* q, float* k, const float* v, const float* qn, const float* kn,
                                     const float* rcos, const float* rsin, uint16_t* kc, uint16_t* vc,
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
        kc[off] = float_to_bf16(o);
        vc[off] = float_to_bf16(v[(size_t)t * kvdim + (hh - nh) * hd + d]);
    }
}

extern "C" __global__ void k_attn(const float* q, float* ao, const uint16_t* kc, const uint16_t* vc, const int* pos0p,
                                  int nh, int nkv, int hd, int qdim, int kvdim, int layer, int n_ctx, float scale) {
    extern __shared__ float sm[];
    __shared__ float red[32];
    float* sc = sm; float* qs = sm + n_ctx;
    const int t = blockIdx.x / nh, hh = blockIdx.x % nh;
    const int pos = *pos0p + t, kvh = hh / (nh / nkv);
    qs[threadIdx.x] = q[(size_t)t * qdim + hh * hd + threadIdx.x];
    __syncthreads();
    const uint16_t* Kb = kc + (size_t)layer * n_ctx * kvdim + kvh * hd;
    const uint16_t* Vb = vc + (size_t)layer * n_ctx * kvdim + kvh * hd;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarp = blockDim.x >> 5;
    for (int j = warp; j <= pos; j += nwarp) {
        float s = 0.f;
        for (int d = lane; d < hd; d += 32) s += qs[d] * bf16_to_float(Kb[(size_t)j * kvdim + d]);
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
    for (int j = 0; j <= pos; j++) acc += (sc[j] * inv) * bf16_to_float(Vb[(size_t)j * kvdim + d]);
    ao[(size_t)t * qdim + hh * hd + d] = acc;
}

// 4-row-per-warp vocabulary logits with shared-memory xq caching
extern "C" __global__ void k_logits(const int8_t* emb, const float* rs, const int8_t* xq, const float* xinv, float* out, int vocab, int H) {
    __shared__ int4 sxq[64]; // H=1024 -> 64 int4s
    const int nw = H / 16;
    for (int i = threadIdx.x; i < nw; i += blockDim.x) sxq[i] = ((const int4*)xq)[i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 4;
    if (r0 >= vocab) return;
    const int4* er0 = (const int4*)(emb + (size_t)r0 * H);
    const int4* er1 = (r0 + 1 < vocab) ? (er0 + nw) : er0;
    const int4* er2 = (r0 + 2 < vocab) ? (er0 + 2 * nw) : er0;
    const int4* er3 = (r0 + 3 < vocab) ? (er0 + 3 * nw) : er0;
    int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int i = lane; i < nw; i += 32) {
        int4 x = sxq[i];
        int4 e0 = er0[i], e1 = er1[i], e2 = er2[i], e3 = er3[i];
        a0 = __dp4a(e0.x, x.x, a0); a0 = __dp4a(e0.y, x.y, a0); a0 = __dp4a(e0.z, x.z, a0); a0 = __dp4a(e0.w, x.w, a0);
        a1 = __dp4a(e1.x, x.x, a1); a1 = __dp4a(e1.y, x.y, a1); a1 = __dp4a(e1.z, x.z, a1); a1 = __dp4a(e1.w, x.w, a1);
        a2 = __dp4a(e2.x, x.x, a2); a2 = __dp4a(e2.y, x.y, a2); a2 = __dp4a(e2.z, x.z, a2); a2 = __dp4a(e2.w, x.w, a2);
        a3 = __dp4a(e3.x, x.x, a3); a3 = __dp4a(e3.y, x.y, a3); a3 = __dp4a(e3.z, x.z, a3); a3 = __dp4a(e3.w, x.w, a3);
    }
    for (int o = 16; o > 0; o >>= 1) {
        a0 += __shfl_xor_sync(0xffffffffu, a0, o);
        a1 += __shfl_xor_sync(0xffffffffu, a1, o);
        a2 += __shfl_xor_sync(0xffffffffu, a2, o);
        a3 += __shfl_xor_sync(0xffffffffu, a3, o);
    }
    if (lane == 0) {
        float inv = xinv[0];
        out[r0 + 0] = (float)a0 * rs[r0 + 0] * inv;
        if (r0 + 1 < vocab) out[r0 + 1] = (float)a1 * rs[r0 + 1] * inv;
        if (r0 + 2 < vocab) out[r0 + 2] = (float)a2 * rs[r0 + 2] * inv;
        if (r0 + 3 < vocab) out[r0 + 3] = (float)a3 * rs[r0 + 3] * inv;
    }
}

// Fast on-GPU Top-K / Argmax reduction across 151,936 logits:
//Launched with 256 blocks of 256 threads; each block finds its top-1 (or top-2) candidate, writing 256 (val, idx) pairs (~2 KB total!)
extern "C" __global__ void k_topk_256(const float* __restrict__ logits, int tok_vocab, float* __restrict__ out_val, int* __restrict__ out_idx) {
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gstride = gridDim.x * blockDim.x;
    float best_v = -3.0e38f;
    int best_i = 0;
    for (int i = bid * blockDim.x + tid; i < tok_vocab; i += gstride) {
        float v = logits[i];
        if (v > best_v) { best_v = v; best_i = i; }
    }
    s_val[tid] = best_v;
    s_idx[tid] = best_i;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_val[bid] = s_val[0];
        out_idx[bid] = s_idx[0];
    }
}
