// Affan AI Engine 2  -  ternary (1.58-bit) transformer inference engine
// (c) MABA Studios / Muhammad Affan Bin Adnan
//
// Runs .affa v2 models (Qwen3-style decoder: GQA, QK-norm, RoPE, SwiGLU, tied embeddings).
//  * ternary weights stay 2-bit packed in RAM (memory-mapped, zero copy)
//  * int8 activations, integer SIMD (AVX2 / NEON / NEON+dotprod), no multiplications on weights
//  * batched prefill, KV-cache (bf16) with prefix reuse between chat turns
#include "affa_engine.h"
#if defined(_WIN32)
#include "affa_cuda.h"
#include "affa_d3d12.h"
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <random>
#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <cctype>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define AFFA_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif
#if defined(__aarch64__)
#define AFFA_ARM64 1
#include <arm_neon.h>
#if defined(__linux__)
#include <sys/auxv.h>
#endif
#endif

#if defined(__GNUC__) || defined(__clang__)
#define AFFA_TARGET_AVX2 __attribute__((target("avx2")))
#else
#define AFFA_TARGET_AVX2
#endif

namespace {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

inline void cpu_relax() {
#if defined(AFFA_X86)
    _mm_pause();
#elif defined(AFFA_ARM64)
    __asm__ __volatile__("yield");
#else
    std::this_thread::yield();
#endif
}

#if defined(__linux__) || defined(__ANDROID__)
#include <sched.h>
#include <sys/resource.h>
#endif

// ============================================================================
//  integer kernels
//  ternary weight layout: per 128 columns = 32 bytes; byte i holds columns 4i..4i+3 (bits 2k..2k+1)
//  code 0 -> 0, 1 -> +1, 2 -> -1.   Activations are stored "plane-major" per 128 block:
//  xp[128b + 32k + i] = x[128b + 4i + k]
// ============================================================================
typedef int (*dot_perm_fn)(const uint8_t*, const int8_t*, int);
typedef int (*dot_nat_fn)(const int8_t*, const int8_t*, int);

int dot_perm_scalar(const uint8_t* w, const int8_t* x, int cols) {
    static const int dec[4] = {0, 1, -1, 0};
    int acc = 0;
    for (int b = 0; b < cols / 128; b++) {
        const uint8_t* wb = w + 32 * b; const int8_t* xb = x + 128 * b;
        for (int i = 0; i < 32; i++) {
            unsigned v = wb[i];
            for (int k = 0; k < 4; k++) acc += dec[(v >> (2 * k)) & 3] * xb[32 * k + i];
        }
    }
    return acc;
}
int dot_nat_scalar(const int8_t* a, const int8_t* b, int n) {
    int acc = 0; for (int i = 0; i < n; i++) acc += (int)a[i] * (int)b[i]; return acc;
}

#if defined(AFFA_X86)
AFFA_TARGET_AVX2 inline int hsum256(__m256i acc) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}
AFFA_TARGET_AVX2 int dot_perm_avx2(const uint8_t* w, const int8_t* x, int cols) {
    const __m256i lut = _mm256_setr_epi8(0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m256i m3 = _mm256_set1_epi8(3), ones8 = _mm256_set1_epi8(1), ones16 = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    for (int b = 0; b < cols / 128; b++) {
        __m256i p = _mm256_loadu_si256((const __m256i*)(w + 32 * b));
        const __m256i* xb = (const __m256i*)(x + 128 * b);
        __m256i w0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(p, m3));
        __m256i w1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p, 2), m3));
        __m256i w2 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p, 4), m3));
        __m256i w3 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p, 6), m3));
        __m256i s = _mm256_maddubs_epi16(ones8, _mm256_sign_epi8(_mm256_loadu_si256(xb + 0), w0));
        s = _mm256_add_epi16(s, _mm256_maddubs_epi16(ones8, _mm256_sign_epi8(_mm256_loadu_si256(xb + 1), w1)));
        s = _mm256_add_epi16(s, _mm256_maddubs_epi16(ones8, _mm256_sign_epi8(_mm256_loadu_si256(xb + 2), w2)));
        s = _mm256_add_epi16(s, _mm256_maddubs_epi16(ones8, _mm256_sign_epi8(_mm256_loadu_si256(xb + 3), w3)));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(s, ones16));
    }
    return hsum256(acc);
}
// 4-row simultaneous AVX2 ternary GEMV kernel: loads x registers ONCE and computes 4 rows in parallel
AFFA_TARGET_AVX2 void dot_perm_4rows_avx2(const uint8_t* __restrict w0, size_t rb, const int8_t* __restrict x, int cols, int out4[4]) {
    const __m256i lut = _mm256_setr_epi8(0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m256i m3 = _mm256_set1_epi8(3), ones8 = _mm256_set1_epi8(1), ones16 = _mm256_set1_epi16(1);
    const uint8_t* w1 = w0 + rb;
    const uint8_t* w2 = w0 + 2 * rb;
    const uint8_t* w3 = w0 + 3 * rb;
    __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256(), a2 = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();
    const int nblk = cols / 128;
    for (int b = 0; b < nblk; b++) {
        const __m256i* xb = (const __m256i*)(x + 128 * b);
        __m256i x0 = _mm256_loadu_si256(xb + 0);
        __m256i x1 = _mm256_loadu_si256(xb + 1);
        __m256i x2 = _mm256_loadu_si256(xb + 2);
        __m256i x3 = _mm256_loadu_si256(xb + 3);
        const size_t off = 32 * (size_t)b;

        __m256i p0 = _mm256_loadu_si256((const __m256i*)(w0 + off));
        __m256i p1 = _mm256_loadu_si256((const __m256i*)(w1 + off));
        __m256i p2 = _mm256_loadu_si256((const __m256i*)(w2 + off));
        __m256i p3 = _mm256_loadu_si256((const __m256i*)(w3 + off));

        __m256i s0 = _mm256_maddubs_epi16(ones8, _mm256_sign_epi8(x0, _mm256_shuffle_epi8(lut, _mm256_and_si256(p0, m3))));
        __m256i s1 = _mm256_maddubs_epi16(ones8, _mm256_sign_epi8(x0, _mm256_shuffle_epi8(lut, _mm256_and_si256(p1, m3))));
        __m256i s2 = _mm256_maddubs_epi16(ones8, _mm256_sign_epi8(x0, _mm256_shuffle_epi8(lut, _mm256_and_si256(p2, m3))));
        __m256i s3 = _mm256_maddubs_epi16(ones8, _mm256_sign_epi8(x0, _mm256_shuffle_epi8(lut, _mm256_and_si256(p3, m3))));

        s0 = _mm256_add_epi16(s0, _mm256_maddubs_epi16(ones8, _mm256_sign_epi8(x1, _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p0, 2), m3)))));
        s1 = _mm256_add_epi16(s1, _mm256_maddubs_epi16(ones8, _mm256_sign_epi8(x1, _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p1, 2), m3)))));
        s2 = _mm256_add_epi16(s2, _mm256_maddubs_epi16(ones8, _mm256_sign_epi8(x1, _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p2, 2), m3)))));
        s3 = _mm256_add_epi16(s3, _mm256_maddubs_epi16(ones8, _mm256_sign_epi8(x1, _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p3, 2), m3)))));

        s0 = _mm256_add_epi16(s0, _mm256_maddubs_epi16(ones8, _mm256_sign_epi8(x2, _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p0, 4), m3)))));
        s1 = _mm256_add_epi16(s1, _mm256_maddubs_epi16(ones8, _mm256_sign_epi8(x2, _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p1, 4), m3)))));
        s2 = _mm256_add_epi16(s2, _mm256_maddubs_epi16(ones8, _mm256_sign_epi8(x2, _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p2, 4), m3)))));
        s3 = _mm256_add_epi16(s3, _mm256_maddubs_epi16(ones8, _mm256_sign_epi8(x2, _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p3, 4), m3)))));

        s0 = _mm256_add_epi16(s0, _mm256_maddubs_epi16(ones8, _mm256_sign_epi8(x3, _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p0, 6), m3)))));
        s1 = _mm256_add_epi16(s1, _mm256_maddubs_epi16(ones8, _mm256_sign_epi8(x3, _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p1, 6), m3)))));
        s2 = _mm256_add_epi16(s2, _mm256_maddubs_epi16(ones8, _mm256_sign_epi8(x3, _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p2, 6), m3)))));
        s3 = _mm256_add_epi16(s3, _mm256_maddubs_epi16(ones8, _mm256_sign_epi8(x3, _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(p3, 6), m3)))));

        a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(s0, ones16));
        a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(s1, ones16));
        a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(s2, ones16));
        a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(s3, ones16));
    }
    out4[0] = hsum256(a0);
    out4[1] = hsum256(a1);
    out4[2] = hsum256(a2);
    out4[3] = hsum256(a3);
}
// 4-row x 4-token batched AVX2 ternary GEMM kernel: decodes 4 weight rows ONCE and computes 16 dot products across 4 tokens
AFFA_TARGET_AVX2 void dot_perm_4x4_avx2(const uint8_t* __restrict w0, size_t rb, const int8_t* __restrict xp0, int xstride, int cols, int out4x4[4][4]) {
    const __m256i lut = _mm256_setr_epi8(0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m256i m3 = _mm256_set1_epi8(3), ones8 = _mm256_set1_epi8(1), ones16 = _mm256_set1_epi16(1);
    const uint8_t* w[4] = { w0, w0 + rb, w0 + 2 * rb, w0 + 3 * rb };
    const int8_t* xp[4] = { xp0, xp0 + xstride, xp0 + 2 * xstride, xp0 + 3 * xstride };
    __m256i a[4][4];
    for (int t = 0; t < 4; t++) for (int r = 0; r < 4; r++) a[t][r] = _mm256_setzero_si256();
    const int nblk = cols / 128;
    for (int b = 0; b < nblk; b++) {
        const size_t off = 32 * (size_t)b;
        __m256i p[4] = {
            _mm256_loadu_si256((const __m256i*)(w[0] + off)),
            _mm256_loadu_si256((const __m256i*)(w[1] + off)),
            _mm256_loadu_si256((const __m256i*)(w[2] + off)),
            _mm256_loadu_si256((const __m256i*)(w[3] + off))
        };
        __m256i s16[4][4];
        for (int t = 0; t < 4; t++) for (int r = 0; r < 4; r++) s16[t][r] = _mm256_setzero_si256();
        for (int k = 0; k < 4; k++) {
            __m256i wk[4];
            for (int r = 0; r < 4; r++) {
                __m256i pk = (k == 0) ? p[r] : (k == 1) ? _mm256_srli_epi16(p[r], 2) : (k == 2) ? _mm256_srli_epi16(p[r], 4) : _mm256_srli_epi16(p[r], 6);
                wk[r] = _mm256_shuffle_epi8(lut, _mm256_and_si256(pk, m3));
            }
            const size_t xo = 128 * (size_t)b + 32 * k;
            for (int t = 0; t < 4; t++) {
                __m256i xt = _mm256_loadu_si256((const __m256i*)(xp[t] + xo));
                for (int r = 0; r < 4; r++) {
                    s16[t][r] = _mm256_add_epi16(s16[t][r], _mm256_maddubs_epi16(ones8, _mm256_sign_epi8(xt, wk[r])));
                }
            }
        }
        for (int t = 0; t < 4; t++) for (int r = 0; r < 4; r++) {
            a[t][r] = _mm256_add_epi32(a[t][r], _mm256_madd_epi16(s16[t][r], ones16));
        }
    }
    for (int t = 0; t < 4; t++) for (int r = 0; r < 4; r++) {
        out4x4[t][r] = hsum256(a[t][r]);
    }
}
AFFA_TARGET_AVX2 void dot_nat_4rows_avx2(const int8_t* __restrict r0, int cols, const int8_t* __restrict x, int out4[4]) {
    const __m256i ones16 = _mm256_set1_epi16(1);
    const int8_t* r1 = r0 + cols;
    const int8_t* r2 = r0 + 2 * cols;
    const int8_t* r3 = r0 + 3 * cols;
    __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256(), a2 = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();
    for (int i = 0; i + 32 <= cols; i += 32) {
        __m256i xv = _mm256_loadu_si256((const __m256i*)(x + i));
        __m256i xa = _mm256_abs_epi8(xv);
        __m256i w0 = _mm256_loadu_si256((const __m256i*)(r0 + i));
        __m256i w1 = _mm256_loadu_si256((const __m256i*)(r1 + i));
        __m256i w2 = _mm256_loadu_si256((const __m256i*)(r2 + i));
        __m256i w3 = _mm256_loadu_si256((const __m256i*)(r3 + i));
        a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(xa, _mm256_sign_epi8(w0, xv)), ones16));
        a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_maddubs_epi16(xa, _mm256_sign_epi8(w1, xv)), ones16));
        a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(_mm256_maddubs_epi16(xa, _mm256_sign_epi8(w2, xv)), ones16));
        a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(_mm256_maddubs_epi16(xa, _mm256_sign_epi8(w3, xv)), ones16));
    }
    out4[0] = hsum256(a0);
    out4[1] = hsum256(a1);
    out4[2] = hsum256(a2);
    out4[3] = hsum256(a3);
}
AFFA_TARGET_AVX2 int dot_nat_avx2(const int8_t* a, const int8_t* b, int n) {
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i x = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i y = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i p = _mm256_maddubs_epi16(_mm256_abs_epi8(x), _mm256_sign_epi8(y, x));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, ones16));
    }
    int r = hsum256(acc);
    for (; i < n; i++) r += (int)a[i] * (int)b[i];
    return r;
}
bool cpu_has_avx2() {
    int info[4];
#if defined(_MSC_VER)
    __cpuid(info, 0); if (info[0] < 7) return false;
    __cpuid(info, 1);
    bool osx = (info[2] >> 27) & 1, avx = (info[2] >> 28) & 1;
    if (!(osx && avx)) return false;
    if ((_xgetbv(0) & 6) != 6) return false;
    __cpuidex(info, 7, 0);
    return (info[1] >> 5) & 1;
#else
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return false;
    if (!((c >> 27) & 1) || !((c >> 28) & 1)) return false;
    unsigned lo, hi; __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    if ((lo & 6) != 6) return false;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return false;
    return (b >> 5) & 1;
#endif
}
#endif  // AFFA_X86

#if defined(AFFA_ARM64)
int dot_perm_neon(const uint8_t* w, const int8_t* x, int cols) {
    static const int8_t lutb[16] = {0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const int8x16_t lut = vld1q_s8(lutb); const uint8x16_t m3 = vdupq_n_u8(3);
    int32x4_t acc = vdupq_n_s32(0);
    for (int b = 0; b < cols / 128; b++) for (int c = 0; c < 2; c++) {
        uint8x16_t p = vld1q_u8(w + 32 * b + 16 * c);
        const int8_t* xb = x + 128 * b + 16 * c;
        int8x16_t w0 = vqtbl1q_s8(lut, vandq_u8(p, m3));
        int8x16_t w1 = vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p, 2), m3));
        int8x16_t w2 = vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p, 4), m3));
        int8x16_t w3 = vqtbl1q_s8(lut, vshrq_n_u8(p, 6));
        int8x16_t x0 = vld1q_s8(xb), x1 = vld1q_s8(xb + 32), x2 = vld1q_s8(xb + 64), x3 = vld1q_s8(xb + 96);
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(x0), vget_low_s8(w0)));
        acc = vpadalq_s16(acc, vmull_high_s8(x0, w0));
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(x1), vget_low_s8(w1)));
        acc = vpadalq_s16(acc, vmull_high_s8(x1, w1));
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(x2), vget_low_s8(w2)));
        acc = vpadalq_s16(acc, vmull_high_s8(x2, w2));
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(x3), vget_low_s8(w3)));
        acc = vpadalq_s16(acc, vmull_high_s8(x3, w3));
    }
    return vaddvq_s32(acc);
}
int dot_nat_neon(const int8_t* a, const int8_t* b, int n) {
    int32x4_t acc = vdupq_n_s32(0); int i = 0;
    for (; i + 16 <= n; i += 16) {
        int8x16_t x = vld1q_s8(a + i), y = vld1q_s8(b + i);
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(x), vget_low_s8(y)));
        acc = vpadalq_s16(acc, vmull_high_s8(x, y));
    }
    int r = vaddvq_s32(acc);
    for (; i < n; i++) r += (int)a[i] * (int)b[i];
    return r;
}
#if defined(__clang__)
#define AFFA_TARGET_DOTPROD __attribute__((target("arch=armv8.2-a+dotprod")))
AFFA_TARGET_DOTPROD int dot_perm_dp(const uint8_t* __restrict w, const int8_t* __restrict x, int cols) {
    static const int8_t lutb[16] = {0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const int8x16_t lut = vld1q_s8(lutb); const uint8x16_t m3 = vdupq_n_u8(3);
    int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
    const int nblk = cols / 128;
    for (int b = 0; b < nblk; b++) {
        const uint8_t* wb = w + 32 * b;
        const int8_t* xb = x + 128 * b;
        uint8x16_t p0 = vld1q_u8(wb);
        uint8x16_t p1 = vld1q_u8(wb + 16);
        acc0 = vdotq_s32(acc0, vld1q_s8(xb),       vqtbl1q_s8(lut, vandq_u8(p0, m3)));
        acc1 = vdotq_s32(acc1, vld1q_s8(xb + 16),  vqtbl1q_s8(lut, vandq_u8(p1, m3)));
        acc0 = vdotq_s32(acc0, vld1q_s8(xb + 32),  vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p0, 2), m3)));
        acc1 = vdotq_s32(acc1, vld1q_s8(xb + 48),  vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p1, 2), m3)));
        acc0 = vdotq_s32(acc0, vld1q_s8(xb + 64),  vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p0, 4), m3)));
        acc1 = vdotq_s32(acc1, vld1q_s8(xb + 80),  vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p1, 4), m3)));
        acc0 = vdotq_s32(acc0, vld1q_s8(xb + 96),  vqtbl1q_s8(lut, vshrq_n_u8(p0, 6)));
        acc1 = vdotq_s32(acc1, vld1q_s8(xb + 112), vqtbl1q_s8(lut, vshrq_n_u8(p1, 6)));
    }
    return vaddvq_s32(vaddq_s32(acc0, acc1));
}
// 4-row simultaneous ternary GEMV kernel: loads x registers ONCE and computes 4 rows in parallel
AFFA_TARGET_DOTPROD void dot_perm_4rows_dp(const uint8_t* __restrict w0, size_t rb, const int8_t* __restrict x, int cols, int out4[4]) {
    static const int8_t lutb[16] = {0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const int8x16_t lut = vld1q_s8(lutb); const uint8x16_t m3 = vdupq_n_u8(3);
    const uint8_t* w1 = w0 + rb;
    const uint8_t* w2 = w0 + 2 * rb;
    const uint8_t* w3 = w0 + 3 * rb;
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0), a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
    const int nblk = cols / 128;
    for (int b = 0; b < nblk; b++) {
        const int8_t* xb = x + 128 * b;
        const size_t off = 32 * (size_t)b;
        for (int c = 0; c < 2; c++) {
            const size_t co = off + 16 * c;
            const int8_t* xbc = xb + 16 * c;
            int8x16_t x0 = vld1q_s8(xbc);
            int8x16_t x1 = vld1q_s8(xbc + 32);
            int8x16_t x2 = vld1q_s8(xbc + 64);
            int8x16_t x3 = vld1q_s8(xbc + 96);

            uint8x16_t p0 = vld1q_u8(w0 + co);
            uint8x16_t p1 = vld1q_u8(w1 + co);
            uint8x16_t p2 = vld1q_u8(w2 + co);
            uint8x16_t p3 = vld1q_u8(w3 + co);

            a0 = vdotq_s32(a0, x0, vqtbl1q_s8(lut, vandq_u8(p0, m3)));
            a1 = vdotq_s32(a1, x0, vqtbl1q_s8(lut, vandq_u8(p1, m3)));
            a2 = vdotq_s32(a2, x0, vqtbl1q_s8(lut, vandq_u8(p2, m3)));
            a3 = vdotq_s32(a3, x0, vqtbl1q_s8(lut, vandq_u8(p3, m3)));

            a0 = vdotq_s32(a0, x1, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p0, 2), m3)));
            a1 = vdotq_s32(a1, x1, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p1, 2), m3)));
            a2 = vdotq_s32(a2, x1, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p2, 2), m3)));
            a3 = vdotq_s32(a3, x1, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p3, 2), m3)));

            a0 = vdotq_s32(a0, x2, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p0, 4), m3)));
            a1 = vdotq_s32(a1, x2, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p1, 4), m3)));
            a2 = vdotq_s32(a2, x2, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p2, 4), m3)));
            a3 = vdotq_s32(a3, x2, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p3, 4), m3)));

            a0 = vdotq_s32(a0, x3, vqtbl1q_s8(lut, vshrq_n_u8(p0, 6)));
            a1 = vdotq_s32(a1, x3, vqtbl1q_s8(lut, vshrq_n_u8(p1, 6)));
            a2 = vdotq_s32(a2, x3, vqtbl1q_s8(lut, vshrq_n_u8(p2, 6)));
            a3 = vdotq_s32(a3, x3, vqtbl1q_s8(lut, vshrq_n_u8(p3, 6)));
        }
    }
    int32x4_t p01 = vpaddq_s32(a0, a1);
    int32x4_t p23 = vpaddq_s32(a2, a3);
    int32x4_t sum4 = vpaddq_s32(p01, p23);
    vst1q_s32(out4, sum4);
}
// 4-row x 4-token batched ternary GEMM kernel: decodes 4 weight rows ONCE and computes 16 dot products across 4 tokens
AFFA_TARGET_DOTPROD void dot_perm_4x4_dp(const uint8_t* __restrict w0, size_t rb, const int8_t* __restrict xp0, int xstride, int cols, int out4x4[4][4]) {
    static const int8_t lutb[16] = {0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const int8x16_t lut = vld1q_s8(lutb); const uint8x16_t m3 = vdupq_n_u8(3);
    const uint8_t* w1 = w0 + rb;
    const uint8_t* w2 = w0 + 2 * rb;
    const uint8_t* w3 = w0 + 3 * rb;
    const int8_t* xp1 = xp0 + xstride;
    const int8_t* xp2 = xp0 + 2 * xstride;
    const int8_t* xp3 = xp0 + 3 * xstride;
    int32x4_t a[4][4];
    for (int t = 0; t < 4; t++) for (int r = 0; r < 4; r++) a[t][r] = vdupq_n_s32(0);
    const int nblk = cols / 128;
    for (int b = 0; b < nblk; b++) {
        const size_t off = 32 * (size_t)b;
        for (int c = 0; c < 2; c++) {
            const size_t co = off + 16 * c;
            uint8x16_t p0 = vld1q_u8(w0 + co);
            uint8x16_t p1 = vld1q_u8(w1 + co);
            uint8x16_t p2 = vld1q_u8(w2 + co);
            uint8x16_t p3 = vld1q_u8(w3 + co);
            for (int k = 0; k < 4; k++) {
                int8x16_t wk0, wk1, wk2, wk3;
                if (k == 0) {
                    wk0 = vqtbl1q_s8(lut, vandq_u8(p0, m3)); wk1 = vqtbl1q_s8(lut, vandq_u8(p1, m3));
                    wk2 = vqtbl1q_s8(lut, vandq_u8(p2, m3)); wk3 = vqtbl1q_s8(lut, vandq_u8(p3, m3));
                } else if (k == 1) {
                    wk0 = vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p0, 2), m3)); wk1 = vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p1, 2), m3));
                    wk2 = vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p2, 2), m3)); wk3 = vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p3, 2), m3));
                } else if (k == 2) {
                    wk0 = vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p0, 4), m3)); wk1 = vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p1, 4), m3));
                    wk2 = vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p2, 4), m3)); wk3 = vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(p3, 4), m3));
                } else {
                    wk0 = vqtbl1q_s8(lut, vshrq_n_u8(p0, 6)); wk1 = vqtbl1q_s8(lut, vshrq_n_u8(p1, 6));
                    wk2 = vqtbl1q_s8(lut, vshrq_n_u8(p2, 6)); wk3 = vqtbl1q_s8(lut, vshrq_n_u8(p3, 6));
                }
                const size_t xo = 128 * (size_t)b + 32 * k + 16 * c;
                int8x16_t xt0 = vld1q_s8(xp0 + xo);
                int8x16_t xt1 = vld1q_s8(xp1 + xo);
                int8x16_t xt2 = vld1q_s8(xp2 + xo);
                int8x16_t xt3 = vld1q_s8(xp3 + xo);
                a[0][0] = vdotq_s32(a[0][0], xt0, wk0); a[0][1] = vdotq_s32(a[0][1], xt0, wk1);
                a[0][2] = vdotq_s32(a[0][2], xt0, wk2); a[0][3] = vdotq_s32(a[0][3], xt0, wk3);
                a[1][0] = vdotq_s32(a[1][0], xt1, wk0); a[1][1] = vdotq_s32(a[1][1], xt1, wk1);
                a[1][2] = vdotq_s32(a[1][2], xt1, wk2); a[1][3] = vdotq_s32(a[1][3], xt1, wk3);
                a[2][0] = vdotq_s32(a[2][0], xt2, wk0); a[2][1] = vdotq_s32(a[2][1], xt2, wk1);
                a[2][2] = vdotq_s32(a[2][2], xt2, wk2); a[2][3] = vdotq_s32(a[2][3], xt2, wk3);
                a[3][0] = vdotq_s32(a[3][0], xt3, wk0); a[3][1] = vdotq_s32(a[3][1], xt3, wk1);
                a[3][2] = vdotq_s32(a[3][2], xt3, wk2); a[3][3] = vdotq_s32(a[3][3], xt3, wk3);
            }
        }
    }
    for (int t = 0; t < 4; t++) {
        int32x4_t p01 = vpaddq_s32(a[t][0], a[t][1]);
        int32x4_t p23 = vpaddq_s32(a[t][2], a[t][3]);
        vst1q_s32(out4x4[t], vpaddq_s32(p01, p23));
    }
}
// 4-row simultaneous int8 GEMV kernel for fast vocabulary logits
AFFA_TARGET_DOTPROD void dot_nat_4rows_dp(const int8_t* __restrict r0, int cols, const int8_t* __restrict x, int out4[4]) {
    const int8_t* r1 = r0 + cols;
    const int8_t* r2 = r0 + 2 * cols;
    const int8_t* r3 = r0 + 3 * cols;
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0), a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
    int i = 0;
    for (; i + 64 <= cols; i += 64) {
        int8x16_t x0 = vld1q_s8(x + i), x1 = vld1q_s8(x + i + 16), x2 = vld1q_s8(x + i + 32), x3 = vld1q_s8(x + i + 48);
        a0 = vdotq_s32(a0, vld1q_s8(r0 + i), x0); a0 = vdotq_s32(a0, vld1q_s8(r0 + i + 16), x1);
        a0 = vdotq_s32(a0, vld1q_s8(r0 + i + 32), x2); a0 = vdotq_s32(a0, vld1q_s8(r0 + i + 48), x3);

        a1 = vdotq_s32(a1, vld1q_s8(r1 + i), x0); a1 = vdotq_s32(a1, vld1q_s8(r1 + i + 16), x1);
        a1 = vdotq_s32(a1, vld1q_s8(r1 + i + 32), x2); a1 = vdotq_s32(a1, vld1q_s8(r1 + i + 48), x3);

        a2 = vdotq_s32(a2, vld1q_s8(r2 + i), x0); a2 = vdotq_s32(a2, vld1q_s8(r2 + i + 16), x1);
        a2 = vdotq_s32(a2, vld1q_s8(r2 + i + 32), x2); a2 = vdotq_s32(a2, vld1q_s8(r2 + i + 48), x3);

        a3 = vdotq_s32(a3, vld1q_s8(r3 + i), x0); a3 = vdotq_s32(a3, vld1q_s8(r3 + i + 16), x1);
        a3 = vdotq_s32(a3, vld1q_s8(r3 + i + 32), x2); a3 = vdotq_s32(a3, vld1q_s8(r3 + i + 48), x3);
    }
    for (; i + 16 <= cols; i += 16) {
        int8x16_t xv = vld1q_s8(x + i);
        a0 = vdotq_s32(a0, vld1q_s8(r0 + i), xv);
        a1 = vdotq_s32(a1, vld1q_s8(r1 + i), xv);
        a2 = vdotq_s32(a2, vld1q_s8(r2 + i), xv);
        a3 = vdotq_s32(a3, vld1q_s8(r3 + i), xv);
    }
    int32x4_t p01 = vpaddq_s32(a0, a1);
    int32x4_t p23 = vpaddq_s32(a2, a3);
    int32x4_t sum4 = vpaddq_s32(p01, p23);
    vst1q_s32(out4, sum4);
}
AFFA_TARGET_DOTPROD int dot_nat_dp(const int8_t* __restrict a, const int8_t* __restrict b, int n) {
    int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0); int i = 0;
    for (; i + 32 <= n; i += 32) {
        acc0 = vdotq_s32(acc0, vld1q_s8(a + i), vld1q_s8(b + i));
        acc1 = vdotq_s32(acc1, vld1q_s8(a + i + 16), vld1q_s8(b + i + 16));
    }
    for (; i + 16 <= n; i += 16) acc0 = vdotq_s32(acc0, vld1q_s8(a + i), vld1q_s8(b + i));
    int r = vaddvq_s32(vaddq_s32(acc0, acc1));
    for (; i < n; i++) r += (int)a[i] * (int)b[i];
    return r;
}
#endif
#endif  // AFFA_ARM64

struct Kernels { dot_perm_fn perm; dot_nat_fn nat; const char* name; bool has_dp; };
Kernels select_kernels() {
    Kernels k{dot_perm_scalar, dot_nat_scalar, "scalar", false};
#if defined(AFFA_X86)
    if (cpu_has_avx2()) k = {dot_perm_avx2, dot_nat_avx2, "avx2", false};
#endif
#if defined(AFFA_ARM64)
    k = {dot_perm_neon, dot_nat_neon, "neon", false};
#if defined(__clang__) && defined(__linux__)
    if (getauxval(AT_HWCAP) & (1UL << 20)) k = {dot_perm_dp, dot_nat_dp, "neon+dotprod", true};
#endif
#endif
    return k;
}
const Kernels& K() { static Kernels k = select_kernels(); return k; }

// ---- activation quantisation: per-token absmax, 127 levels (NEON accelerated on ARM64) ----
inline float quant_perm(const float* __restrict x, int cols, int8_t* __restrict xp) {
#if defined(AFFA_ARM64)
    float32x4_t vmax0 = vdupq_n_f32(0.f), vmax1 = vdupq_n_f32(0.f);
    for (int i = 0; i < cols; i += 8) {
        vmax0 = vmaxq_f32(vmax0, vabsq_f32(vld1q_f32(x + i)));
        vmax1 = vmaxq_f32(vmax1, vabsq_f32(vld1q_f32(x + i + 4)));
    }
    float amax = std::max(vmaxvq_f32(vmaxq_f32(vmax0, vmax1)), 1e-5f);
    const float s = 127.0f / amax;
    const float32x4_t vs = vdupq_n_f32(s);
    for (int b = 0; b < cols / 128; b++) {
        const float* xb = x + 128 * b;
        int8_t* dst = xp + 128 * b;
        for (int g = 0; g < 2; g++) {
            const float* xg = xb + 64 * g;
            // deinterleave 64 floats into 4 planes of 16 bytes
            float32x4x4_t f0 = vld4q_f32(xg + 0);
            float32x4x4_t f1 = vld4q_f32(xg + 16);
            float32x4x4_t f2 = vld4q_f32(xg + 32);
            float32x4x4_t f3 = vld4q_f32(xg + 48);
            for (int k = 0; k < 4; k++) {
                int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(f0.val[k], vs));
                int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(f1.val[k], vs));
                int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(f2.val[k], vs));
                int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(f3.val[k], vs));
                int16x8_t s01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
                int16x8_t s23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
                int8x16_t q8 = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
                vst1q_s8(dst + 32 * k + 16 * g, q8);
            }
        }
    }
    return 1.0f / s;
#elif defined(AFFA_X86)
    __m256 vmax = _mm256_setzero_ps();
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    for (int i = 0; i < cols; i += 8) {
        vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(x + i), sign_mask));
    }
    __m128 m128 = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
    m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
    m128 = _mm_max_ss(m128, _mm_shuffle_ps(m128, m128, 1));
    float amax = std::max(_mm_cvtss_f32(m128), 1e-5f);
    const float s = 127.0f / amax;
    const __m256 vs = _mm256_set1_ps(s);
    alignas(32) int8_t qbuf[128];
    const __m128i pshuf = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const __m256i pshuf256 = _mm256_broadcastsi128_si256(pshuf);
    for (int b = 0; b < cols / 128; b++) {
        const float* xb = x + 128 * b;
        for (int g = 0; g < 4; g++) {
            const float* xg = xb + 32 * g;
            __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xg + 0), vs));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xg + 8), vs));
            __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xg + 16), vs));
            __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xg + 24), vs));
            __m256i s01 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i0, i1), 0xD8);
            __m256i s23 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i2, i3), 0xD8);
            __m256i b8 = _mm256_permute4x64_epi64(_mm256_packs_epi16(s01, s23), 0xD8);
            b8 = _mm256_shuffle_epi8(b8, pshuf256);
            _mm256_store_si256((__m256i*)(qbuf + 32 * g), b8);
        }
        int8_t* dst = xp + 128 * b;
        for (int k = 0; k < 4; k++) {
            for (int g = 0; g < 4; g++) {
                std::memcpy(dst + 32 * k + 8 * g + 0, qbuf + 32 * g + 4 * k, 4);
                std::memcpy(dst + 32 * k + 8 * g + 4, qbuf + 32 * g + 16 + 4 * k, 4);
            }
        }
    }
    return 1.0f / s;
#else
    float amax = 0.f;
    for (int i = 0; i < cols; i++) amax = std::max(amax, std::fabs(x[i]));
    amax = std::max(amax, 1e-5f);
    const float s = 127.0f / amax;
    for (int b = 0; b < cols / 128; b++)
        for (int k = 0; k < 4; k++)
            for (int i = 0; i < 32; i++) {
                long q = lrintf(x[128 * b + 4 * i + k] * s);
                q = q > 127 ? 127 : (q < -127 ? -127 : q);
                xp[128 * b + 32 * k + i] = (int8_t)q;
            }
    return 1.0f / s;
#endif
}
inline float quant_nat(const float* __restrict x, int cols, int8_t* __restrict xq) {
#if defined(AFFA_ARM64)
    float32x4_t vmax0 = vdupq_n_f32(0.f), vmax1 = vdupq_n_f32(0.f);
    for (int i = 0; i < cols; i += 8) {
        vmax0 = vmaxq_f32(vmax0, vabsq_f32(vld1q_f32(x + i)));
        vmax1 = vmaxq_f32(vmax1, vabsq_f32(vld1q_f32(x + i + 4)));
    }
    float amax = std::max(vmaxvq_f32(vmaxq_f32(vmax0, vmax1)), 1e-5f);
    const float s = 127.0f / amax;
    const float32x4_t vs = vdupq_n_f32(s);
    for (int i = 0; i < cols; i += 16) {
        int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i), vs));
        int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i + 4), vs));
        int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i + 8), vs));
        int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i + 12), vs));
        int16x8_t s01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
        int16x8_t s23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
        vst1q_s8(xq + i, vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23)));
    }
    return 1.0f / s;
#elif defined(AFFA_X86)
    __m256 vmax = _mm256_setzero_ps();
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    for (int i = 0; i < cols; i += 8) {
        vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(x + i), sign_mask));
    }
    __m128 m128 = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
    m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
    m128 = _mm_max_ss(m128, _mm_shuffle_ps(m128, m128, 1));
    float amax = std::max(_mm_cvtss_f32(m128), 1e-5f);
    const float s = 127.0f / amax;
    const __m256 vs = _mm256_set1_ps(s);
    for (int i = 0; i < cols; i += 32) {
        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + i + 0), vs));
        __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + i + 8), vs));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + i + 16), vs));
        __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + i + 24), vs));
        __m256i s01 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i0, i1), 0xD8);
        __m256i s23 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i2, i3), 0xD8);
        __m256i b8 = _mm256_permute4x64_epi64(_mm256_packs_epi16(s01, s23), 0xD8);
        _mm256_storeu_si256((__m256i*)(xq + i), b8);
    }
    return 1.0f / s;
#else
    float amax = 0.f;
    for (int i = 0; i < cols; i++) amax = std::max(amax, std::fabs(x[i]));
    amax = std::max(amax, 1e-5f);
    const float s = 127.0f / amax;
    for (int i = 0; i < cols; i++) {
        long q = lrintf(x[i] * s); q = q > 127 ? 127 : (q < -127 ? -127 : q); xq[i] = (int8_t)q;
    }
    return 1.0f / s;
#endif
}

inline float bf2f(uint16_t u) { uint32_t b = (uint32_t)u << 16; float f; std::memcpy(&f, &b, 4); return f; }
inline uint16_t f2bf(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

// ============================================================================
//  Dynamic CPU Topology Detector & Thread Pool (big.LITTLE frequency-aware)
// ============================================================================
struct CpuTopology {
    std::vector<int> sorted_cores; // sorted descending by max frequency (big cores first)
    int optimal_threads = 4;
};

CpuTopology detect_cpu_topology() {
    CpuTopology topo;
    int n_hw = (int)std::thread::hardware_concurrency();
    if (n_hw <= 0) n_hw = 4;
#if defined(__linux__) || defined(__ANDROID__)
    std::vector<std::pair<long long, int>> freq_core;
    for (int c = 0; c < n_hw; c++) {
        long long freq = 0;
        char path[128];
        std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", c);
        FILE* f = std::fopen(path, "r");
        if (f) {
            if (std::fscanf(f, "%lld", &freq) != 1) freq = 0;
            std::fclose(f);
        }
        freq_core.push_back({freq, c});
    }
    // Sort descending by frequency (so Cortex-A75 / Prime / Big cores are first, efficient cores last)
    std::sort(freq_core.begin(), freq_core.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second > b.second;
    });
    for (auto& fc : freq_core) topo.sorted_cores.push_back(fc.second);
    // Reserve 1 to 1.5 efficient cores for Android OS / UI rendering, take all remaining big + efficient cores
    if (n_hw >= 8) topo.optimal_threads = n_hw - 1;      // e.g. 7 threads on 8-core (2 big + 5 little, leaving 1 little for OS)
    else if (n_hw >= 4) topo.optimal_threads = n_hw - 1;
    else topo.optimal_threads = std::max(1, n_hw);
#else
    for (int c = 0; c < n_hw; c += 2) topo.sorted_cores.push_back(c);
    for (int c = 1; c < n_hw; c += 2) topo.sorted_cores.push_back(c);
    // On x86 SMT/HyperThreaded CPUs (e.g. 6C/12T), physical core count (n_hw / 2) avoids L1/L2 cache thrashing & spin-lock contention
    topo.optimal_threads = (n_hw >= 8) ? std::max(4, n_hw / 2) : std::max(2, n_hw - 1);
#endif
    return topo;
}

void pin_thread_to_core(int rank, const std::vector<int>& sorted_cores) {
#if defined(__linux__) || defined(__ANDROID__)
    setpriority(PRIO_PROCESS, 0, -10);
    if (!sorted_cores.empty()) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (rank == 0) {
            // Pin main worker to the fastest big cores
            CPU_SET(sorted_cores[0], &cpuset);
            if (sorted_cores.size() > 1) CPU_SET(sorted_cores[1], &cpuset);
        } else {
            // Pin each worker to its assigned core + allow spillover across active pool cores (excluding the reserved OS core)
            int active_n = std::max(1, (int)sorted_cores.size() - 1);
            int target = sorted_cores[rank % active_n];
            CPU_SET(target, &cpuset);
            for (int i = 0; i < active_n; i++) CPU_SET(sorted_cores[i], &cpuset);
        }
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    }
#elif defined(_WIN32)
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    if (!sorted_cores.empty()) {
        int active_n = std::max(1, (int)sorted_cores.size() - 1);
        DWORD_PTR mask = 0;
        for (int i = 0; i < active_n && i < 64; i++) {
            mask |= (DWORD_PTR(1) << sorted_cores[i]);
        }
        if (mask != 0) SetThreadAffinityMask(GetCurrentThread(), mask);
    }
    (void)rank;
#else
    (void)rank; (void)sorted_cores;
#endif
}

class Pool {
public:
    Pool() { topo_ = detect_cpu_topology(); }
    ~Pool() { stop(); }
    int optimal_threads() const { return topo_.optimal_threads; }
    void start(int n) {
        stop();
        nthreads_ = (n > 0) ? n : topo_.optimal_threads;
        quit_ = false; gen_ = 0; sleepers_ = 0;
        for (int t = 1; t < nthreads_; t++) workers_.emplace_back([this, t] { loop(t); });
    }
    int size() const { return nthreads_; }
    void run(int nblocks, const std::function<void(int, int)>& f) {
        if (nthreads_ == 1 || nblocks <= 1) { for (int b = 0; b < nblocks; b++) f(b, 0); return; }
        if (!pinned_main_) { pin_thread_to_core(0, topo_.sorted_cores); pinned_main_ = true; }
        job_ = &f; total_ = nblocks; next_.store(0, std::memory_order_relaxed); done_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
        if (sleepers_.load(std::memory_order_relaxed) > 0) { std::lock_guard<std::mutex> lk(m_); cv_.notify_all(); }
        work(0);
        const int nw = nthreads_ - 1;
        while (done_.load(std::memory_order_acquire) < nw) cpu_relax();
    }
private:
    void work(int tid) {
        for (;;) {
            int b = next_.fetch_add(1, std::memory_order_relaxed);
            if (b >= total_) break;
            (*job_)(b, tid);
        }
    }
    void loop(int tid) {
        pin_thread_to_core(tid, topo_.sorted_cores);
        uint64_t my = gen_.load(std::memory_order_relaxed);
        while (!quit_) {
            int spin = 0;
            while (gen_.load(std::memory_order_acquire) == my && !quit_) {
                if (++spin < 200000) cpu_relax();
                else {
                    sleepers_++;
                    std::unique_lock<std::mutex> lk(m_);
                    cv_.wait_for(lk, std::chrono::milliseconds(10), [&] { return gen_.load() != my || quit_; });
                    sleepers_--;
                }
            }
            if (quit_) break;
            my = gen_.load(std::memory_order_relaxed);
            work(tid);
            done_.fetch_add(1, std::memory_order_release);
        }
    }
    void stop() {
        quit_ = true;
        { std::lock_guard<std::mutex> lk(m_); cv_.notify_all(); }
        for (auto& t : workers_) t.join();
        workers_.clear();
    }
    CpuTopology topo_;
    bool pinned_main_ = false;
    int nthreads_ = 1;
    std::vector<std::thread> workers_;
    std::atomic<uint64_t> gen_{0};
    std::atomic<int> next_{0}, done_{0}, sleepers_{0};
    int total_ = 0;
    const std::function<void(int, int)>* job_ = nullptr;
    std::atomic<bool> quit_{false};
    std::mutex m_; std::condition_variable cv_;
};

// ============================================================================
//  memory-mapped file
// ============================================================================
struct MapFile {
    const uint8_t* p = nullptr; size_t size = 0; bool heap = false;
#if defined(_WIN32)
    HANDLE hf = INVALID_HANDLE_VALUE, hm = nullptr;
#else
    int fd = -1;
#endif
    MapFile() {}
    MapFile(const MapFile&) = delete;
    ~MapFile() { close_(); }
    void close_() {
#if defined(_WIN32)
        if (p && !heap) UnmapViewOfFile(p);
        if (hm) CloseHandle(hm);
        if (hf != INVALID_HANDLE_VALUE) CloseHandle(hf);
        hm = nullptr; hf = INVALID_HANDLE_VALUE;
#else
        if (p && !heap) munmap((void*)p, size);
        if (fd >= 0) ::close(fd);
        fd = -1;
#endif
        if (p && heap) free((void*)p);
        p = nullptr; size = 0;
    }
#if defined(_WIN32)
    bool open_path(const char* utf8, std::string& err) {
        int n = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, nullptr, 0);
        std::vector<wchar_t> w(n + 1); MultiByteToWideChar(CP_UTF8, 0, utf8, -1, w.data(), n);
        hf = CreateFileW(w.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hf == INVALID_HANDLE_VALUE) { err = "cannot open file"; return false; }
        LARGE_INTEGER sz; GetFileSizeEx(hf, &sz); size = (size_t)sz.QuadPart;
        hm = CreateFileMappingW(hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hm) { err = "CreateFileMapping failed"; return false; }
        p = (const uint8_t*)MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
        if (!p) { err = "MapViewOfFile failed"; return false; }
        return true;
    }
#else
    bool open_path(const char* path, std::string& err) {
        int f = ::open(path, O_RDONLY);
        if (f < 0) { err = "cannot open file"; return false; }
        return open_fd(f, err);
    }
    bool open_fd(int f, std::string& err) {
        fd = f;
        struct stat st; if (fstat(fd, &st) != 0 || st.st_size <= 0) { err = "cannot stat file"; return false; }
        size = (size_t)st.st_size;
        void* m = mmap(nullptr, size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
        if (m == MAP_FAILED) m = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m != MAP_FAILED) {
            madvise(m, size, MADV_WILLNEED);
            p = (const uint8_t*)m;
            return true;
        }
        // fallback: read into heap (some content providers are not mappable)
        uint8_t* buf = (uint8_t*)malloc(size);
        if (!buf) { err = "out of memory"; return false; }
        lseek(fd, 0, SEEK_SET); size_t got = 0;
        while (got < size) { ssize_t r = read(fd, buf + got, size - got); if (r <= 0) break; got += (size_t)r; }
        if (got != size) { free(buf); err = "read failed"; return false; }
        p = buf; heap = true; return true;
    }
#endif
};

// ============================================================================
//  tokenizer (byte-level BPE, Qwen-style pre-tokenizer)
// ============================================================================
inline uint32_t cp_at(const std::string& s, size_t i, size_t& len) {
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) { len = 1; return c; }
    int n = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 1;
    if (n == 1 || i + n > s.size()) { len = 1; return 0xFFFD; }
    uint32_t cp = c & (0xFF >> (n + 1));
    for (int k = 1; k < n; k++) {
        unsigned char d = (unsigned char)s[i + k];
        if ((d & 0xC0) != 0x80) { len = 1; return 0xFFFD; }
        cp = (cp << 6) | (d & 0x3F);
    }
    len = n; return cp;
}
inline bool is_space_cp(uint32_t c) {
    return c == ' ' || (c >= 9 && c <= 13) || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
inline bool is_nl_cp(uint32_t c) { return c == '\n' || c == '\r'; }
inline bool is_digit_cp(uint32_t c) { return c >= '0' && c <= '9'; }
inline bool is_letter_cp(uint32_t c) {
    if (c < 128) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    if (is_space_cp(c)) return false;
    if ((c >= 0xA1 && c <= 0xBF && c != 0xAA && c != 0xB5 && c != 0xBA) || c == 0xD7 || c == 0xF7) return false;
    if ((c >= 0x2010 && c <= 0x2027) || (c >= 0x2030 && c <= 0x205E) || (c >= 0x20A0 && c <= 0x20CF) ||
        (c >= 0x2190 && c <= 0x2BFF) || (c >= 0x3001 && c <= 0x303F) || (c >= 0xFF01 && c <= 0xFF0F) ||
        (c >= 0xFF1A && c <= 0xFF20) || (c >= 0xFE00 && c <= 0xFE0F) || (c >= 0x1F000 && c <= 0x1FAFF)) return false;
    return true;
}

struct Tokenizer {
    std::vector<std::string> tok;       // id -> bytes
    std::vector<uint8_t> added;         // id -> is added/special token
    std::unordered_map<uint64_t, std::pair<int, int>> merges;   // (a<<32|b) -> (rank, result)
    int byte_id[256];
    bool ok = false;

    bool load(const uint8_t* p, size_t n, std::string& err) {
        size_t o = 0;
        auto need = [&](size_t k) { return o + k <= n; };
        if (!need(4)) { err = "tokenizer blob truncated"; return false; }
        uint32_t cnt; std::memcpy(&cnt, p + o, 4); o += 4;
        tok.resize(cnt); added.assign(cnt, 0);
        for (int i = 0; i < 256; i++) byte_id[i] = -1;
        for (uint32_t i = 0; i < cnt; i++) {
            if (!need(3)) { err = "tokenizer blob truncated"; return false; }
            uint8_t fl = p[o]; uint16_t len; std::memcpy(&len, p + o + 1, 2); o += 3;
            if (!need(len)) { err = "tokenizer blob truncated"; return false; }
            tok[i].assign((const char*)p + o, len); o += len; added[i] = fl;
            if (!fl && len == 1) byte_id[(uint8_t)tok[i][0]] = (int)i;
        }
        if (!need(4)) { err = "tokenizer merges missing"; return false; }
        uint32_t nm; std::memcpy(&nm, p + o, 4); o += 4;
        if (!need((size_t)nm * 12)) { err = "tokenizer merges truncated"; return false; }
        merges.reserve(nm * 2);
        for (uint32_t i = 0; i < nm; i++) {
            uint32_t t[3]; std::memcpy(t, p + o + (size_t)i * 12, 12);
            merges.emplace(((uint64_t)t[0] << 32) | t[1], std::make_pair((int)i, (int)t[2]));
        }
        for (int i = 0; i < 256; i++) if (byte_id[i] < 0) { err = "tokenizer lacks byte tokens"; return false; }
        ok = true; return true;
    }
    void bpe(const char* s, size_t len, std::vector<int>& out) const {
        std::vector<int> ids(len);
        for (size_t i = 0; i < len; i++) ids[i] = byte_id[(uint8_t)s[i]];
        while (ids.size() > 1) {
            int best = INT32_MAX; size_t bi = 0; int res = 0;
            for (size_t i = 0; i + 1 < ids.size(); i++) {
                auto it = merges.find(((uint64_t)(uint32_t)ids[i] << 32) | (uint32_t)ids[i + 1]);
                if (it != merges.end() && it->second.first < best) { best = it->second.first; bi = i; res = it->second.second; }
            }
            if (best == INT32_MAX) break;
            ids[bi] = res; ids.erase(ids.begin() + bi + 1);
        }
        out.insert(out.end(), ids.begin(), ids.end());
    }
    // Qwen2/3 pre-tokenizer regex, hand-written
    void encode(const std::string& s, std::vector<int>& out) const {
        size_t i = 0, n = s.size();
        auto emit = [&](size_t a, size_t b) { bpe(s.data() + a, b - a, out); };
        while (i < n) {
            size_t l; uint32_t c = cp_at(s, i, l); size_t st = i;
            // 1. contractions
            if (c == '\'' && i + 1 < n) {
                char d = (char)std::tolower((unsigned char)s[i + 1]);
                if (d == 's' || d == 't' || d == 'm' || d == 'd') { emit(st, i + 2); i += 2; continue; }
                if (i + 2 < n) {
                    char e = (char)std::tolower((unsigned char)s[i + 2]);
                    if ((d == 'r' && e == 'e') || (d == 'v' && e == 'e') || (d == 'l' && e == 'l')) { emit(st, i + 3); i += 3; continue; }
                }
            }
            // 2. [^\r\n\p{L}\p{N}]?\p{L}+
            {
                size_t p = i; bool ok2 = false;
                if (is_letter_cp(c)) ok2 = true;
                else if (!is_nl_cp(c) && !is_digit_cp(c)) {
                    size_t k = i + l;
                    if (k < n) { size_t l2; uint32_t c2 = cp_at(s, k, l2); if (is_letter_cp(c2)) { ok2 = true; p = k; } }
                }
                if (ok2) {
                    while (p < n) { size_t l3; uint32_t c3 = cp_at(s, p, l3); if (!is_letter_cp(c3)) break; p += l3; }
                    emit(st, p); i = p; continue;
                }
            }
            // 3. single digit
            if (is_digit_cp(c)) { emit(st, i + l); i += l; continue; }
            // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
            {
                size_t p = i; if (c == ' ') p += 1;
                size_t q = p; int cnt = 0;
                while (q < n) { size_t l4; uint32_t c4 = cp_at(s, q, l4); if (is_space_cp(c4) || is_letter_cp(c4) || is_digit_cp(c4)) break; q += l4; cnt++; }
                if (cnt > 0) { while (q < n && (s[q] == '\n' || s[q] == '\r')) q++; emit(st, q); i = q; continue; }
            }
            // whitespace run
            if (is_space_cp(c)) {
                size_t e = i; size_t last_nl_end = 0; size_t prev_start = i; int ncp = 0;
                while (e < n) {
                    size_t l5; uint32_t c5 = cp_at(s, e, l5);
                    if (!is_space_cp(c5)) break;
                    prev_start = e; e += l5; ncp++;
                    if (is_nl_cp(c5)) last_nl_end = e;
                }
                if (last_nl_end) { emit(st, last_nl_end); i = last_nl_end; continue; }       // \s*[\r\n]+
                if (e == n) { emit(st, e); i = e; continue; }                                  // \s+(?!\S) at end
                if (ncp > 1) { emit(st, prev_start); i = prev_start; continue; }               // leave last space for next token
                emit(st, e); i = e; continue;                                                  // \s+
            }
            emit(st, i + l); i += l;
        }
    }
};

// ============================================================================
//  model
// ============================================================================
struct Tensor { int dtype = 0; const uint8_t* data = nullptr; int rows = 0, cols = 0; float scale = 1.f; const float* rscale = nullptr; };
struct Layer { const float *ln1, *ln2, *qn, *kn; Tensor q, k, v, o, g, u, d; };

std::string json_escape(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '"') o += "\\\""; else if (c == '\\') o += "\\\\"; else if (c == '\n') o += "\\n";
        else if (c < 0x20) o += ' '; else o += (char)c;
    }
    return o;
}

}  // namespace

struct AffaModel {
    MapFile file;
    std::unordered_map<std::string, std::string> kv;
    std::unordered_map<std::string, Tensor> tensors;
    Tokenizer tk;
    // config
    int H, I, L, nh, nkv, hd, vocab, tok_vocab, max_pos, n_ctx;
    int qdim, kvdim;
    float theta, eps;
    int id_im_start, id_im_end, id_eot, id_think_open, id_think_close;
    std::string name, owner, studio, system_default, info;
    float def_temp, def_top_p, def_rep; int def_top_k;
    Tensor embed; const float* final_norm; std::vector<Layer> layers;
    // runtime
    Pool pool; int threads = 4;
    uint16_t* kc = nullptr; uint16_t* vc = nullptr;
    std::vector<int> cache_ids;
    std::vector<float> rope_cos, rope_sin;
    static constexpr int CH = 64;
    std::vector<float> x, h, q, k, v, ao, g, u, tmp, logits;
    std::vector<int8_t> xq, xq_nat; std::vector<float> xinv;
    std::vector<std::vector<float>> scores;
    std::mutex gen_mutex; std::atomic<bool> stop_flag{false};
    std::mt19937 rng;
    void* cuda_handle = nullptr;
    void* d3d12_handle = nullptr;
    std::string cuda_dev_name;
    std::string d3d12_dev_name;
    int active_backend = 0; // 0 = CPU AVX2/NEON, 1 = CUDA, 2 = DirectML / D3D12
    std::string backend_mode = "auto";
    std::string gpu_label;

    ~AffaModel() {
#if defined(_WIN32)
        if (cuda_handle) affa_cuda_free(cuda_handle);
        if (d3d12_handle) affa_d3d12_free(d3d12_handle);
#endif
        free(kc); free(vc);
    }

    void update_info_json() {
        long long params = 0;
        for (auto& kvp : tensors) params += (long long)kvp.second.rows * kvp.second.cols;
        std::string simd_str = K().name;
        gpu_label.clear();
        if (active_backend == 1 && cuda_handle) {
            gpu_label = "CUDA | GPU";
            simd_str = "CUDA | GPU";
        } else if (active_backend == 2 && d3d12_handle) {
            gpu_label = "DirectML | GPU";
            simd_str = "DirectML | GPU";
        }
        char buf[2048];
        std::snprintf(buf, sizeof buf,
            "{\"name\":\"%s\",\"owner\":\"%s\",\"studio\":\"%s\",\"arch\":\"%s\",\"quant\":\"%s\",\"layers\":%d,\"hidden\":%d,"
            "\"heads\":%d,\"kv_heads\":%d,\"vocab\":%d,\"ctx\":%d,\"max_ctx\":%d,\"file_mb\":%.1f,\"params_m\":%.0f,\"threads\":%d,\"simd\":\"%s\","
            "\"backend_mode\":\"%s\",\"active_backend\":\"%s\",\"has_cuda\":%s,\"has_dml\":%s}",
            json_escape(name).c_str(), json_escape(owner).c_str(), json_escape(studio).c_str(),
            json_escape(kv["arch"]).c_str(), json_escape(kv["quant"]).c_str(), L, H, nh, nkv, vocab, n_ctx, max_pos,
            (double)file.size / 1048576.0, (double)params / 1e6, threads,
            json_escape(simd_str).c_str(),
            json_escape(backend_mode).c_str(),
            active_backend == 1 ? "cuda" : (active_backend == 2 ? "dml" : "cpu"),
            cuda_handle ? "true" : "false",
            (d3d12_handle || !d3d12_dev_name.empty()) ? "true" : "false");
        info = buf;
    }

#if defined(_WIN32)
    bool ensure_d3d12_loaded() {
        if (d3d12_handle) return true;
        if (d3d12_dev_name.empty()) return false;
        AffaCudaCfg c_cfg = { H, I, L, nh, nkv, hd, vocab, n_ctx, eps };
        std::vector<AffaCudaLayer> c_layers(L);
        for (int l = 0; l < L; l++) {
            c_layers[l].ln1 = layers[l].ln1;
            c_layers[l].ln2 = layers[l].ln2;
            c_layers[l].qn = layers[l].qn;
            c_layers[l].kn = layers[l].kn;
            const Tensor* w_arr[7] = { &layers[l].q, &layers[l].k, &layers[l].v, &layers[l].o, &layers[l].g, &layers[l].u, &layers[l].d };
            for (int i = 0; i < 7; i++) {
                c_layers[l].w[i] = w_arr[i]->data;
                c_layers[l].s[i] = w_arr[i]->scale;
            }
        }
        char err_buf[256] = {0};
        d3d12_handle = affa_d3d12_create(&c_cfg, (const int8_t*)embed.data, embed.rscale, final_norm,
                                         c_layers.data(), rope_cos.data(), rope_sin.data(), err_buf, sizeof(err_buf));
        return d3d12_handle != nullptr;
    }
#endif

    int select_backend(const std::string& mode) {
        std::string m = mode;
        for (char& c : m) c = (char)std::tolower((unsigned char)c);
        cache_ids.clear();
#if defined(_WIN32)
        if (m == "auto" || m.empty()) {
            backend_mode = "auto";
            if (cuda_handle) { active_backend = 1; update_info_json(); return 0; }
            if (ensure_d3d12_loaded()) { active_backend = 2; update_info_json(); return 0; }
            active_backend = 0; update_info_json(); return 0;
        } else if (m == "cuda") {
            if (!cuda_handle) return -1;
            backend_mode = "cuda"; active_backend = 1; update_info_json(); return 0;
        } else if (m == "dml" || m == "directml" || m == "d3d12") {
            if (!ensure_d3d12_loaded()) return -1;
            backend_mode = "dml"; active_backend = 2; update_info_json(); return 0;
        } else if (m == "cpu" || m == "vanilla" || m == "avx2") {
            backend_mode = "cpu"; active_backend = 0; update_info_json(); return 0;
        }
#else
        backend_mode = "auto"; active_backend = 0; update_info_json(); return 0;
#endif
        return -1;
    }

    const Tensor& T(const std::string& n) const {
        auto it = tensors.find(n);
        if (it == tensors.end()) throw std::runtime_error("missing tensor " + n);
        return it->second;
    }

    void matmul(const std::pair<const Tensor*, float*>* mms, int nmm, const int8_t* xp, int xstride, const float* xi, int n) {
        const int RB = 16;
        int total = 0; int starts[8];
        for (int i = 0; i < nmm; i++) { starts[i] = total; total += (mms[i].first->rows + RB - 1) / RB; }
        const auto perm = K().perm;
#if defined(AFFA_ARM64) && defined(__clang__)
        const bool use_dp4 = K().has_dp;
#endif
#if defined(AFFA_X86)
        const bool use_avx2 = (perm == dot_perm_avx2);
#endif
        std::function<void(int, int)> f = [&](int blk, int) {
            int mi = nmm - 1; while (mi > 0 && blk < starts[mi]) mi--;
            const Tensor& w = *mms[mi].first; float* out = mms[mi].second;
            int r0 = (blk - starts[mi]) * RB, r1 = std::min(w.rows, r0 + RB);
            const size_t rb = (size_t)w.cols / 4;
#if defined(AFFA_X86)
            if (use_avx2) {
                int r = r0;
                for (; r + 4 <= r1; r += 4) {
                    const uint8_t* wr0 = w.data + (size_t)r * rb;
                    int t = 0;
                    for (; t + 4 <= n; t += 4) {
                        int acc4x4[4][4];
                        dot_perm_4x4_avx2(wr0, rb, xp + (size_t)t * xstride, xstride, w.cols, acc4x4);
                        for (int dt = 0; dt < 4; dt++) {
                            const float sc = w.scale * xi[t + dt];
                            float* dst = out + (size_t)(t + dt) * w.rows + r;
                            dst[0] = (float)acc4x4[dt][0] * sc;
                            dst[1] = (float)acc4x4[dt][1] * sc;
                            dst[2] = (float)acc4x4[dt][2] * sc;
                            dst[3] = (float)acc4x4[dt][3] * sc;
                        }
                    }
                    for (; t < n; t++) {
                        int acc4[4];
                        dot_perm_4rows_avx2(wr0, rb, xp + (size_t)t * xstride, w.cols, acc4);
                        const float sc = w.scale * xi[t];
                        float* dst = out + (size_t)t * w.rows + r;
                        dst[0] = (float)acc4[0] * sc;
                        dst[1] = (float)acc4[1] * sc;
                        dst[2] = (float)acc4[2] * sc;
                        dst[3] = (float)acc4[3] * sc;
                    }
                }
                for (; r < r1; r++) {
                    const uint8_t* wr = w.data + (size_t)r * rb;
                    for (int t = 0; t < n; t++) {
                        int acc = perm(wr, xp + (size_t)t * xstride, w.cols);
                        out[(size_t)t * w.rows + r] = (float)acc * (w.scale * xi[t]);
                    }
                }
                return;
            }
#endif
#if defined(AFFA_ARM64) && defined(__clang__)
            if (use_dp4) {
                int r = r0;
                for (; r + 4 <= r1; r += 4) {
                    const uint8_t* wr0 = w.data + (size_t)r * rb;
                    int t = 0;
                    for (; t + 4 <= n; t += 4) {
                        int acc4x4[4][4];
                        dot_perm_4x4_dp(wr0, rb, xp + (size_t)t * xstride, xstride, w.cols, acc4x4);
                        for (int dt = 0; dt < 4; dt++) {
                            const float sc = w.scale * xi[t + dt];
                            float* dst = out + (size_t)(t + dt) * w.rows + r;
                            dst[0] = (float)acc4x4[dt][0] * sc;
                            dst[1] = (float)acc4x4[dt][1] * sc;
                            dst[2] = (float)acc4x4[dt][2] * sc;
                            dst[3] = (float)acc4x4[dt][3] * sc;
                        }
                    }
                    for (; t < n; t++) {
                        int acc4[4];
                        dot_perm_4rows_dp(wr0, rb, xp + (size_t)t * xstride, w.cols, acc4);
                        const float sc = w.scale * xi[t];
                        float* dst = out + (size_t)t * w.rows + r;
                        dst[0] = (float)acc4[0] * sc;
                        dst[1] = (float)acc4[1] * sc;
                        dst[2] = (float)acc4[2] * sc;
                        dst[3] = (float)acc4[3] * sc;
                    }
                }
                for (; r < r1; r++) {
                    const uint8_t* wr = w.data + (size_t)r * rb;
                    for (int t = 0; t < n; t++) {
                        int acc = perm(wr, xp + (size_t)t * xstride, w.cols);
                        out[(size_t)t * w.rows + r] = (float)acc * (w.scale * xi[t]);
                    }
                }
                return;
            }
#endif
            for (int r = r0; r < r1; r++) {
                const uint8_t* wr = w.data + (size_t)r * rb;
                for (int t = 0; t < n; t++) {
                    int acc = perm(wr, xp + (size_t)t * xstride, w.cols);
                    out[(size_t)t * w.rows + r] = (float)acc * (w.scale * xi[t]);
                }
            }
        };
        pool.run(total, f);
    }
    static void rms(const float* xx, const float* w, float* out, int n, float eps) {
#if defined(AFFA_ARM64)
        float32x4_t vss0 = vdupq_n_f32(0.f), vss1 = vdupq_n_f32(0.f);
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            float32x4_t x0 = vld1q_f32(xx + i), x1 = vld1q_f32(xx + i + 4);
            vss0 = vfmaq_f32(vss0, x0, x0);
            vss1 = vfmaq_f32(vss1, x1, x1);
        }
        float ss = vaddvq_f32(vaddq_f32(vss0, vss1));
        for (; i < n; i++) ss += xx[i] * xx[i];
        float r = 1.0f / std::sqrt(ss / (float)n + eps);
        float32x4_t vr = vdupq_n_f32(r);
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            vst1q_f32(out + j, vmulq_f32(vmulq_f32(vld1q_f32(xx + j), vr), vld1q_f32(w + j)));
            vst1q_f32(out + j + 4, vmulq_f32(vmulq_f32(vld1q_f32(xx + j + 4), vr), vld1q_f32(w + j + 4)));
        }
        for (; j < n; j++) out[j] = xx[j] * r * w[j];
#else
        float ss = 0.f; for (int i = 0; i < n; i++) ss += xx[i] * xx[i];
        float r = 1.0f / std::sqrt(ss / (float)n + eps);
        for (int i = 0; i < n; i++) out[i] = xx[i] * r * w[i];
#endif
    }
    void compute_logits(const float* xrow, float* out) {
        rms(xrow, final_norm, tmp.data(), H, eps);
        float inv = quant_nat(tmp.data(), H, xq_nat.data());
        const auto nat = K().nat; const int8_t* eq = (const int8_t*)embed.data; const float* es = embed.rscale;
        const int active_v = tok_vocab > 0 ? tok_vocab : vocab;
        const int RB = 512; int blocks = (active_v + RB - 1) / RB;
#if defined(AFFA_ARM64) && defined(__clang__)
        const bool use_dp4 = K().has_dp;
#endif
#if defined(AFFA_X86)
        const bool use_avx2 = (nat == dot_nat_avx2);
#endif
        std::function<void(int, int)> f = [&](int blk, int) {
            int r0 = blk * RB, r1 = std::min(active_v, r0 + RB);
#if defined(AFFA_X86)
            if (use_avx2) {
                int r = r0;
                for (; r + 4 <= r1; r += 4) {
                    int acc4[4];
                    dot_nat_4rows_avx2(eq + (size_t)r * H, H, xq_nat.data(), acc4);
                    out[r + 0] = (float)acc4[0] * es[r + 0] * inv;
                    out[r + 1] = (float)acc4[1] * es[r + 1] * inv;
                    out[r + 2] = (float)acc4[2] * es[r + 2] * inv;
                    out[r + 3] = (float)acc4[3] * es[r + 3] * inv;
                }
                for (; r < r1; r++) out[r] = (float)nat(eq + (size_t)r * H, xq_nat.data(), H) * es[r] * inv;
                return;
            }
#endif
#if defined(AFFA_ARM64) && defined(__clang__)
            if (use_dp4) {
                int r = r0;
                for (; r + 4 <= r1; r += 4) {
                    int acc4[4];
                    dot_nat_4rows_dp(eq + (size_t)r * H, H, xq_nat.data(), acc4);
                    out[r + 0] = (float)acc4[0] * es[r + 0] * inv;
                    out[r + 1] = (float)acc4[1] * es[r + 1] * inv;
                    out[r + 2] = (float)acc4[2] * es[r + 2] * inv;
                    out[r + 3] = (float)acc4[3] * es[r + 3] * inv;
                }
                for (; r < r1; r++) out[r] = (float)nat(eq + (size_t)r * H, xq_nat.data(), H) * es[r] * inv;
                return;
            }
#endif
            for (int r = r0; r < r1; r++) out[r] = (float)nat(eq + (size_t)r * H, xq_nat.data(), H) * es[r] * inv;
        };
        pool.run(blocks, f);
    }
    // mode: 0 none, 1 last token logits -> out, 2 all tokens -> out[n*vocab]
    void forward(const int* ids, int n, int pos0, int mode, float* out) {
#if defined(_WIN32)
        if (active_backend == 1 && cuda_handle) {
            affa_cuda_forward(cuda_handle, ids, n, pos0, mode, out);
            return;
        }
        if (active_backend == 2 && d3d12_handle) {
            affa_d3d12_forward(d3d12_handle, ids, n, pos0, mode, out);
            return;
        }
#endif
        const int eqs = H;
        for (int t = 0; t < n; t++) {
            const int8_t* er = (const int8_t*)embed.data + (size_t)ids[t] * eqs; float s = embed.rscale[ids[t]];
            for (int i = 0; i < H; i++) x[(size_t)t * H + i] = (float)er[i] * s;
        }
        const int half = hd / 2, rep = nh / nkv; const float scale = 1.0f / std::sqrt((float)hd);
        for (int l = 0; l < L; l++) {
            const Layer& ly = layers[l];
            for (int t = 0; t < n; t++) {
                rms(&x[(size_t)t * H], ly.ln1, &h[(size_t)t * H], H, eps);
                xinv[t] = quant_perm(&h[(size_t)t * H], H, &xq[(size_t)t * H]);
            }
            std::pair<const Tensor*, float*> qkv[3] = {{&ly.q, q.data()}, {&ly.k, k.data()}, {&ly.v, v.data()}};
            matmul(qkv, 3, xq.data(), H, xinv.data(), n);
            // qk-norm, rope, cache write
            for (int t = 0; t < n; t++) {
                int pos = pos0 + t; const float* cs = &rope_cos[(size_t)pos * half]; const float* sn = &rope_sin[(size_t)pos * half];
                for (int hh = 0; hh < nh; hh++) {
                    float* qh = &q[(size_t)t * qdim + hh * hd];
                    rms(qh, ly.qn, qh, hd, eps);
                    for (int i = 0; i < half; i++) { float a = qh[i], b = qh[i + half]; qh[i] = a * cs[i] - b * sn[i]; qh[i + half] = b * cs[i] + a * sn[i]; }
                }
                for (int hh = 0; hh < nkv; hh++) {
                    float* kh = &k[(size_t)t * kvdim + hh * hd];
                    rms(kh, ly.kn, kh, hd, eps);
                    for (int i = 0; i < half; i++) { float a = kh[i], b = kh[i + half]; kh[i] = a * cs[i] - b * sn[i]; kh[i + half] = b * cs[i] + a * sn[i]; }
                }
                uint16_t* kd = kc + ((size_t)l * n_ctx + pos) * kvdim; uint16_t* vd = vc + ((size_t)l * n_ctx + pos) * kvdim;
                for (int i = 0; i < kvdim; i++) { kd[i] = f2bf(k[(size_t)t * kvdim + i]); vd[i] = f2bf(v[(size_t)t * kvdim + i]); }
            }
            // attention
            {
                std::function<void(int, int)> f = [&](int blk, int tid) {
                    int t = blk / nh, hh = blk % nh; int pos = pos0 + t; int kvh = hh / rep;
                    const float* qh = &q[(size_t)t * qdim + hh * hd];
                    float* sc = scores[tid].data();
                    const uint16_t* Kb = kc + (size_t)l * n_ctx * kvdim + kvh * hd;
                    const uint16_t* Vb = vc + (size_t)l * n_ctx * kvdim + kvh * hd;
                    float mx = -1e30f;
                    for (int j = 0; j <= pos; j++) {
                        const uint16_t* kj = Kb + (size_t)j * kvdim; float s = 0.f;
                        for (int d = 0; d < hd; d++) s += qh[d] * bf2f(kj[d]);
                        s *= scale; sc[j] = s; if (s > mx) mx = s;
                    }
                    float sum = 0.f; for (int j = 0; j <= pos; j++) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
                    float inv = 1.0f / sum; float* o = &ao[(size_t)t * qdim + hh * hd];
                    for (int d = 0; d < hd; d++) o[d] = 0.f;
                    for (int j = 0; j <= pos; j++) {
                        const uint16_t* vj = Vb + (size_t)j * kvdim; float p = sc[j] * inv;
                        for (int d = 0; d < hd; d++) o[d] += p * bf2f(vj[d]);
                    }
                };
                pool.run(n * nh, f);
            }
            for (int t = 0; t < n; t++) xinv[t] = quant_perm(&ao[(size_t)t * qdim], qdim, &xq[(size_t)t * qdim]);
            { std::pair<const Tensor*, float*> om[1] = {{&ly.o, tmp_big()}}; matmul(om, 1, xq.data(), qdim, xinv.data(), n); }
            for (size_t i = 0; i < (size_t)n * H; i++) x[i] += big[i];
            for (int t = 0; t < n; t++) {
                rms(&x[(size_t)t * H], ly.ln2, &h[(size_t)t * H], H, eps);
                xinv[t] = quant_perm(&h[(size_t)t * H], H, &xq[(size_t)t * H]);
            }
            std::pair<const Tensor*, float*> gu[2] = {{&ly.g, g.data()}, {&ly.u, u.data()}};
            matmul(gu, 2, xq.data(), H, xinv.data(), n);
            for (size_t i = 0; i < (size_t)n * I; i++) { float gv = g[i]; g[i] = gv / (1.0f + std::exp(-gv)) * u[i]; }
            for (int t = 0; t < n; t++) xinv[t] = quant_perm(&g[(size_t)t * I], I, &xq[(size_t)t * I]);
            { std::pair<const Tensor*, float*> dm[1] = {{&ly.d, tmp_big()}}; matmul(dm, 1, xq.data(), I, xinv.data(), n); }
            for (size_t i = 0; i < (size_t)n * H; i++) x[i] += big[i];
        }
        if (mode == 1) compute_logits(&x[(size_t)(n - 1) * H], out);
        else if (mode == 2) for (int t = 0; t < n; t++) compute_logits(&x[(size_t)t * H], out + (size_t)t * vocab);
    }
    std::vector<float> big;
    float* tmp_big() { return big.data(); }

    void init_runtime(int n_threads, int ctx) {
        threads = (n_threads > 0) ? n_threads : pool.optimal_threads();
        pool.start(threads);
        n_ctx = std::min(std::max(64, ctx), max_pos);
        kc = (uint16_t*)calloc((size_t)L * n_ctx * kvdim, 2); vc = (uint16_t*)calloc((size_t)L * n_ctx * kvdim, 2);
        if (!kc || !vc) throw std::runtime_error("out of memory for KV cache");
        int half = hd / 2; rope_cos.resize((size_t)n_ctx * half); rope_sin.resize((size_t)n_ctx * half);
        for (int p = 0; p < n_ctx; p++) for (int i = 0; i < half; i++) {
            float inv = 1.0f / std::pow(theta, (float)(2 * i) / (float)hd);
            float a = (float)p * inv; rope_cos[(size_t)p * half + i] = std::cos(a); rope_sin[(size_t)p * half + i] = std::sin(a);
        }
        int mc = std::max(H, std::max(qdim, I));
        x.assign((size_t)CH * H, 0); h.assign((size_t)CH * H, 0); q.assign((size_t)CH * qdim, 0); k.assign((size_t)CH * kvdim, 0);
        v.assign((size_t)CH * kvdim, 0); ao.assign((size_t)CH * qdim, 0); g.assign((size_t)CH * I, 0); u.assign((size_t)CH * I, 0);
        tmp.assign(std::max(H, hd), 0); big.assign((size_t)CH * H, 0); logits.assign(vocab, 0);
        xq.assign((size_t)CH * mc, 0); xq_nat.assign(H, 0); xinv.assign(CH, 0);
        scores.assign(64, std::vector<float>()); for (auto& s : scores) s.assign(n_ctx + 8, 0);
        cache_ids.clear();
#if defined(_WIN32)
        char dev_name[256] = {0};
        if (affa_cuda_device_name(dev_name, sizeof(dev_name)) == 0) {
            AffaCudaCfg c_cfg = { H, I, L, nh, nkv, hd, vocab, n_ctx, eps };
            std::vector<AffaCudaLayer> c_layers(L);
            for (int l = 0; l < L; l++) {
                c_layers[l].ln1 = layers[l].ln1;
                c_layers[l].ln2 = layers[l].ln2;
                c_layers[l].qn = layers[l].qn;
                c_layers[l].kn = layers[l].kn;
                const Tensor* w_arr[7] = { &layers[l].q, &layers[l].k, &layers[l].v, &layers[l].o, &layers[l].g, &layers[l].u, &layers[l].d };
                for (int i = 0; i < 7; i++) {
                    c_layers[l].w[i] = w_arr[i]->data;
                    c_layers[l].s[i] = w_arr[i]->scale;
                }
            }
            char err_buf[256] = {0};
            cuda_handle = affa_cuda_create(&c_cfg, (const int8_t*)embed.data, embed.rscale, final_norm,
                                           c_layers.data(), rope_cos.data(), rope_sin.data(), err_buf, sizeof(err_buf));
            if (cuda_handle) {
                cuda_dev_name = dev_name;
            }
        }
        char d3d_name[256] = {0};
        if (affa_d3d12_device_name(d3d_name, sizeof(d3d_name)) == 0) {
            d3d12_dev_name = d3d_name;
        }
        // Dynamic 3-Tier Cascade: Tier 1 (CUDA) -> Tier 2 (DirectML / D3D12) -> Tier 3 (CPU AVX2)
        if (cuda_handle) {
            active_backend = 1;
        } else if (!d3d12_dev_name.empty() && ensure_d3d12_loaded()) {
            active_backend = 2;
        } else {
            active_backend = 0;
        }
#endif
        update_info_json();
    }
    void run_tokens(const int* ids, int n, int pos0, bool last_logits) {
        int done = 0;
        int chunk_sz = (active_backend != 0) ? 256 : CH;
        while (done < n) {
            int c = std::min(chunk_sz, n - done);
            bool lastchunk = (done + c == n);
            forward(ids + done, c, pos0 + done, (lastchunk && last_logits) ? 1 : 0, logits.data());
            done += c;
        }
    }
};

namespace {

bool parse_header(AffaModel& m, std::string& err) {
    const uint8_t* p = m.file.p; size_t sz = m.file.size;
    if (sz < 64 || std::memcmp(p, "AFFA", 4) != 0) { err = "not an .affa file (bad magic)"; return false; }
    uint32_t ver; uint64_t hlen; std::memcpy(&ver, p + 4, 4); std::memcpy(&hlen, p + 8, 8);
    if (ver != 2) { err = "unsupported .affa version " + std::to_string(ver); return false; }
    if (16 + hlen > sz) { err = "header truncated"; return false; }
    std::string hdr((const char*)p + 16, (size_t)hlen);
    size_t data_start = (16 + (size_t)hlen + 63) / 64 * 64;
    size_t pos = 0;
    while (pos < hdr.size()) {
        size_t e = hdr.find('\n', pos); if (e == std::string::npos) e = hdr.size();
        std::string line = hdr.substr(pos, e - pos); pos = e + 1;
        if (line == "END") break;
        if (line.rfind("T|", 0) == 0) {
            std::vector<std::string> f; size_t a = 0;
            while (true) { size_t b = line.find('|', a); if (b == std::string::npos) { f.push_back(line.substr(a)); break; } f.push_back(line.substr(a, b - a)); a = b + 1; }
            if (f.size() != 8) { err = "bad tensor line"; return false; }
            Tensor t; std::string dt = f[2];
            size_t off = std::stoull(f[3]), nb = std::stoull(f[4]);
            t.rows = std::stoi(f[5]); t.cols = std::stoi(f[6]); t.scale = (float)std::strtod(f[7].c_str(), nullptr);
            if (data_start + off + nb > sz) { err = "tensor '" + f[1] + "' out of bounds (file truncated?)"; return false; }
            t.data = p + data_start + off;
            if (dt == "t2") { t.dtype = 2; if (nb != (size_t)t.rows * t.cols / 4 || t.cols % 128) { err = "bad t2 tensor " + f[1]; return false; } }
            else if (dt == "q8") { t.dtype = 8; if (nb != (size_t)t.rows * t.cols + (size_t)t.rows * 4) { err = "bad q8 tensor"; return false; } t.rscale = (const float*)(t.data + (size_t)t.rows * t.cols); }
            else if (dt == "f32") t.dtype = 32; else { err = "unknown dtype " + dt; return false; }
            m.tensors[f[1]] = t;
        } else {
            size_t eq = line.find('='); if (eq == std::string::npos) continue;
            m.kv[line.substr(0, eq)] = line.substr(eq + 1);
        }
    }
    auto gi = [&](const char* k) { auto it = m.kv.find(k); if (it == m.kv.end()) throw std::runtime_error(std::string("header missing ") + k); return std::stoi(it->second); };
    auto gf = [&](const char* k, float d) { auto it = m.kv.find(k); return it == m.kv.end() ? d : (float)std::strtod(it->second.c_str(), nullptr); };
    auto gs = [&](const char* k) { auto it = m.kv.find(k); return it == m.kv.end() ? std::string() : it->second; };
    m.H = gi("hidden"); m.I = gi("inter"); m.L = gi("layers"); m.nh = gi("heads"); m.nkv = gi("kv_heads"); m.hd = gi("head_dim");
    m.vocab = gi("vocab"); m.tok_vocab = gi("tok_vocab"); m.max_pos = gi("max_pos");
    m.theta = gf("rope_theta", 1e6f); m.eps = gf("rms_eps", 1e-6f);
    m.id_im_start = gi("im_start"); m.id_im_end = gi("im_end"); m.id_eot = gi("eot");
    m.id_think_open = gi("think_open"); m.id_think_close = gi("think_close");
    m.name = gs("name"); m.owner = gs("owner"); m.studio = gs("studio"); m.system_default = gs("system_default");
    m.def_temp = gf("def_temp", 0.7f); m.def_top_p = gf("def_top_p", 0.9f); m.def_top_k = (int)gf("def_top_k", 40); m.def_rep = gf("def_rep_pen", 1.08f);
    m.qdim = m.nh * m.hd; m.kvdim = m.nkv * m.hd;
    if (m.nh % m.nkv || m.hd % 2 || m.H % 128 || m.I % 128 || m.qdim % 128) { err = "unsupported model dimensions"; return false; }
    m.embed = m.T("embed");
    if (m.embed.rows != m.vocab || m.embed.cols != m.H) { err = "embedding shape mismatch"; return false; }
    m.final_norm = (const float*)m.T("final_norm").data;
    m.layers.resize(m.L);
    for (int i = 0; i < m.L; i++) {
        std::string P = "L" + std::to_string(i) + ".";
        Layer& l = m.layers[i];
        l.ln1 = (const float*)m.T(P + "ln1").data; l.ln2 = (const float*)m.T(P + "ln2").data;
        l.qn = (const float*)m.T(P + "qn").data; l.kn = (const float*)m.T(P + "kn").data;
        l.q = m.T(P + "q"); l.k = m.T(P + "k"); l.v = m.T(P + "v"); l.o = m.T(P + "o");
        l.g = m.T(P + "g"); l.u = m.T(P + "u"); l.d = m.T(P + "d");
        if (l.q.rows != m.qdim || l.k.rows != m.kvdim || l.o.cols != m.qdim || l.d.cols != m.I || l.g.rows != m.I) { err = "layer shape mismatch"; return false; }
    }
    size_t toff = (size_t)gi("tok_offset"), tn = (size_t)gi("tok_nbytes");
    if (data_start + toff + tn > sz) { err = "tokenizer out of bounds (file truncated?)"; return false; }
    if (!m.tk.load(p + data_start + toff, tn, err)) return false;
    return true;
}

AffaModel* finish_load(AffaModel* m, int n_threads, int n_ctx, char* err, int errlen, const std::string& e0) {
    auto fail = [&](const std::string& e) { if (err && errlen) { std::snprintf(err, errlen, "%s", e.c_str()); } delete m; return (AffaModel*)nullptr; };
    if (!e0.empty()) return fail(e0);
    try {
        std::string e;
        if (!parse_header(*m, e)) return fail(e);
        m->init_runtime(n_threads, n_ctx);
    } catch (const std::exception& ex) { return fail(ex.what()); }
    return m;
}

// utf-8 streaming helper: returns number of bytes of a complete prefix
size_t utf8_complete_prefix(const std::string& s) {
    size_t n = s.size(), i = n;
    int back = 0;
    while (i > 0 && back < 4) {
        unsigned char c = (unsigned char)s[i - 1];
        if ((c & 0xC0) == 0x80) { i--; back++; continue; }
        size_t need = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 1;
        return (n - (i - 1) >= need) ? n : (i - 1);
    }
    return n;
}

int sample_token(AffaModel* m, float* lg, const AffaParams& P, const std::vector<int>& recent, std::mt19937& rng, bool allow_think_close = false, bool allow_im_end = true) {
    const int V = m->vocab, TV = m->tok_vocab;
    const uint8_t* added = m->tk.added.data();
    const int id_end = m->id_im_end, id_eot = m->id_eot, id_tc = m->id_think_close;
    if (P.rep_penalty > 1.0f) {
        for (int id : recent) if (id >= 0 && id < TV) lg[id] = lg[id] > 0 ? lg[id] / P.rep_penalty : lg[id] * P.rep_penalty;
    }
    auto is_allowed = [&](int i) -> bool {
        if (!added[i]) return true;
        if (allow_im_end && (i == id_end || i == id_eot)) return true;
        if (allow_think_close && i == id_tc) return true;
        return false;
    };
    if (P.temperature <= 0.f) {
        int b = 0; float best_v = -1e30f;
        for (int i = 0; i < TV; i++) {
            if (lg[i] > best_v && is_allowed(i)) { best_v = lg[i]; b = i; }
        }
        return b;
    }
    int k = P.top_k > 0 ? std::min(P.top_k, TV) : std::min(TV, 64);
    static thread_local std::vector<std::pair<float, int>> cand;
    cand.clear();
    cand.reserve(k + 1);
    float threshold = -1e30f;
    auto min_cmp = [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; };
    for (int i = 0; i < TV; i++) {
        float v = lg[i];
        if (v <= threshold) continue;
        if (!is_allowed(i)) continue;
        if ((int)cand.size() < k) {
            cand.push_back({v, i});
            if ((int)cand.size() == k) {
                std::make_heap(cand.begin(), cand.end(), min_cmp);
                threshold = cand[0].first;
            }
        } else {
            std::pop_heap(cand.begin(), cand.end(), min_cmp);
            cand.back() = {v, i};
            std::push_heap(cand.begin(), cand.end(), min_cmp);
            threshold = cand[0].first;
        }
    }
    if (cand.empty()) return 0;
    std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    int actual_k = (int)cand.size();
    float mx = cand[0].first, sum = 0.f; std::vector<float> pr(actual_k);
    for (int i = 0; i < actual_k; i++) { pr[i] = std::exp((cand[i].first - mx) / P.temperature); sum += pr[i]; }
    for (int i = 0; i < actual_k; i++) pr[i] /= sum;
    int keep = actual_k;
    if (P.top_p > 0.f && P.top_p < 1.f) { float cum = 0; for (int i = 0; i < actual_k; i++) { cum += pr[i]; if (cum >= P.top_p) { keep = i + 1; break; } } }
    float s2 = 0; for (int i = 0; i < keep; i++) s2 += pr[i];
    float r = std::uniform_real_distribution<float>(0.f, s2)(rng), c = 0;
    for (int i = 0; i < keep; i++) { c += pr[i]; if (r <= c) return cand[i].second; }
    return cand[0].second;
}

struct Msg { std::string role, content; };

bool is_self_identity_query(const std::string& raw_msg) {
    std::string s = " ";
    for (unsigned char c : raw_msg) {
        if (std::isalnum(c)) s += (char)std::tolower(c);
        else if (!s.empty() && s.back() != ' ') s += ' ';
    }
    if (s.back() != ' ') s += ' ';

    // Check if the query refers to the AI itself (2nd person pronouns or model/creator names)
    static const char* self_words[] = {
        " you ", " your ", " yours ", " yourself ", " u ", " ur ",
        " affan ", " maba ", " qwen ", " openai ", " chatgpt ", " assistant ", " model "
    };
    for (const char* w : self_words) {
        if (s.find(w) != std::string::npos) return true;
    }

    // Also keep identity system prompt for short greetings
    std::string trimmed = s.substr(1, s.size() >= 2 ? s.size() - 2 : 0);
    static const char* greetings[] = {
        "hi", "hello", "hey", "yo", "sup", "hola", "salam", "assalamualaikum", "assalamu alaikum",
        "hi there", "hello there", "hey there", "good morning", "good afternoon", "good evening"
    };
    for (const char* g : greetings) {
        if (trimmed == g) return true;
    }
    return false;
}

std::string rewrite_third_person_who(const std::string& raw_msg) {
    if (is_self_identity_query(raw_msg)) return raw_msg;
    size_t i = 0;
    while (i < raw_msg.size() && std::isspace((unsigned char)raw_msg[i])) i++;
    std::string lower;
    for (size_t j = i; j < std::min(raw_msg.size(), i + 24); j++) {
        lower += (char)std::tolower((unsigned char)raw_msg[j]);
    }
    static const char* prefixes[] = {
        "who was ", "who is ", "who were ", "who are ", "who's ", "whos "
    };
    for (const char* p : prefixes) {
        size_t plen = std::strlen(p);
        if (lower.compare(0, plen, p) == 0) {
            std::string subj = raw_msg.substr(i + plen);
            while (!subj.empty() && (subj.back() == '?' || subj.back() == '.' || subj.back() == '!' || std::isspace((unsigned char)subj.back()))) {
                subj.pop_back();
            }
            if (!subj.empty()) {
                return "Tell me about the history and biography of " + subj + ".";
            }
        }
    }
    return raw_msg;
}

void build_prompt(AffaModel* m, const std::vector<Msg>& msgs, const std::string& sys, std::vector<int>& ids, const std::string& assistant_prefix = "") {
    ids.clear();
    auto enc = [&](const std::string& s) { m->tk.encode(s, ids); };
    auto close_think = [&]() {
        if (m->id_think_open >= 0 && m->id_think_close >= 0) {
            ids.push_back(m->id_think_open); enc("\n\n"); ids.push_back(m->id_think_close); enc("\n\n");
        } else enc("<think>\n\n</think>\n\n");
    };
    if (!sys.empty()) { ids.push_back(m->id_im_start); enc("system\n" + sys); ids.push_back(m->id_im_end); enc("\n"); }
    for (size_t idx = 0; idx < msgs.size(); idx++) {
        const auto& ms = msgs[idx];
        if (ms.role == "assistant") {
            std::string clean = ms.content;
            size_t ts = clean.find("<think>");
            if (ts != std::string::npos) {
                size_t te = clean.find("</think>", ts);
                if (te != std::string::npos) {
                    clean = clean.substr(0, ts) + clean.substr(te + 8);
                }
            }
            while (!clean.empty() && (clean.front() == '\n' || clean.front() == '\r' || clean.front() == ' ')) clean.erase(clean.begin());
            ids.push_back(m->id_im_start); enc("assistant\n"); close_think(); enc(clean); ids.push_back(m->id_im_end); enc("\n");
        } else {
            std::string content = (idx + 1 == msgs.size() && ms.role == "user") ? rewrite_third_person_who(ms.content) : ms.content;
            ids.push_back(m->id_im_start); enc(ms.role + "\n" + content); ids.push_back(m->id_im_end); enc("\n");
        }
    }
    ids.push_back(m->id_im_start); enc("assistant\n"); close_think();
    if (!assistant_prefix.empty()) enc(assistant_prefix);
}

}  // namespace

// ============================================================================
//  C API
// ============================================================================
extern "C" {

AffaModel* affa_load_file(const char* path, int n_threads, int n_ctx, char* err, int errlen) {
    AffaModel* m = new AffaModel(); std::string e;
    if (!m->file.open_path(path, e)) return finish_load(m, n_threads, n_ctx, err, errlen, e);
    return finish_load(m, n_threads, n_ctx, err, errlen, "");
}
AffaModel* affa_load_fd(int fd, int n_threads, int n_ctx, char* err, int errlen) {
#if defined(_WIN32)
    (void)fd; (void)n_threads; (void)n_ctx; if (err && errlen) std::snprintf(err, errlen, "fd loading not supported on Windows"); return nullptr;
#else
    AffaModel* m = new AffaModel(); std::string e;
    if (!m->file.open_fd(fd, e)) return finish_load(m, n_threads, n_ctx, err, errlen, e);
    return finish_load(m, n_threads, n_ctx, err, errlen, "");
#endif
}
void affa_free(AffaModel* m) {
    if (!m) return;
    m->stop_flag = true;
    { std::lock_guard<std::mutex> lk(m->gen_mutex); }   // wait for any running generation to finish
    delete m;
}
const char* affa_info_json(AffaModel* m) { return m->info.c_str(); }
const char* affa_gpu_device(AffaModel* m) { return (m && !m->gpu_label.empty()) ? m->gpu_label.c_str() : ""; }
const char* affa_get_backend(AffaModel* m) {
    if (!m) return "cpu";
    return m->active_backend == 1 ? "cuda" : (m->active_backend == 2 ? "dml" : "cpu");
}
int affa_set_backend(AffaModel* m, const char* mode) {
    if (!m || !mode) return -1;
    std::lock_guard<std::mutex> lk(m->gen_mutex);
    return m->select_backend(mode);
}
void affa_default_params(AffaModel* m, AffaParams* p) {
    p->temperature = m->def_temp; p->top_p = m->def_top_p; p->top_k = m->def_top_k; p->rep_penalty = m->def_rep;
    p->max_new_tokens = 2048; p->seed = 0; p->think = 0;
}
void affa_set_threads(AffaModel* m, int n) {
    std::lock_guard<std::mutex> lk(m->gen_mutex);
    n = std::max(1, std::min(n, 32));
    if (n != m->threads) { m->threads = n; m->pool.start(n); m->update_info_json(); }
}
void affa_stop(AffaModel* m) { m->stop_flag = true; }
void affa_reset(AffaModel* m) { std::lock_guard<std::mutex> lk(m->gen_mutex); m->cache_ids.clear(); }

int affa_tokenize(AffaModel* m, const char* text, int* out, int max_out) {
    std::vector<int> ids; m->tk.encode(text, ids);
    int n = std::min((int)ids.size(), max_out); for (int i = 0; i < n; i++) out[i] = ids[i];
    return (int)ids.size();
}
int affa_detokenize(AffaModel* m, const int* ids, int n, char* out, int max_out) {
    std::string s; for (int i = 0; i < n; i++) if (ids[i] >= 0 && ids[i] < (int)m->tk.tok.size() && !m->tk.added[ids[i]]) s += m->tk.tok[ids[i]];
    int k = std::min((int)s.size(), max_out - 1); if (max_out > 0) { std::memcpy(out, s.data(), k); out[k] = 0; } return (int)s.size();
}

int affa_forward_logits(AffaModel* m, const int* ids, int n, float* out, int all_positions) {
    std::lock_guard<std::mutex> lk(m->gen_mutex);
    m->cache_ids.clear();
    if (n > m->n_ctx) return -1;
    int done = 0;
    while (done < n) {
        int c = std::min(AffaModel::CH, n - done);
        bool last = done + c == n;
        if (all_positions) m->forward(ids + done, c, done, 2, out + (size_t)done * m->vocab);
        else m->forward(ids + done, c, done, last ? 1 : 0, out);
        done += c;
    }
    return m->vocab;
}

int affa_bench(AffaModel* m, int n_prompt, int n_gen, double* out3) {
    std::lock_guard<std::mutex> lk(m->gen_mutex);
    m->cache_ids.clear();
    n_prompt = std::min(n_prompt, m->n_ctx - n_gen - 1); if (n_prompt < 1) return -1;
    std::vector<int> ids(n_prompt); uint32_t s = 12345;
    for (auto& i : ids) { s = s * 1664525u + 1013904223u; i = 1000 + (int)((s >> 8) % 4000); }
    double t0 = now_ms(); m->run_tokens(ids.data(), n_prompt, 0, true); double t1 = now_ms();
    int pos = n_prompt; int tok = 0;
    for (int i = 1; i < m->tok_vocab; i++) if (m->logits[i] > m->logits[tok]) tok = i;
    double t2 = now_ms();
    for (int i = 0; i < n_gen; i++) {
        m->forward(&tok, 1, pos++, 1, m->logits.data());
        tok = 0; for (int j = 1; j < m->tok_vocab; j++) if (m->logits[j] > m->logits[tok]) tok = j;
    }
    double t3 = now_ms();
    out3[0] = n_prompt / ((t1 - t0) / 1000.0); out3[1] = n_gen / ((t3 - t2) / 1000.0); out3[2] = t1 - t0;
    m->cache_ids.clear();
    return 0;
}

int affa_chat(AffaModel* m, int n_msgs, const char** roles, const char** contents, const char* system,
              const AffaParams* P0, AffaTokenCb cb, void* user, AffaStats* stats) {
    std::lock_guard<std::mutex> lk(m->gen_mutex);
    m->stop_flag = false;
    double t0 = now_ms();
    AffaParams P = *P0;
    std::vector<Msg> msgs;
    for (int i = 0; i < n_msgs; i++) msgs.push_back({roles[i], contents[i]});
    std::string sys;
    if (system && system[0] != '\0') {
        sys = system;
    } else if (!msgs.empty() && !is_self_identity_query(msgs.back().content)) {
        sys = "You are a helpful, knowledgeable AI assistant. Answer the user's question accurately and in detail.";
    } else {
        sys = m->system_default;
    }
    std::vector<int> ids;
    size_t first = 0;
    for (;;) {
        std::vector<Msg> cur(msgs.begin() + first, msgs.end());
        build_prompt(m, cur, sys, ids);
        if ((int)ids.size() + 64 <= m->n_ctx) break;
        if (first + 1 >= msgs.size()) return -2;           // single message too long
        first += 1;                                        // drop oldest message
    }

    std::mt19937 rng(P.seed > 0 ? (uint32_t)P.seed : std::random_device{}());
    std::string pending;
    int total_gen = 0;
    double t_first = 0;
    bool stopped = false;

    auto flush_pending = [&](bool force_all) {
        size_t cl = force_all ? pending.size() : utf8_complete_prefix(pending);
        if (cl) {
            if (cb && cb(pending.data(), (int)cl, user)) {
                pending.erase(0, cl);
                stopped = true;
            } else {
                pending.erase(0, cl);
            }
        }
    };

    int max_new = std::max(1, std::min(P.max_new_tokens, m->n_ctx - (int)ids.size() - 1));
    int lcp = 0, lim = (int)std::min(m->cache_ids.size(), ids.size());
    while (lcp < lim && m->cache_ids[lcp] == ids[lcp]) lcp++;
    if (lcp >= (int)ids.size()) lcp = (int)ids.size() - 1;
    m->cache_ids.resize(lcp);
    if (!stopped && !m->stop_flag) {
        m->run_tokens(ids.data() + lcp, (int)ids.size() - lcp, lcp, true);
        m->cache_ids.insert(m->cache_ids.end(), ids.begin() + lcp, ids.end());
    }
    double t_pre = now_ms();
    std::vector<int> recent(ids.end() - std::min<size_t>(ids.size(), 64), ids.end());
    int gen = 0;
    int pos = (int)ids.size();
    while (gen < max_new && !m->stop_flag && !stopped) {
        int tok = sample_token(m, m->logits.data(), P, recent, rng, false, gen >= 1);
        if (total_gen == 0) t_first = now_ms();
        if (tok == m->id_im_end || tok == m->id_eot) break;
        gen++;
        total_gen++;
        recent.push_back(tok); if (recent.size() > 128) recent.erase(recent.begin());
        if (!m->tk.added[tok]) {
            pending += m->tk.tok[tok];
        }
        flush_pending(false);
        if (stopped || gen >= max_new || pos + 1 >= m->n_ctx) break;
        m->forward(&tok, 1, pos++, 1, m->logits.data());
        m->cache_ids.push_back(tok);
    }
    flush_pending(true);
    double t_end = now_ms();
    if (stats) {
        stats->prompt_tokens = (int)ids.size(); stats->reused_tokens = lcp; stats->gen_tokens = total_gen;
        stats->prefill_ms = t_pre - t0; stats->ttft_ms = (total_gen ? t_first : t_end) - t0;
        stats->decode_ms = t_end - (total_gen ? t_first : t_end);
        stats->tps = (total_gen > 1 && stats->decode_ms > 0) ? (total_gen - 1) / (stats->decode_ms / 1000.0) : 0.0;
    }
    return total_gen;
}

}  // extern "C"
