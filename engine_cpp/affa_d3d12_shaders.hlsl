// Affan AI Engine 2 - DirectML / DirectX 12 Universal GPU Compute Shaders (HLSL SM 5.1)
// Compatible with ALL DirectX 12 GPUs (AMD RX 580 / Polaris / RDNA, Intel Arc / Iris Xe / UHD, NVIDIA GTX/RTX)
// Zero external SDK dependencies: compiled to DXBC and executed via d3d12.dll + dxgi.dll built into Windows 10/11.

cbuffer Params : register(b0) {
    uint p_rows;      // rows or H or nh
    uint p_cols;      // cols or nkv
    uint p_extra0;    // n or hd or mode or tok_vocab
    uint p_extra1;    // qdim or layer
    uint p_extra2;    // kvdim or n_ctx
    uint p_extra3;    // n_ctx or pos0
    float p_f0;       // wscale or eps or scale
    float p_f1;       // extra scale
};

// Buffers bound per dispatch:
// SRVs: t0, t1, t2, t3, t4, t5
// UAVs: u0, u1, u2
ByteAddressBuffer b_in0 : register(t0);
ByteAddressBuffer b_in1 : register(t1);
ByteAddressBuffer b_in2 : register(t2);
ByteAddressBuffer b_in3 : register(t3);
ByteAddressBuffer b_in4 : register(t4);
ByteAddressBuffer b_in5 : register(t5);

RWByteAddressBuffer b_out0 : register(u0);
RWByteAddressBuffer b_out1 : register(u1);
RWByteAddressBuffer b_out2 : register(u2);

// Helper: unpack 4 signed int8s from a 32-bit uint and dot-product with 4 ternary 2-bit codes in `wbyte`
int dot4_ternary(uint wbyte, uint xpack) {
    int x0 = (asint(xpack << 24) >> 24);
    int x1 = (asint(xpack << 16) >> 24);
    int x2 = (asint(xpack << 8)  >> 24);
    int x3 = (asint(xpack)       >> 24);

    int c0 = (int)(wbyte & 3u);        int t0 = (c0 & 1) - ((c0 >> 1) & 1);
    int c1 = (int)((wbyte >> 2) & 3u); int t1 = (c1 & 1) - ((c1 >> 1) & 1);
    int c2 = (int)((wbyte >> 4) & 3u); int t2 = (c2 & 1) - ((c2 >> 1) & 1);
    int c3 = (int)((wbyte >> 6) & 3u); int t3 = (c3 & 1) - ((c3 >> 1) & 1);

    return t0 * x0 + t1 * x1 + t2 * x2 + t3 * x3;
}

int dot4_i8(uint apack, uint bpack) {
    int a0 = (asint(apack << 24) >> 24), b0 = (asint(bpack << 24) >> 24);
    int a1 = (asint(apack << 16) >> 24), b1 = (asint(bpack << 16) >> 24);
    int a2 = (asint(apack << 8)  >> 24), b2 = (asint(bpack << 8)  >> 24);
    int a3 = (asint(apack)       >> 24), b3 = (asint(bpack)       >> 24);
    return a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
}

uint f32_to_bf16(float f) {
    uint u = asuint(f);
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

float bf16_to_f32(uint b) {
    return asfloat((b & 0xffffu) << 16);
}

// ============================================================================
// 1. cs_embed: t0=embed(int8), t1=erscale(float), t2=ids(int), u0=x(float)
//    p_rows = H
// ============================================================================
[numthreads(256, 1, 1)]
void cs_embed(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint t = gid.x;
    uint H = p_rows;
    uint id = b_in2.Load(t * 4);
    float s = asfloat(b_in1.Load(id * 4));
    uint base_off = id * H;
    for (uint i = tid.x * 4; i < H; i += 256 * 4) {
        uint pack = b_in0.Load(base_off + i);
        float v0 = (float)(asint(pack << 24) >> 24) * s;
        float v1 = (float)(asint(pack << 16) >> 24) * s;
        float v2 = (float)(asint(pack << 8)  >> 24) * s;
        float v3 = (float)(asint(pack)       >> 24) * s;
        uint out_off = (t * H + i) * 4;
        b_out0.Store4(out_off, uint4(asuint(v0), asuint(v1), asuint(v2), asuint(v3)));
    }
}

// ============================================================================
// 2. cs_norm_quant:
//    u0 = x (read/write float), t0 = add (float, when p_extra1 != 0), t1 = w (float)
//    u1 = xq (packed int8), u2 = xinv (float)
//    p_cols = cols (up to 3072), p_extra0 = mode (0=rmsnorm, 1=quant, 2=silu*add), p_extra1 = has_add, p_f0 = eps
// ============================================================================
groupshared float sh_vec[3072];
groupshared float sh_red[256];

[numthreads(256, 1, 1)]
void cs_norm_quant(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint t = gid.x;
    uint cols = p_cols;
    uint mode = p_extra0;
    uint has_add = p_extra1;
    float eps = p_f0;
    uint idx = tid.x;

    float ss = 0.0f;
    for (uint i = idx; i < cols; i += 256) {
        uint off = (t * cols + i) * 4;
        float xval = asfloat(b_out0.Load(off));
        float v;
        if (mode == 2) {
            float aval = asfloat(b_in0.Load(off));
            v = (xval / (1.0f + exp(-xval))) * aval;
        } else if (has_add != 0) {
            float aval = asfloat(b_in0.Load(off));
            v = xval + aval;
            b_out0.Store(off, asuint(v));
        } else {
            v = xval;
        }
        sh_vec[i] = v;
        ss += v * v;
    }

    sh_red[idx] = ss;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (idx < s) sh_red[idx] += sh_red[idx + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float r = (mode == 0) ? rsqrt(sh_red[0] / (float)cols + eps) : 1.0f;

    float am = 0.0f;
    for (uint j = idx; j < cols; j += 256) {
        float h = sh_vec[j];
        if (mode == 0) {
            float wval = asfloat(b_in1.Load(j * 4));
            h = h * r * wval;
        }
        sh_vec[j] = h;
        am = max(am, abs(h));
    }

    sh_red[idx] = am;
    GroupMemoryBarrierWithGroupSync();
    for (uint s2 = 128; s2 > 0; s2 >>= 1) {
        if (idx < s2) sh_red[idx] = max(sh_red[idx], sh_red[idx + s2]);
        GroupMemoryBarrierWithGroupSync();
    }
    float max_am = max(sh_red[0], 1e-5f);
    float scale = 127.0f / max_am;

    // Pack 4 quantized int8s into 1 uint32 store
    for (uint k = idx * 4; k < cols; k += 256 * 4) {
        int q0 = clamp((int)round(sh_vec[k + 0] * scale), -127, 127);
        int q1 = clamp((int)round(sh_vec[k + 1] * scale), -127, 127);
        int q2 = clamp((int)round(sh_vec[k + 2] * scale), -127, 127);
        int q3 = clamp((int)round(sh_vec[k + 3] * scale), -127, 127);
        uint pack = ((uint)(q0 & 0xff)) | (((uint)(q1 & 0xff)) << 8) | (((uint)(q2 & 0xff)) << 16) | (((uint)(q3 & 0xff)) << 24);
        b_out1.Store(t * cols + k, pack);
    }
    if (idx == 0) {
        b_out2.Store(t * 4, asuint(1.0f / scale));
    }
}

// ============================================================================
// 3. cs_matmul: Universal SM 5.1 Ternary GEMV / GEMM
//    t0 = W (packed 2-bit ternary), t1 = xq (packed int8), t2 = xinv (float)
//    u0 = out (float)
//    p_rows = rows, p_cols = cols, p_extra0 = n, p_f0 = wscale
//    256 threads = 8 Subgroups of 32 threads; each subgroup of 32 threads computes 2 rows simultaneously! (16 rows per threadgroup)
// ============================================================================
groupshared uint4 sh_xq[256]; // caches 1024 uint32s = 4096 int8s in fast LDS / Shared Memory
groupshared int sh_mm_red[256];

[numthreads(256, 1, 1)]
void cs_matmul(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint idx = tid.x;
    uint sub = idx >> 5;       // 0..7
    uint lane = idx & 31u;     // 0..31
    uint rows = p_rows;
    uint cols = p_cols;
    uint t_tok = gid.y;
    float wscale = p_f0;

    uint nw = cols >> 4; // number of 16-col chunks (each is 4 bytes of W and 16 bytes of xq)
    uint xq_base = t_tok * cols;
    for (uint i = idx; i < nw; i += 256) {
        sh_xq[i] = b_in1.Load4(xq_base + (i << 4));
    }
    GroupMemoryBarrierWithGroupSync();

    uint r0 = (gid.x * 8u + sub) * 2u;
    bool valid_r0 = (r0 < rows);
    uint safe_r0 = valid_r0 ? r0 : 0u;

    uint row_bytes = cols >> 2;
    uint w_off0 = safe_r0 * row_bytes;
    uint w_off1 = (safe_r0 + 1u < rows) ? (w_off0 + row_bytes) : w_off0;

    int a0 = 0;
    int a1 = 0;
    for (uint wi = lane; wi < nw; wi += 32) {
        uint4 x4 = sh_xq[wi];
        uint w0 = b_in0.Load(w_off0 + (wi << 2));
        uint w1 = b_in0.Load(w_off1 + (wi << 2));

        a0 += dot4_ternary(w0 & 0xffu,         x4.x);
        a0 += dot4_ternary((w0 >> 8) & 0xffu,  x4.y);
        a0 += dot4_ternary((w0 >> 16) & 0xffu, x4.z);
        a0 += dot4_ternary(w0 >> 24,           x4.w);

        a1 += dot4_ternary(w1 & 0xffu,         x4.x);
        a1 += dot4_ternary((w1 >> 8) & 0xffu,  x4.y);
        a1 += dot4_ternary((w1 >> 16) & 0xffu, x4.z);
        a1 += dot4_ternary(w1 >> 24,           x4.w);
    }

    // Subgroup reduction across the 32 lanes using shared memory (works on 32-wide NVIDIA, 64-wide AMD GCN RX 580, and 16-wide Intel!)
    sh_mm_red[idx] = a0;
    GroupMemoryBarrierWithGroupSync();
    if (lane < 16u) sh_mm_red[idx] += sh_mm_red[idx + 16u]; GroupMemoryBarrierWithGroupSync();
    if (lane < 8u)  sh_mm_red[idx] += sh_mm_red[idx + 8u];  GroupMemoryBarrierWithGroupSync();
    if (lane < 4u)  sh_mm_red[idx] += sh_mm_red[idx + 4u];  GroupMemoryBarrierWithGroupSync();
    if (lane < 2u)  sh_mm_red[idx] += sh_mm_red[idx + 2u];  GroupMemoryBarrierWithGroupSync();
    int sum0 = sh_mm_red[sub << 5] + sh_mm_red[(sub << 5) + 1u];
    GroupMemoryBarrierWithGroupSync();

    sh_mm_red[idx] = a1;
    GroupMemoryBarrierWithGroupSync();
    if (lane < 16u) sh_mm_red[idx] += sh_mm_red[idx + 16u]; GroupMemoryBarrierWithGroupSync();
    if (lane < 8u)  sh_mm_red[idx] += sh_mm_red[idx + 8u];  GroupMemoryBarrierWithGroupSync();
    if (lane < 4u)  sh_mm_red[idx] += sh_mm_red[idx + 4u];  GroupMemoryBarrierWithGroupSync();
    if (lane < 2u)  sh_mm_red[idx] += sh_mm_red[idx + 2u];  GroupMemoryBarrierWithGroupSync();
    int sum1 = sh_mm_red[sub << 5] + sh_mm_red[(sub << 5) + 1u];

    if (lane == 0u && valid_r0) {
        float xinv = asfloat(b_in2.Load(t_tok * 4));
        float sc = wscale * xinv;
        uint out_base = (t_tok * rows + r0) * 4;
        b_out0.Store(out_base, asuint((float)sum0 * sc));
        if (r0 + 1u < rows) {
            b_out0.Store(out_base + 4, asuint((float)sum1 * sc));
        }
    }
}

// ============================================================================
// 4. cs_qk_rope:
//    u0 = q (read/write float), u1 = k (read/write float), t0 = v (float),
//    t1 = qn (float), t2 = kn (float), t3 = rcos (float), t4 = rsin (float),
//    t5 = pos0p (int), u2 = kv_cache (packed bf16: kc in first half, vc in second half)
//    p_rows = nh, p_cols = nkv, p_extra0 = hd (128), p_extra1 = layer, p_extra2 = n_ctx, p_extra3 = kv_half_bytes, p_f0 = eps
// ============================================================================
groupshared float sh_qk[128];
groupshared float sh_qk_red[128];

[numthreads(128, 1, 1)]
void cs_qk_rope(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint t = gid.x;
    uint hh = gid.y;
    uint d = tid.x;
    uint nh = p_rows;
    uint nkv = p_cols;
    uint hd = p_extra0;
    uint layer = p_extra1;
    uint n_ctx = p_extra2;
    uint vc_byte_offset = p_extra3;
    float eps = p_f0;

    uint qdim = nh * hd;
    uint kvdim = nkv * hd;
    uint half_hd = hd >> 1;
    uint pos = b_in5.Load(0) + t;
    bool isq = (hh < nh);

    uint elem_idx = isq ? (t * qdim + hh * hd + d) : (t * kvdim + (hh - nh) * hd + d);
    float val = asfloat(isq ? b_out0.Load(elem_idx * 4) : b_out1.Load(elem_idx * 4));
    sh_qk_red[d] = val * val;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 64; s > 0; s >>= 1) {
        if (d < s) sh_qk_red[d] += sh_qk_red[d + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float r = rsqrt(sh_qk_red[0] / (float)hd + eps);
    float w = asfloat(isq ? b_in1.Load(d * 4) : b_in2.Load(d * 4));
    float yv = val * r * w;
    sh_qk[d] = yv;
    GroupMemoryBarrierWithGroupSync();

    uint i = (d < half_hd) ? d : (d - half_hd);
    uint rope_idx = (pos * half_hd + i) * 4;
    float c = asfloat(b_in3.Load(rope_idx));
    float sn = asfloat(b_in4.Load(rope_idx));
    float o = (d < half_hd) ? (yv * c - sh_qk[d + half_hd] * sn) : (yv * c + sh_qk[d - half_hd] * sn);
    GroupMemoryBarrierWithGroupSync();
    sh_qk[d] = o;
    GroupMemoryBarrierWithGroupSync();

    if (isq) {
        b_out0.Store(elem_idx * 4, asuint(o));
    } else {
        b_out1.Store(elem_idx * 4, asuint(o));
        if ((d & 1u) == 0u) {
            uint kvh = hh - nh;
            uint v_idx0 = (t * kvdim + kvh * hd + d) * 4;
            float v0 = asfloat(b_in0.Load(v_idx0));
            float v1 = asfloat(b_in0.Load(v_idx0 + 4));
            uint k_pack = (f32_to_bf16(sh_qk[d]) & 0xffffu) | ((f32_to_bf16(sh_qk[d + 1u]) & 0xffffu) << 16);
            uint v_pack = (f32_to_bf16(v0) & 0xffffu) | ((f32_to_bf16(v1) & 0xffffu) << 16);
            uint kv_byte_off = (((layer * n_ctx + pos) * kvdim) + kvh * hd + d) * 2;
            b_out2.Store(kv_byte_off, k_pack);
            b_out2.Store(vc_byte_offset + kv_byte_off, v_pack);
        }
    }
}

// ============================================================================
// 5. cs_attn:
//    t0 = q (float), t1 = kv_cache (packed bf16: kc + vc), t2 = pos0p (int)
//    u0 = ao (float)
//    p_rows = nh, p_cols = nkv, p_extra0 = hd (128), p_extra1 = layer, p_extra2 = n_ctx, p_extra3 = vc_byte_offset, p_f0 = scale
// ============================================================================
groupshared float sh_sc[2048];
groupshared float sh_q[128];
groupshared float sh_at_red[128];

[numthreads(128, 1, 1)]
void cs_attn(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint d = tid.x;
    uint nh = p_rows;
    uint nkv = p_cols;
    uint hd = p_extra0;
    uint layer = p_extra1;
    uint n_ctx = p_extra2;
    uint vc_byte_offset = p_extra3;
    float scale = p_f0;

    uint t = gid.x / nh;
    uint hh = gid.x % nh;
    uint pos = b_in2.Load(0) + t;
    uint kvh = hh / (nh / nkv);
    uint qdim = nh * hd;
    uint kvdim = nkv * hd;

    sh_q[d] = asfloat(b_in0.Load((t * qdim + hh * hd + d) * 4));
    GroupMemoryBarrierWithGroupSync();

    // 128 threads = 4 subgroups of 32 threads; each subgroup computes dot product for step j
    uint sub = d >> 5;
    uint lane = d & 31u;
    uint k_layer_base = (layer * n_ctx * kvdim + kvh * hd) * 2;

    uint num_iters = (pos + 4u) >> 2;
    for (uint it = 0; it < num_iters; it++) {
        uint j = (it << 2) + sub;
        float s = 0.0f;
        if (j <= pos) {
            uint k_off = k_layer_base + j * kvdim * 2;
            uint p0 = b_in1.Load(k_off + (lane * 4u) * 2u);
            uint p1 = b_in1.Load(k_off + (lane * 4u + 2u) * 2u);
            uint d0 = lane * 4u;
            s += sh_q[d0 + 0u] * bf16_to_f32(p0 & 0xffffu);
            s += sh_q[d0 + 1u] * bf16_to_f32(p0 >> 16);
            s += sh_q[d0 + 2u] * bf16_to_f32(p1 & 0xffffu);
            s += sh_q[d0 + 3u] * bf16_to_f32(p1 >> 16);
        }

        sh_at_red[d] = s;
        GroupMemoryBarrierWithGroupSync();
        if (lane < 16u) sh_at_red[d] += sh_at_red[d + 16u]; GroupMemoryBarrierWithGroupSync();
        if (lane < 8u)  sh_at_red[d] += sh_at_red[d + 8u];  GroupMemoryBarrierWithGroupSync();
        if (lane < 4u)  sh_at_red[d] += sh_at_red[d + 4u];  GroupMemoryBarrierWithGroupSync();
        if (lane < 2u)  sh_at_red[d] += sh_at_red[d + 2u];  GroupMemoryBarrierWithGroupSync();
        if (lane == 0u && j <= pos) {
            sh_sc[j] = (sh_at_red[sub << 5] + sh_at_red[(sub << 5) + 1u]) * scale;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float mx = -3.0e38f;
    for (uint j2 = d; j2 <= pos; j2 += 128u) mx = max(mx, sh_sc[j2]);
    sh_at_red[d] = mx;
    GroupMemoryBarrierWithGroupSync();
    for (uint sr = 64; sr > 0; sr >>= 1) {
        if (d < sr) sh_at_red[d] = max(sh_at_red[d], sh_at_red[d + sr]);
        GroupMemoryBarrierWithGroupSync();
    }
    mx = sh_at_red[0];

    float sum = 0.0f;
    for (uint j3 = d; j3 <= pos; j3 += 128u) {
        float e = exp(sh_sc[j3] - mx);
        sh_sc[j3] = e;
        sum += e;
    }
    sh_at_red[d] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint sr2 = 64; sr2 > 0; sr2 >>= 1) {
        if (d < sr2) sh_at_red[d] += sh_at_red[d + sr2];
        GroupMemoryBarrierWithGroupSync();
    }
    float inv = 1.0f / sh_at_red[0];
    GroupMemoryBarrierWithGroupSync();

    // Each thread d=0..63 accumulates 2 dimensions (2*d and 2*d+1) using 32-bit aligned loads!
    if (d < 64u) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        uint v_base = vc_byte_offset + k_layer_base + (d * 2u) * 2u;
        for (uint j4 = 0; j4 <= pos; j4++) {
            float p = sh_sc[j4] * inv;
            uint vpack = b_in1.Load(v_base + j4 * kvdim * 2u);
            acc0 += p * bf16_to_f32(vpack & 0xffffu);
            acc1 += p * bf16_to_f32(vpack >> 16);
        }
        uint ao_off = (t * qdim + hh * hd + d * 2u) * 4u;
        b_out0.Store2(ao_off, uint2(asuint(acc0), asuint(acc1)));
    }
}

// ============================================================================
// 6. cs_logits:
//    t0 = embed (packed int8), t1 = erscale (float), t2 = xq (packed int8), t3 = xinv (float)
//    u0 = logits (float)
//    p_rows = vocab, p_cols = H, p_extra0 = last_token_idx
// ============================================================================
groupshared uint4 sh_lxq[64];
groupshared int sh_lg_red[256];

[numthreads(256, 1, 1)]
void cs_logits(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint idx = tid.x;
    uint sub = idx >> 5;
    uint lane = idx & 31u;
    uint vocab = p_rows;
    uint H = p_cols;
    uint last_t = p_extra0;

    uint nw = H >> 4;
    uint xq_base = last_t * H;
    if (idx < nw) {
        sh_lxq[idx] = b_in2.Load4(xq_base + (idx << 4));
    }
    GroupMemoryBarrierWithGroupSync();

    uint r0 = (gid.x * 8u + sub) * 2u;
    bool valid_r0 = (r0 < vocab);
    uint safe_r0 = valid_r0 ? r0 : 0u;

    uint e_off0 = safe_r0 * H;
    uint e_off1 = (safe_r0 + 1u < vocab) ? (e_off0 + H) : e_off0;
    int a0 = 0, a1 = 0;
    for (uint i = lane; i < nw; i += 32u) {
        uint4 x = sh_lxq[i];
        uint4 e0 = b_in0.Load4(e_off0 + (i << 4));
        uint4 e1 = b_in0.Load4(e_off1 + (i << 4));
        a0 += dot4_i8(e0.x, x.x) + dot4_i8(e0.y, x.y) + dot4_i8(e0.z, x.z) + dot4_i8(e0.w, x.w);
        a1 += dot4_i8(e1.x, x.x) + dot4_i8(e1.y, x.y) + dot4_i8(e1.z, x.z) + dot4_i8(e1.w, x.w);
    }

    sh_lg_red[idx] = a0;
    GroupMemoryBarrierWithGroupSync();
    if (lane < 16u) sh_lg_red[idx] += sh_lg_red[idx + 16u]; GroupMemoryBarrierWithGroupSync();
    if (lane < 8u)  sh_lg_red[idx] += sh_lg_red[idx + 8u];  GroupMemoryBarrierWithGroupSync();
    if (lane < 4u)  sh_lg_red[idx] += sh_lg_red[idx + 4u];  GroupMemoryBarrierWithGroupSync();
    if (lane < 2u)  sh_lg_red[idx] += sh_lg_red[idx + 2u];  GroupMemoryBarrierWithGroupSync();
    int sum0 = sh_lg_red[sub << 5] + sh_lg_red[(sub << 5) + 1u];
    GroupMemoryBarrierWithGroupSync();

    sh_lg_red[idx] = a1;
    GroupMemoryBarrierWithGroupSync();
    if (lane < 16u) sh_lg_red[idx] += sh_lg_red[idx + 16u]; GroupMemoryBarrierWithGroupSync();
    if (lane < 8u)  sh_lg_red[idx] += sh_lg_red[idx + 8u];  GroupMemoryBarrierWithGroupSync();
    if (lane < 4u)  sh_lg_red[idx] += sh_lg_red[idx + 4u];  GroupMemoryBarrierWithGroupSync();
    if (lane < 2u)  sh_lg_red[idx] += sh_lg_red[idx + 2u];  GroupMemoryBarrierWithGroupSync();
    int sum1 = sh_lg_red[sub << 5] + sh_lg_red[(sub << 5) + 1u];

    if (lane == 0u && valid_r0) {
        float xinv = asfloat(b_in3.Load(last_t * 4));
        float rs0 = asfloat(b_in1.Load(r0 * 4));
        b_out0.Store(r0 * 4, asuint((float)sum0 * rs0 * xinv));
        if (r0 + 1u < vocab) {
            float rs1 = asfloat(b_in1.Load((r0 + 1u) * 4));
            b_out0.Store((r0 + 1u) * 4, asuint((float)sum1 * rs1 * xinv));
        }
    }
}
