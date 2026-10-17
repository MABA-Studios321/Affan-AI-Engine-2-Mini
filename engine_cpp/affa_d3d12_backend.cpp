// Affan AI Engine 2 - Universal DirectML / DirectX 12 GPU Compute Backend
// Compatible with AMD Radeon RX 580 / Polaris / RDNA, Intel Arc / Iris Xe / UHD, Qualcomm Adreno, and NVIDIA GTX/RTX.
// Dynamically loads d3d12.dll and dxgi.dll at runtime so the .exe has zero hard DLL dependencies.
// Records the entire 28-layer Transformer pass into a single DirectX 12 Command List for minimal driver overhead!

#include "affa_d3d12.h"
#include "affa_d3d12_bytecode.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

namespace {

typedef HRESULT (WINAPI *CreateDXGIFactory1_f)(REFIID, void**);
typedef HRESULT (WINAPI *D3D12CreateDevice_f)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
typedef HRESULT (WINAPI *D3D12SerializeRootSignature_f)(const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**, ID3DBlob**);

struct D3D12Drv {
    HMODULE hDxgi = NULL;
    HMODULE hD3d12 = NULL;
    CreateDXGIFactory1_f fnCreateDXGIFactory1 = NULL;
    D3D12CreateDevice_f fnD3D12CreateDevice = NULL;
    D3D12SerializeRootSignature_f fnD3D12SerializeRootSignature = NULL;

    bool init() {
        if (hDxgi && hD3d12) return true;
        hDxgi = LoadLibraryA("dxgi.dll");
        hD3d12 = LoadLibraryA("d3d12.dll");
        if (!hDxgi || !hD3d12) return false;
        fnCreateDXGIFactory1 = (CreateDXGIFactory1_f)GetProcAddress(hDxgi, "CreateDXGIFactory1");
        fnD3D12CreateDevice = (D3D12CreateDevice_f)GetProcAddress(hD3d12, "D3D12CreateDevice");
        fnD3D12SerializeRootSignature = (D3D12SerializeRootSignature_f)GetProcAddress(hD3d12, "D3D12SerializeRootSignature");
        return (fnCreateDXGIFactory1 && fnD3D12CreateDevice && fnD3D12SerializeRootSignature);
    }
} g_d3d;

// Find the best hardware GPU adapter (prefers discrete GPUs like RX 580 / RTX 5050 with highest dedicated VRAM, skips Microsoft Basic Render Driver)
IDXGIAdapter1* find_best_gpu_adapter(std::string* out_name) {
    if (!g_d3d.init()) return nullptr;
    IDXGIFactory4* factory = nullptr;
    if (FAILED(g_d3d.fnCreateDXGIFactory1(__uuidof(IDXGIFactory4), (void**)&factory)) || !factory) return nullptr;

    IDXGIAdapter1* best_adapter = nullptr;
    SIZE_T best_vram = 0;
    std::string best_desc;

    for (UINT i = 0; ; i++) {
        IDXGIAdapter1* adapter = nullptr;
        if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            adapter->Release();
            continue;
        }
        // Check if adapter supports D3D_FEATURE_LEVEL_11_0 (RX 580 supports 12_0, all DX12 GPUs support >= 11_0)
        if (SUCCEEDED(g_d3d.fnD3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr))) {
            SIZE_T score = desc.DedicatedVideoMemory + (desc.SharedSystemMemory / 4);
            if (!best_adapter || score > best_vram) {
                if (best_adapter) best_adapter->Release();
                best_adapter = adapter;
                best_vram = score;
                char utf8[256] = {0};
                WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, utf8, sizeof(utf8), NULL, NULL);
                best_desc = utf8;
            } else {
                adapter->Release();
            }
        } else {
            adapter->Release();
        }
    }
    factory->Release();
    if (best_adapter && out_name) *out_name = best_desc;
    return best_adapter;
}

struct D3DWeight {
    ID3D12Resource* res = nullptr;
    D3D12_GPU_VIRTUAL_ADDRESS gpu = 0;
    int rows = 0, cols = 0;
    float scale = 0.f;
};

struct D3DLayerData {
    ID3D12Resource *ln1 = nullptr, *ln2 = nullptr, *qn = nullptr, *kn = nullptr;
    D3D12_GPU_VIRTUAL_ADDRESS g_ln1 = 0, g_ln2 = 0, g_qn = 0, g_kn = 0;
    D3DWeight w[7]; // q, k, v, o, g, u, d
};

constexpr int CHG = 256;

struct ShaderParams {
    uint32_t p_rows;
    uint32_t p_cols;
    uint32_t p_extra0;
    uint32_t p_extra1;
    uint32_t p_extra2;
    uint32_t p_extra3;
    float p_f0;
    float p_f1;
};

struct AffaD3D12State {
    AffaCudaCfg cfg{};
    std::string dev_name;

    ID3D12Device* dev = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* cmd_alloc = nullptr;
    ID3D12GraphicsCommandList* cmd = nullptr;
    ID3D12Fence* fence = nullptr;
    HANDLE fence_event = NULL;
    UINT64 fence_val = 0;

    ID3D12RootSignature* root_sig = nullptr;
    ID3D12PipelineState* pso_embed = nullptr;
    ID3D12PipelineState* pso_norm_quant = nullptr;
    ID3D12PipelineState* pso_matmul = nullptr;
    ID3D12PipelineState* pso_qk_rope = nullptr;
    ID3D12PipelineState* pso_attn = nullptr;
    ID3D12PipelineState* pso_logits = nullptr;

    D3D12_GPU_VIRTUAL_ADDRESS g_embed = 0, g_erscale = 0, g_fnorm = 0;
    D3D12_GPU_VIRTUAL_ADDRESS g_kv = 0, g_rcos = 0, g_rsin = 0;
    D3D12_GPU_VIRTUAL_ADDRESS g_x = 0, g_q = 0, g_k = 0, g_v = 0, g_ao = 0, g_g = 0, g_u = 0, g_big = 0;
    D3D12_GPU_VIRTUAL_ADDRESS g_xq = 0, g_xinv = 0, g_logits = 0, g_ids = 0, g_pos = 0, g_dummy = 0;
    uint32_t vc_byte_offset = 0;

    ID3D12Resource* res_logits = nullptr;
    ID3D12Resource* upload_ids = nullptr;
    ID3D12Resource* readback_logits = nullptr;
    uint8_t* mapped_ids = nullptr;
    float* mapped_logits = nullptr;

    std::vector<D3DLayerData> L;
    std::vector<ID3D12Resource*> resources;
};

void wait_gpu(AffaD3D12State* s) {
    s->fence_val++;
    s->queue->Signal(s->fence, s->fence_val);
    if (s->fence->GetCompletedValue() < s->fence_val) {
        s->fence->SetEventOnCompletion(s->fence_val, s->fence_event);
        WaitForSingleObject(s->fence_event, INFINITE);
    }
}

void uav_barrier(AffaD3D12State* s) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    b.UAV.pResource = nullptr; // global UAV barrier
    s->cmd->ResourceBarrier(1, &b);
}

ID3D12Resource* create_buffer(AffaD3D12State* s, size_t bytes, D3D12_HEAP_TYPE heap_type, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES init_state) {
    size_t aligned = (bytes + 255) & ~size_t(255);
    if (aligned == 0) aligned = 256;
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = heap_type;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = aligned;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = flags;
    ID3D12Resource* res = nullptr;
    if (FAILED(s->dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, init_state, nullptr, __uuidof(ID3D12Resource), (void**)&res))) {
        return nullptr;
    }
    s->resources.push_back(res);
    return res;
}

D3D12_GPU_VIRTUAL_ADDRESS alloc_uav(AffaD3D12State* s, size_t bytes, ID3D12Resource** out_res = nullptr) {
    ID3D12Resource* r = create_buffer(s, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    if (!r) return 0;
    if (out_res) *out_res = r;
    return r->GetGPUVirtualAddress();
}

D3D12_GPU_VIRTUAL_ADDRESS upload_static(AffaD3D12State* s, const void* src, size_t bytes) {
    size_t aligned = (bytes + 15) & ~size_t(15);
    ID3D12Resource* dst = create_buffer(s, aligned, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);
    ID3D12Resource* up = create_buffer(s, aligned, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    if (!dst || !up) return 0;
    void* ptr = nullptr;
    if (FAILED(up->Map(0, nullptr, &ptr))) return 0;
    memset(ptr, 0, aligned);
    memcpy(ptr, src, bytes);
    up->Unmap(0, nullptr);

    s->cmd_alloc->Reset();
    s->cmd->Reset(s->cmd_alloc, nullptr);
    s->cmd->CopyBufferRegion(dst, 0, up, 0, aligned);
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = dst;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    s->cmd->ResourceBarrier(1, &b);
    s->cmd->Close();
    ID3D12CommandList* lists[] = { s->cmd };
    s->queue->ExecuteCommandLists(1, lists);
    wait_gpu(s);
    return dst->GetGPUVirtualAddress();
}

ID3D12PipelineState* make_pso(AffaD3D12State* s, const uint8_t* bc, size_t len) {
    D3D12_COMPUTE_PIPELINE_STATE_DESC d{};
    d.pRootSignature = s->root_sig;
    d.CS.pShaderBytecode = bc;
    d.CS.BytecodeLength = len;
    ID3D12PipelineState* pso = nullptr;
    s->dev->CreateComputePipelineState(&d, __uuidof(ID3D12PipelineState), (void**)&pso);
    return pso;
}

void set_params(AffaD3D12State* s, uint32_t r, uint32_t c, uint32_t e0, uint32_t e1, uint32_t e2, uint32_t e3, float f0, float f1 = 0.f) {
    ShaderParams p{ r, c, e0, e1, e2, e3, f0, f1 };
    s->cmd->SetComputeRoot32BitConstants(0, 8, &p, 0);
}

void dispatch_mm(AffaD3D12State* s, const D3DWeight& w, D3D12_GPU_VIRTUAL_ADDRESS g_xq, D3D12_GPU_VIRTUAL_ADDRESS g_out, int n, bool set_pso = false) {
    if (set_pso) s->cmd->SetPipelineState(s->pso_matmul);
    set_params(s, (uint32_t)w.rows, (uint32_t)w.cols, (uint32_t)n, 0, 0, 0, w.scale);
    s->cmd->SetComputeRootShaderResourceView(1, w.gpu);
    s->cmd->SetComputeRootShaderResourceView(2, g_xq);
    s->cmd->SetComputeRootShaderResourceView(3, s->g_xinv);
    s->cmd->SetComputeRootUnorderedAccessView(7, g_out);
    // 8 subgroups * 2 rows = 16 rows per threadgroup
    UINT gx = (UINT)((w.rows + 15) / 16);
    s->cmd->Dispatch(gx, (UINT)n, 1);
}

void record_forward(AffaD3D12State* s, int n, int mode) {
    const AffaCudaCfg& c = s->cfg;
    const int H = c.H, I = c.I, nh = c.nh, nkv = c.nkv, hd = c.hd, vocab = c.vocab, n_ctx = c.n_ctx;
    const int qdim = nh * hd, kvdim = nkv * hd;
    const float eps = c.eps;
    const float scale = 1.0f / std::sqrt((float)hd);

    s->cmd->SetComputeRootSignature(s->root_sig);
    // Bind dummy buffer to all slots initially so no slot is unbound
    for (int i = 1; i <= 6; i++) s->cmd->SetComputeRootShaderResourceView(i, s->g_dummy);
    for (int i = 7; i <= 9; i++) s->cmd->SetComputeRootUnorderedAccessView(i, s->g_dummy);

    // 1. Embedding Lookup
    s->cmd->SetPipelineState(s->pso_embed);
    set_params(s, (uint32_t)H, 0, 0, 0, 0, 0, 0.f);
    s->cmd->SetComputeRootShaderResourceView(1, s->g_embed);
    s->cmd->SetComputeRootShaderResourceView(2, s->g_erscale);
    s->cmd->SetComputeRootShaderResourceView(3, s->g_ids);
    s->cmd->SetComputeRootUnorderedAccessView(7, s->g_x);
    s->cmd->Dispatch((UINT)n, 1, 1);
    uav_barrier(s);

    // 2. Layer 0 input norm
    s->cmd->SetPipelineState(s->pso_norm_quant);
    set_params(s, 0, (uint32_t)H, 0, 0, 0, 0, eps);
    s->cmd->SetComputeRootShaderResourceView(1, s->g_dummy);
    s->cmd->SetComputeRootShaderResourceView(2, s->L[0].g_ln1);
    s->cmd->SetComputeRootUnorderedAccessView(7, s->g_x);
    s->cmd->SetComputeRootUnorderedAccessView(8, s->g_xq);
    s->cmd->SetComputeRootUnorderedAccessView(9, s->g_xinv);
    s->cmd->Dispatch((UINT)n, 1, 1);
    uav_barrier(s);

    // 3. Transformer Layers
    for (int l = 0; l < c.L; l++) {
        const D3DLayerData& ly = s->L[l];

        // Q, K, V projections (no barrier needed between them since they write to separate UAVs!)
        s->cmd->SetPipelineState(s->pso_matmul);
        dispatch_mm(s, ly.w[0], s->g_xq, s->g_q, n);
        dispatch_mm(s, ly.w[1], s->g_xq, s->g_k, n);
        dispatch_mm(s, ly.w[2], s->g_xq, s->g_v, n);
        uav_barrier(s);

        // QK-Norm + RoPE + KV Cache store
        s->cmd->SetPipelineState(s->pso_qk_rope);
        set_params(s, (uint32_t)nh, (uint32_t)nkv, (uint32_t)hd, (uint32_t)l, (uint32_t)n_ctx, s->vc_byte_offset, eps);
        s->cmd->SetComputeRootShaderResourceView(1, s->g_v);
        s->cmd->SetComputeRootShaderResourceView(2, ly.g_qn);
        s->cmd->SetComputeRootShaderResourceView(3, ly.g_kn);
        s->cmd->SetComputeRootShaderResourceView(4, s->g_rcos);
        s->cmd->SetComputeRootShaderResourceView(5, s->g_rsin);
        s->cmd->SetComputeRootShaderResourceView(6, s->g_pos);
        s->cmd->SetComputeRootUnorderedAccessView(7, s->g_q);
        s->cmd->SetComputeRootUnorderedAccessView(8, s->g_k);
        s->cmd->SetComputeRootUnorderedAccessView(9, s->g_kv);
        s->cmd->Dispatch((UINT)n, (UINT)(nh + nkv), 1);
        uav_barrier(s);

        // Grouped-Query Attention
        s->cmd->SetPipelineState(s->pso_attn);
        set_params(s, (uint32_t)nh, (uint32_t)nkv, (uint32_t)hd, (uint32_t)l, (uint32_t)n_ctx, s->vc_byte_offset, scale);
        s->cmd->SetComputeRootShaderResourceView(1, s->g_q);
        s->cmd->SetComputeRootShaderResourceView(2, s->g_kv);
        s->cmd->SetComputeRootShaderResourceView(3, s->g_pos);
        s->cmd->SetComputeRootUnorderedAccessView(7, s->g_ao);
        s->cmd->Dispatch((UINT)(n * nh), 1, 1);
        uav_barrier(s);

        // Quantize attention output
        s->cmd->SetPipelineState(s->pso_norm_quant);
        set_params(s, 0, (uint32_t)qdim, 1, 0, 0, 0, eps);
        s->cmd->SetComputeRootShaderResourceView(1, s->g_dummy);
        s->cmd->SetComputeRootShaderResourceView(2, s->g_dummy);
        s->cmd->SetComputeRootUnorderedAccessView(7, s->g_ao);
        s->cmd->SetComputeRootUnorderedAccessView(8, s->g_xq);
        s->cmd->SetComputeRootUnorderedAccessView(9, s->g_xinv);
        s->cmd->Dispatch((UINT)n, 1, 1);
        uav_barrier(s);

        // Out projection
        dispatch_mm(s, ly.w[3], s->g_xq, s->g_big, n, true);
        uav_barrier(s);

        // Residual add + ln2 + quantize
        s->cmd->SetPipelineState(s->pso_norm_quant);
        set_params(s, 0, (uint32_t)H, 0, 1, 0, 0, eps);
        s->cmd->SetComputeRootShaderResourceView(1, s->g_big);
        s->cmd->SetComputeRootShaderResourceView(2, ly.g_ln2);
        s->cmd->SetComputeRootUnorderedAccessView(7, s->g_x);
        s->cmd->SetComputeRootUnorderedAccessView(8, s->g_xq);
        s->cmd->SetComputeRootUnorderedAccessView(9, s->g_xinv);
        s->cmd->Dispatch((UINT)n, 1, 1);
        uav_barrier(s);

        // Gate & Up projections (no barrier needed between them!)
        s->cmd->SetPipelineState(s->pso_matmul);
        dispatch_mm(s, ly.w[4], s->g_xq, s->g_g, n);
        dispatch_mm(s, ly.w[5], s->g_xq, s->g_u, n);
        uav_barrier(s);

        // Silu(g) * u + quantize
        s->cmd->SetPipelineState(s->pso_norm_quant);
        set_params(s, 0, (uint32_t)I, 2, 1, 0, 0, eps);
        s->cmd->SetComputeRootShaderResourceView(1, s->g_u);
        s->cmd->SetComputeRootShaderResourceView(2, s->g_dummy);
        s->cmd->SetComputeRootUnorderedAccessView(7, s->g_g);
        s->cmd->SetComputeRootUnorderedAccessView(8, s->g_xq);
        s->cmd->SetComputeRootUnorderedAccessView(9, s->g_xinv);
        s->cmd->Dispatch((UINT)n, 1, 1);
        uav_barrier(s);

        // Down projection
        dispatch_mm(s, ly.w[6], s->g_xq, s->g_big, n, true);
        uav_barrier(s);

        // Residual add + next layer ln1 (or final norm)
        D3D12_GPU_VIRTUAL_ADDRESS next_norm = (l + 1 < c.L) ? s->L[l + 1].g_ln1 : s->g_fnorm;
        s->cmd->SetPipelineState(s->pso_norm_quant);
        set_params(s, 0, (uint32_t)H, 0, 1, 0, 0, eps);
        s->cmd->SetComputeRootShaderResourceView(1, s->g_big);
        s->cmd->SetComputeRootShaderResourceView(2, next_norm);
        s->cmd->SetComputeRootUnorderedAccessView(7, s->g_x);
        s->cmd->SetComputeRootUnorderedAccessView(8, s->g_xq);
        s->cmd->SetComputeRootUnorderedAccessView(9, s->g_xinv);
        s->cmd->Dispatch((UINT)n, 1, 1);
        uav_barrier(s);

        if (l + 1 == c.L && mode == 1) {
            s->cmd->SetPipelineState(s->pso_logits);
            set_params(s, (uint32_t)vocab, (uint32_t)H, (uint32_t)(n - 1), 0, 0, 0, 0.f);
            s->cmd->SetComputeRootShaderResourceView(1, s->g_embed);
            s->cmd->SetComputeRootShaderResourceView(2, s->g_erscale);
            s->cmd->SetComputeRootShaderResourceView(3, s->g_xq);
            s->cmd->SetComputeRootShaderResourceView(4, s->g_xinv);
            s->cmd->SetComputeRootUnorderedAccessView(7, s->g_logits);
            UINT gx = (UINT)((vocab + 15) / 16);
            s->cmd->Dispatch(gx, 1, 1);
            uav_barrier(s);
        }
    }
}

} // namespace

extern "C" {

int affa_d3d12_device_name(char* out, int n) {
    std::string name;
    IDXGIAdapter1* adapter = find_best_gpu_adapter(&name);
    if (!adapter) return -1;
    adapter->Release();
    snprintf(out, n, "%s", name.c_str());
    return 0;
}

void* affa_d3d12_create(const AffaCudaCfg* cfg, const int8_t* embed, const float* erscale, const float* fnorm,
                        const AffaCudaLayer* layers, const float* rcos, const float* rsin, char* err, int errlen) {
    std::string dev_name;
    IDXGIAdapter1* adapter = find_best_gpu_adapter(&dev_name);
    if (!adapter) {
        if (err && errlen) snprintf(err, errlen, "No DirectX 12 compatible GPU found");
        return nullptr;
    }

    AffaD3D12State* s = new AffaD3D12State();
    s->cfg = *cfg;
    s->dev_name = dev_name;

    HRESULT hr = g_d3d.fnD3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), (void**)&s->dev);
    adapter->Release();
    if (FAILED(hr) || !s->dev) {
        if (err && errlen) snprintf(err, errlen, "D3D12CreateDevice failed");
        delete s;
        return nullptr;
    }

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    qd.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH;
    if (FAILED(s->dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&s->queue)) ||
        FAILED(s->dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), (void**)&s->cmd_alloc)) ||
        FAILED(s->dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, s->cmd_alloc, nullptr, __uuidof(ID3D12GraphicsCommandList), (void**)&s->cmd)) ||
        FAILED(s->dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&s->fence))) {
        if (err && errlen) snprintf(err, errlen, "Failed to create D3D12 command queue/list");
        affa_d3d12_free(s);
        return nullptr;
    }
    s->cmd->Close();
    s->fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    // Build Root Signature: Slot 0 = 8x 32-bit constants (b0), Slots 1..6 = SRVs (t0..t5), Slots 7..9 = UAVs (u0..u2)
    D3D12_ROOT_PARAMETER rp[10]{};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rp[0].Constants.ShaderRegister = 0;
    rp[0].Constants.RegisterSpace = 0;
    rp[0].Constants.Num32BitValues = 8;
    rp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    for (int i = 0; i < 6; i++) {
        rp[1 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rp[1 + i].Descriptor.ShaderRegister = i;
        rp[1 + i].Descriptor.RegisterSpace = 0;
        rp[1 + i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    for (int i = 0; i < 3; i++) {
        rp[7 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        rp[7 + i].Descriptor.ShaderRegister = i;
        rp[7 + i].Descriptor.RegisterSpace = 0;
        rp[7 + i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    D3D12_ROOT_SIGNATURE_DESC rsd{};
    rsd.NumParameters = 10;
    rsd.pParameters = rp;
    rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    ID3DBlob *sig_blob = nullptr, *err_blob = nullptr;
    if (FAILED(g_d3d.fnD3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig_blob, &err_blob))) {
        if (err && errlen) snprintf(err, errlen, "Failed to serialize D3D12 root signature");
        if (err_blob) err_blob->Release();
        affa_d3d12_free(s);
        return nullptr;
    }
    hr = s->dev->CreateRootSignature(0, sig_blob->GetBufferPointer(), sig_blob->GetBufferSize(), __uuidof(ID3D12RootSignature), (void**)&s->root_sig);
    sig_blob->Release();
    if (FAILED(hr) || !s->root_sig) {
        if (err && errlen) snprintf(err, errlen, "CreateRootSignature failed");
        affa_d3d12_free(s);
        return nullptr;
    }

    s->pso_embed = make_pso(s, kDxbc_cs_embed, sizeof(kDxbc_cs_embed));
    s->pso_norm_quant = make_pso(s, kDxbc_cs_norm_quant, sizeof(kDxbc_cs_norm_quant));
    s->pso_matmul = make_pso(s, kDxbc_cs_matmul, sizeof(kDxbc_cs_matmul));
    s->pso_qk_rope = make_pso(s, kDxbc_cs_qk_rope, sizeof(kDxbc_cs_qk_rope));
    s->pso_attn = make_pso(s, kDxbc_cs_attn, sizeof(kDxbc_cs_attn));
    s->pso_logits = make_pso(s, kDxbc_cs_logits, sizeof(kDxbc_cs_logits));

    if (!s->pso_embed || !s->pso_norm_quant || !s->pso_matmul || !s->pso_qk_rope || !s->pso_attn || !s->pso_logits) {
        if (err && errlen) snprintf(err, errlen, "Failed to create D3D12 compute PSOs");
        affa_d3d12_free(s);
        return nullptr;
    }

    const AffaCudaCfg& c = s->cfg;
    const int qdim = c.nh * c.hd, kvdim = c.nkv * c.hd;

    s->g_embed = upload_static(s, embed, (size_t)c.vocab * c.H);
    s->g_erscale = upload_static(s, erscale, (size_t)c.vocab * 4);
    s->g_fnorm = upload_static(s, fnorm, (size_t)c.H * 4);

    const int rows[7] = { qdim, kvdim, kvdim, c.H, c.I, c.I, c.H };
    const int cols[7] = { c.H, c.H, c.H, qdim, c.H, c.H, c.I };
    s->L.resize(c.L);
    for (int l = 0; l < c.L; l++) {
        D3DLayerData& cd = s->L[l];
        const AffaCudaLayer& sl = layers[l];
        cd.g_ln1 = upload_static(s, sl.ln1, (size_t)c.H * 4);
        cd.g_ln2 = upload_static(s, sl.ln2, (size_t)c.H * 4);
        cd.g_qn = upload_static(s, sl.qn, (size_t)c.hd * 4);
        cd.g_kn = upload_static(s, sl.kn, (size_t)c.hd * 4);
        for (int i = 0; i < 7; i++) {
            cd.w[i].gpu = upload_static(s, sl.w[i], (size_t)rows[i] * cols[i] / 4);
            cd.w[i].rows = rows[i];
            cd.w[i].cols = cols[i];
            cd.w[i].scale = sl.s[i];
        }
    }

    const int half = c.hd / 2;
    s->g_rcos = upload_static(s, rcos, (size_t)c.n_ctx * half * 4);
    s->g_rsin = upload_static(s, rsin, (size_t)c.n_ctx * half * 4);

    // Combined KV Cache buffer: kc in first half, vc in second half
    const size_t kv_half = ((size_t)c.L * c.n_ctx * kvdim * 2 + 255) & ~size_t(255);
    s->vc_byte_offset = (uint32_t)kv_half;
    s->g_kv = alloc_uav(s, kv_half * 2);

    int mc = c.H > qdim ? c.H : qdim;
    if (c.I > mc) mc = c.I;
    s->g_x = alloc_uav(s, (size_t)CHG * c.H * 4);
    s->g_q = alloc_uav(s, (size_t)CHG * qdim * 4);
    s->g_k = alloc_uav(s, (size_t)CHG * kvdim * 4);
    s->g_v = alloc_uav(s, (size_t)CHG * kvdim * 4);
    s->g_ao = alloc_uav(s, (size_t)CHG * qdim * 4);
    s->g_g = alloc_uav(s, (size_t)CHG * c.I * 4);
    s->g_u = alloc_uav(s, (size_t)CHG * c.I * 4);
    s->g_big = alloc_uav(s, (size_t)CHG * c.H * 4);
    s->g_xq = alloc_uav(s, (size_t)CHG * mc);
    s->g_xinv = alloc_uav(s, CHG * 4);
    s->g_logits = alloc_uav(s, (size_t)c.vocab * 4, &s->res_logits);
    s->g_dummy = alloc_uav(s, 4096);

    // Persistent Upload buffer for ids (CHG * 4 bytes) + pos0 (at offset 2048)
    s->upload_ids = create_buffer(s, 4096, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    s->upload_ids->Map(0, nullptr, (void**)&s->mapped_ids);
    s->g_ids = s->upload_ids->GetGPUVirtualAddress();
    s->g_pos = s->g_ids + 2048;

    // Persistent Readback buffer for logits
    s->readback_logits = create_buffer(s, (size_t)c.vocab * 4, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    s->readback_logits->Map(0, nullptr, (void**)&s->mapped_logits);

    return s;
}

void affa_d3d12_forward(void* h, const int* ids, int n, int pos0, int mode, float* out) {
    AffaD3D12State* s = (AffaD3D12State*)h;
    if (!s) return;
    const AffaCudaCfg& c = s->cfg;

    if (mode == 2) {
        for (int t = 0; t < n; t++) {
            affa_d3d12_forward(h, ids + t, 1, pos0 + t, 1, out + (size_t)t * c.vocab);
        }
        return;
    }

    while (n > CHG) {
        affa_d3d12_forward(h, ids, CHG, pos0, 0, nullptr);
        ids += CHG; n -= CHG; pos0 += CHG;
    }

    memcpy(s->mapped_ids, ids, (size_t)n * 4);
    *(int*)(s->mapped_ids + 2048) = pos0;

    s->cmd_alloc->Reset();
    s->cmd->Reset(s->cmd_alloc, nullptr);

    record_forward(s, n, mode);

    if (mode == 1) {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = s->res_logits;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        s->cmd->ResourceBarrier(1, &b);

        s->cmd->CopyBufferRegion(s->readback_logits, 0, s->res_logits, 0, (size_t)c.vocab * 4);

        b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        s->cmd->ResourceBarrier(1, &b);
    }

    s->cmd->Close();
    ID3D12CommandList* lists[] = { s->cmd };
    s->queue->ExecuteCommandLists(1, lists);
    wait_gpu(s);

    if (mode == 1 && out) {
        memcpy(out, s->mapped_logits, (size_t)c.vocab * 4);
    }
}

void affa_d3d12_sync(void* h) {
    AffaD3D12State* s = (AffaD3D12State*)h;
    if (s) wait_gpu(s);
}

void affa_d3d12_free(void* h) {
    AffaD3D12State* s = (AffaD3D12State*)h;
    if (!s) return;
    if (s->queue && s->fence && s->fence_event) wait_gpu(s);
    if (s->upload_ids && s->mapped_ids) s->upload_ids->Unmap(0, nullptr);
    if (s->readback_logits && s->mapped_logits) s->readback_logits->Unmap(0, nullptr);
    for (ID3D12Resource* r : s->resources) if (r) r->Release();
    if (s->pso_embed) s->pso_embed->Release();
    if (s->pso_norm_quant) s->pso_norm_quant->Release();
    if (s->pso_matmul) s->pso_matmul->Release();
    if (s->pso_qk_rope) s->pso_qk_rope->Release();
    if (s->pso_attn) s->pso_attn->Release();
    if (s->pso_logits) s->pso_logits->Release();
    if (s->root_sig) s->root_sig->Release();
    if (s->cmd) s->cmd->Release();
    if (s->cmd_alloc) s->cmd_alloc->Release();
    if (s->queue) s->queue->Release();
    if (s->fence) s->fence->Release();
    if (s->fence_event) CloseHandle(s->fence_event);
    if (s->dev) s->dev->Release();
    delete s;
}

} // extern "C"
