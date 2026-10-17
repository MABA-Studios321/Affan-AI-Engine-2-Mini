# Affan AI Engine 2 Mini

<div align="center">

![Affan AI Engine 2 Mini](ui/logo.png)

### Ultra-Fast, Zero-Dependency 1.58-Bit Ternary LLM & Native Inference Engine
**Engineered by Muhammad Affan Bin Adnan · MABA Studios**

[![Platform Windows](https://img.shields.io/badge/Platform-Windows%20%7C%20Android-blue?style=flat-square&logo=windows)](https://github.com)
[![Architecture Ternary](https://img.shields.io/badge/Architecture-1.58--Bit%20Ternary%20%7B--1%2C%200%2C%20%2B1%7D-purple?style=flat-square)](https://github.com)
[![CUDA Speed](https://img.shields.io/badge/CUDA%20Speed-335%2B%20tok%2Fs-success?style=flat-square&logo=nvidia)](https://github.com)
[![DirectX 12 Speed](https://img.shields.io/badge/DirectX%2012%20Speed-280%2B%20tok%2Fs-red?style=flat-square&logo=amd)](https://github.com)
[![RAM Footprint](https://img.shields.io/badge/RAM%20Footprint-~255%20MB%20(720p%20Video)-orange?style=flat-square)](https://github.com)
[![License MABA Studios](https://img.shields.io/badge/License-Non--Commercial%20Attribution-lightgrey?style=flat-square)](LICENSE.txt)

<br/>

[**Download Release v1.0.0 (Portable ZIP)**](https://github.com) · [**Architecture & Benchmarks**](#benchmarks) · [**Quick Start**](#quick-start) · [**License**](#license)

</div>

---

## ⚡ The Breakthrough: An LLM That Takes the RAM of a 720p Video

Traditional large language models require gigabytes of expensive VRAM, bulky Python runtimes, and gigabyte-heavy frameworks like PyTorch just to run.

**Affan AI Engine 2 Mini shatters this paradigm.**

By constraining neural weights to pure ternary states $\{-1, 0, +1\}$ packed at **4 weights per single byte** inside the proprietary `.affa` format, the entire neural network occupies **only ~255 MB in memory** (with the 28 core transformer layers taking just **105.5 MB**). 

> **Real-World Perspective:** Playing a single **720p HD YouTube video** in a browser tab uses **~180 – 260 MB of RAM**. **Affan AI Engine 2 Mini** fits inside that exact same memory footprint while streaming conversational AI at **335+ tokens per second**.

---

## 🚀 Key Highlights

- **⚡ 335+ Tokens/Second on CUDA**: Powered by **Full 28-Layer CUDA Graph Capture** (`CUgraphExec`), fused single-launch `Q+K+V` and `Gate+Up SwiGLU` kernels, and carry-safe register ternary unpacking.
- **🎮 Universal DirectX 12 Compute (AMD RX 580 / Intel / Modern GPUs)**: Hand-written **Shader Model 5.1 (`cs_5_1`)** integer bitfield shaders allow older GPUs like the AMD Radeon RX 580 and integrated Intel graphics to hit **280+ tokens/second** without needing Tensor Cores.
- **🧠 620 KB Zero-Dependency Native Windows Executable**: No Python. No PyTorch. No CUDA Toolkit required. The desktop studio, embedded HTTP server, tokenizer, and compute shaders run out of the box.
- **🎨 Dynamic Hardware Badge & Seamless UI**: Minimalist Gemini-style interface that displays real-time hardware status:
  - `CUDA | GPU` (Green badge)
  - `DirectML | GPU` (Red badge)
  - `Vanilla | CPU` (Yellow/Blue badge)
- **📱 Android APK Included**: High-speed native ARM64 NEON execution for mobile devices.

---

## 📊 Benchmarks

### 1. Affan AI Engine 2 Mini vs. Original Unquantized Models (FP16/BF16)

| Backend Tier | Time-To-First-Token (`TTFT`) | Live Decode Speed (`tok/s`) | Peak Prefill Speed (`tok/s`) | Hardware Compatibility |
| :--- | :---: | :---: | :---: | :--- |
| **`CUDA | GPU`** *(Tier 1)* | **14 ms** | **334.8 – 336.4 tok/s** | **4,050+ tok/s** | NVIDIA GTX / RTX GPUs |
| **`DirectML | GPU`** *(Tier 2)* | **48 ms** | **283.0 – 293.4 tok/s** | **1,048+ tok/s** | **AMD RX 580**, AMD RDNA, Intel Arc, NVIDIA |
| **`Vanilla | CPU`** *(Tier 3)* | **323 ms** | **32.2 – 36.1 tok/s** | **130+ tok/s** | Any x86-64 CPU (Intel / AMD) |
| **Mobile CPU** *(ARM64 NEON)* | **~450 ms** | **16 – 24 tok/s** | **~85 tok/s** | Modern ARM64 Smartphones |

*Up to **5.5x faster TTFT**, **3.5x – 6.0x faster token generation**, and **~5.0x lower RAM/VRAM footprint** than conventional unquantized 16-bit baselines.*

---

### 2. Memory Footprint Comparison

| Component | Standard Unquantized 0.6B (FP16) | **Affan AI Engine 2 Mini (1.58-bit)** | Memory Saved |
| :--- | :---: | :---: | :---: |
| **Pure Model in Memory** | **1,196 MB (~1.20 GB)** | **254.6 MB** | **4.7x smaller** |
| **28 Transformer Layers** | **884 MB** | **105.5 MB** | **8.4x smaller** |
| **Total Active VRAM (with 2K Context)** | **~1,800 – 2,000 MB** | **~371 MB** | **~5.0x less VRAM** |
| **Real-World Equivalent** | **4K 60fps Ultra HD Video** | **720p / 1080p YouTube Video** | **Runs on any device** |

---

## 🛠️ Architecture Deep-Dive

```text
┌─────────────────────────────────────────────────────────────────┐
│               Affan AI Engine 2 Mini Desktop App                │
│             Minimalist Gemini WebUI + Native HTTP Server        │
└────────────────────────────────┬────────────────────────────────┘
                                 │
     ┌───────────────────────────┴───────────────────────────┐
     ▼                                                       ▼
┌──────────────────────────┐               ┌──────────────────────────┐
│ Tier 1: NVIDIA CUDA      │               │ Tier 2: DirectML / D3D12 │
│ - 28-Layer CUDA Graphs   │               │ - SM 5.1 HLSL Shaders    │
│ - Fused QKV & SwiGLU     │               │ - RX 580 / Intel / AMD   │
│ - Register Bitfield Math │               │ - Single Command List    │
└──────────────────────────┘               └──────────────────────────┘
                                 │
                                 ▼
                   ┌──────────────────────────┐
                   │ Tier 3: Vanilla CPU      │
                   │ - AVX2 4x4 Tiled Kernels │
                   │ - Dynamic Core Affinity  │
                   │ - Fast Min-Heap Sampler  │
                   └──────────────────────────┘
```

1. **Carry-Safe Register Unpacking**: Eliminates shared-memory lookup tables entirely, unpacking 4 ternary weights simultaneously in single-cycle bitwise ALU operations without bank conflicts.
2. **Instant Single-Driver Execution via CUDA Graphs**: Instead of launching 150+ discrete kernels per token, the entire forward pass across all 28 layers is captured into an instantiated `CUgraphExec`, executing in a single GPU driver call.
3. **Single-Pass Min-Heap Top-K Sampler**: Replaced sorting overhead with a single-pass threshold filter and size-$K$ min-heap, removing CPU bottlenecks and unlocking maximum GPU throughput.

---

## 📦 Quick Start (Portable & Zero-Install)

1. Download [`Affan_AI_Engine_2_Mini.zip`](https://github.com) from the **Releases** tab.
2. Extract the archive:
   ```text
   Affan AI Engine 2 Mini/
   ├── Affan AI Engine 2 Mini.exe     # Standalone Native Executable (620 KB)
   ├── models/
   │   └── affan_ai_engine_2_mini.affa # 1.58-bit Model Weights (257 MB)
   ├── LICENSE.txt
   └── README.md
   ```
3. Double-click **`Affan AI Engine 2 Mini.exe`**.
4. The engine automatically detects your GPU (`CUDA | GPU` or `DirectML | GPU`) or cascades to CPU (`Vanilla | CPU`) and opens the interface.

---

## 💻 Building from Source

### Prerequisites
- Visual Studio 2022 / Build Tools (with C++ Desktop Development)
- Python 3.10+ (for shader compilation scripts only)
- Windows 10/11 (x64)

### Build Commands
```cmd
# 1. Clone repository
git clone https://github.com/your-username/affan-ai-engine-2-mini.git
cd affan-ai-engine-2-mini

# 2. Compile shaders & embed assets
python tools/compile_cuda_ptx.py
python tools/compile_d3d12_shaders.py
powershell -ExecutionPolicy Bypass -File tools/gen_ui_assets.ps1

# 3. Build standalone executable
engine_cpp/build_desktop.bat
```
The compiled executable will be placed in `dist/Affan AI Engine 2 Mini.exe`.

---

## 📜 License & Attribution

Copyright © 2026 **Muhammad Affan Bin Adnan (MABA Studios)**. All Rights Reserved.

- **Non-Commercial Use**: Free for personal, research, educational, and evaluation purposes.
- **Third-Party Integration**: Developers are permitted to embed and use this model and inference engine in **free, non-commercial** third-party apps, provided the following attribution is placed prominently at the **VERY TOP** of the Credits / About page:
  > *"AI Model & Inference Engine powered by Affan AI Engine 2 Mini"*  
  > *"Created & Engineered by Muhammad Affan Bin Adnan / MABA Studios"*
- **Commercial Use**: Commercial use, selling, or monetization is **strictly prohibited** without an explicit commercial license. Contact **MABA Studios** for commercial inquiries.

---

<div align="center">
<b>Affan AI Engine 2 Mini</b> · Engineered with pride by <b>Muhammad Affan Bin Adnan</b>
</div>
