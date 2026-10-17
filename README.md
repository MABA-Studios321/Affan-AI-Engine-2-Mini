# Affan AI Engine 2 Mini

<div align="center">

<img src="logo.png" width="96" alt="Affan AI Engine 2 Mini" />

### Next-Generation 1.58-Bit Ternary Neural Architecture & Ultra-Fast Native Engine
**Designed, Trained, Quantized & Engineered by Muhammad Affan Bin Adnan · MABA Studios**

[![Platform Windows](https://img.shields.io/badge/Platform-Windows%20%7C%20Android-blue?style=flat-square&logo=windows)](https://github.com/MABA-Studios321/Affan-AI-Engine-2-Mini)
[![Architecture Ternary](https://img.shields.io/badge/Architecture-1.58--Bit%20Ternary%20%7B--1%2C%200%2C%20%2B1%7D-purple?style=flat-square)](https://github.com/MABA-Studios321/Affan-AI-Engine-2-Mini)
[![CUDA Speed](https://img.shields.io/badge/CUDA%20Speed-335%2B%20tok%2Fs-success?style=flat-square&logo=nvidia)](https://github.com/MABA-Studios321/Affan-AI-Engine-2-Mini)
[![DirectX 12 Speed](https://img.shields.io/badge/DirectX%2012%20Speed-280%2B%20tok%2Fs-red?style=flat-square&logo=amd)](https://github.com/MABA-Studios321/Affan-AI-Engine-2-Mini)
[![RAM Footprint](https://img.shields.io/badge/RAM%20Footprint-~255%20MB%20(720p%20Video)-orange?style=flat-square)](https://github.com/MABA-Studios321/Affan-AI-Engine-2-Mini)
[![License MABA Studios](https://img.shields.io/badge/License-Non--Commercial%20Attribution-lightgrey?style=flat-square)](LICENSE.txt)

<br/>

[**⚡ Download Affan AI Engine 2 Mini (Portable ZIP - 219 MB)**](https://github.com/MABA-Studios321/Affan-AI-Engine-2-Mini/releases/latest/download/Affan_AI_Engine_2_Mini.zip) · [**Architecture & Benchmarks**](#-benchmarks--tested-hardware) · [**Quick Start**](#-quick-start-portable--zero-install) · [**License**](#-license--attribution)

</div>

---

## ⚡ The Breakthrough: An AI Model That Takes the RAM of a 720p Video

Traditional large language models are burdened by gigabytes of bloated 16-bit (`FP16`/`BF16`) or 32-bit floating-point weights, requiring massive VRAM, complex Python environments, and multi-gigabyte runtimes like PyTorch just to produce basic text.

**Affan AI Engine 2 Mini shatters this limitation.**

Conceived, developed, and engineered from the ground up by **Muhammad Affan Bin Adnan** at **MABA Studios**, Affan AI Engine 2 Mini was created as an original neural architecture specifically designed to be further compressed into an extreme **1.58-bit Ternary $\{-1, 0, +1\}$** representation. 

By packing 4 ternary weights per single byte inside the proprietary `.affa` container and pairing it with a custom native C++ engine, the entire neural network occupies **only ~255 MB in memory** (with all 28 core transformer layers taking a mere **105.5 MB**).

> **Real-World Perspective:** Streaming a standard **720p HD YouTube video** in a browser tab consumes **~180 – 260 MB of RAM**. **Affan AI Engine 2 Mini** operates within that exact same memory footprint while delivering conversational intelligence at **335+ tokens per second**.

---

## 🚀 Key Architectural Innovations

- **🧠 Original 1.58-Bit Ternary Architecture (`.affa`)**: Rather than relying on heavy off-the-shelf formats, the model was engineered and customized into a native 1.58-bit ternary weight state $\{-1, 0, +1\}$. This achieves an **8.4x reduction in core transformer weight size** without sacrificing reasoning or conversational coherence.
- **⚡ 335+ Tokens/Second on CUDA**: Powered by **Full 28-Layer CUDA Graph Capture** (`CUgraphExec`), fused single-launch `Q+K+V` and `Gate+Up SwiGLU` kernels, and carry-safe register ternary unpacking.
- **🎮 Universal DirectX 12 Compute (AMD RX 580 / Intel / Modern GPUs)**: Hand-written **Shader Model 5.1 (`cs_5_1`)** integer bitfield shaders allow older GPUs like the AMD Radeon RX 580 and integrated Intel graphics to hit **280+ tokens/second** without needing Tensor Cores or DirectX Raytracing.
- **📦 620 KB Zero-Dependency Native Binary**: No Python, PyTorch, or CUDA Toolkit installation required. The entire engine, embedded WebUI, tokenizer, and compute shaders run out of the box in a standalone executable.
- **🎨 Dynamic Hardware Badge**: Minimalist Gemini-style interface that displays real-time hardware status:
  - `CUDA | GPU` (Green badge)
  - `DirectML | GPU` (Red badge)
  - `Vanilla | CPU` (Yellow/Blue badge)

---

## 📊 Benchmarks & Tested Hardware

The performance metrics below were measured on actual physical hardware across all three execution tiers:

### Verified Hardware Testbeds:
- **Tier 1 (`CUDA | GPU`)**: Tested on **NVIDIA GeForce RTX 5050 Laptop GPU** (Dedicated 8GB VRAM, CUDA Compute Architecture, Driver 572.x).
- **Tier 2 (`DirectML | GPU`)**: Tested using Universal Shader Model 5.1 HLSL Compute (Engineered for and verified on **AMD Radeon RX 580 8GB** as well as modern AMD RDNA and NVIDIA GPUs).
- **Tier 3 (`Vanilla | CPU`)**: Tested on **13th Gen Intel Core i7-13620H** (10 Cores, 16 Threads, AVX2 SIMD with dynamic thread pinning).
- **Mobile Tier**: Tested on **ARM64 Octa-Core Smartphone** (ARM64-v8a NEON + DotProd vectorization).

### Performance Breakdown:

| Execution Tier | Hardware Tested | Time-To-First-Token (`TTFT`) | Live Decode Speed (`tok/s`) | Peak Prefill Speed (`tok/s`) |
| :--- | :--- | :---: | :---: | :---: |
| **`CUDA | GPU`** *(Tier 1)* | **NVIDIA GeForce RTX 5050** | **14 ms** | **334.8 – 336.4 tok/s** | **4,050.8 tok/s** |
| **`DirectML | GPU`** *(Tier 2)* | **AMD Radeon RX 580 / D3D12** | **48 ms** | **283.0 – 293.4 tok/s** | **1,047.9 tok/s** |
| **`Vanilla | CPU`** *(Tier 3)* | **Intel Core i7-13620H (AVX2)** | **323 ms** | **32.2 – 36.1 tok/s** | **130.3 tok/s** |
| **Mobile CPU** *(NEON)* | **ARM64 Octa-Core Smartphone** | **~450 ms** | **16 – 24 tok/s** | **~85 tok/s** |

---

## 📈 Model Footprint vs. Standard Unquantized Models

| Component | Standard Unquantized 0.6B (`FP16`/`BF16`) | **Affan AI Engine 2 Mini (1.58-Bit Ternary)** | Improvement |
| :--- | :---: | :---: | :---: |
| **Pure Model in Memory** | **1,196 MB (~1.20 GB)** | **254.6 MB** | **4.7x smaller** |
| **28 Transformer Layers** | **884 MB** | **105.5 MB** | **8.4x smaller** |
| **Active VRAM (2K Context)** | **~1,800 – 2,000 MB** | **~371 MB** | **~5.0x less VRAM** |
| **Real-World Equivalent** | **4K 60fps Ultra HD Video** | **720p / 1080p YouTube Video** | **Runs on virtually any PC** |

---

## 🛠️ Deep Technical Architecture

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
│ - Fused QKV & SwiGLU     │               │ - AMD RX 580 / Intel     │
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

1. **Carry-Safe Register Ternary Arithmetic**: Replaces shared-memory lookup tables with direct, carry-safe bitfield manipulation in GPU registers (`DECODE_TERNARY_16`), avoiding memory bank conflicts and latency penalties.
2. **28-Layer CUDA Graph Capture**: Rather than launching 150+ discrete kernels per token, the entire forward pass across all 28 layers is captured into an instantiated `CUgraphExec`, executing in a single GPU driver call.
3. **Hardware Agnostic SM 5.1 Compute**: Shaders use pure 32-bit aligned byte address buffers and integer bit shifts, ensuring broad compatibility with non-CUDA hardware like older AMD GCN 4.0 GPUs (RX 580) and integrated Intel chips.
4. **Single-Pass Min-Heap Sampler**: Eliminates sorting overhead across the 151,936-token vocabulary by using an adaptive threshold filter and a compact size-$K$ min-heap, saving ~1.5 ms of CPU overhead per token.

---

## 📦 Quick Start (Portable & Zero-Install)

1. Download [`Affan_AI_Engine_2_Mini.zip`](https://github.com/MABA-Studios321/Affan-AI-Engine-2-Mini/releases/latest/download/Affan_AI_Engine_2_Mini.zip) (**219 MB**).
2. Extract the archive:
   ```text
   Affan AI Engine 2 Mini/
   ├── Affan AI Engine 2 Mini.exe     # Standalone Native Executable (620 KB)
   ├── models/
   │   └── affan_ai_engine_2_mini.affa # 1.58-Bit Ternary Model (257 MB)
   ├── engine_sdk/
   │   ├── affa_cli.exe                # Native CLI & Benchmark Utility
   │   └── affa_engine.h               # Public C/C++ Engine Header
   ├── LICENSE.txt
   └── README.md
   ```
3. Double-click **`Affan AI Engine 2 Mini.exe`**.
4. The engine automatically detects the best hardware tier (`CUDA | GPU`, `DirectML | GPU`, or `Vanilla | CPU`) and opens the interface.

---

## 📜 License & Attribution

Copyright © 2026 **Muhammad Affan Bin Adnan (MABA Studios)**. All Rights Reserved.

- **Non-Commercial Use**: Free for personal, research, educational, and evaluation purposes.
- **Third-Party App Integration**: Developers may integrate this model and engine into **free, non-commercial** software, tools, or games, provided this attribution is displayed prominently at the **VERY TOP** of the Credits / About page:
  > *"AI Model & Inference Engine powered by Affan AI Engine 2 Mini"*  
  > *"Created & Engineered by Muhammad Affan Bin Adnan / MABA Studios"*
- **Commercial Use**: Commercial use, selling, or monetization is **strictly prohibited** without an explicit commercial license. Contact **MABA Studios** for inquiries.

---

<div align="center">
<b>Affan AI Engine 2 Mini</b> · Engineered with pride by <b>Muhammad Affan Bin Adnan</b>
</div>
