# Affan AI Engine 2 Mini
### Next-Generation 1.58-Bit Ternary On-Device AI Model & Ultra-Fast Native Inference Engine
**Created & Engineered by Muhammad Affan Bin Adnan · MABA Studios**

---

## Overview: A Breakthrough in On-Device AI Engineering

**Affan AI Engine 2 Mini** is a ground-up, ultra-efficient **1.58-bit Ternary $\{-1, 0, +1\}$ Large Language Model** paired with a bespoke, zero-dependency **Native C++ Inference Engine** designed and engineered by **Muhammad Affan Bin Adnan** at **MABA Studios**.

Traditional AI models rely on bloated 16-bit (`FP16`/`BF16`) or 32-bit floating-point weights, requiring gigabytes of expensive VRAM, heavy Python frameworks, and multi-gigabyte runtime installs just to generate a few tokens per second.

**Affan AI Engine 2 Mini shatters that paradigm.**
By constraining neural weights to pure ternary states $\{-1, 0, +1\}$ packed at **4 weights per byte** inside the proprietary `.affa` container format—and executing them through hand-written **Fused CUDA PTX + CUDA Graphs**, **Universal DirectX 12 / DirectML Compute Shaders**, and **AVX2 / ARM64 NEON SIMD Kernels**—Affan AI Engine 2 Mini delivers blistering, instantaneous intelligence entirely offline on virtually any PC or smartphone.

---

## Why Affan AI Engine 2 Mini Is a Game-Changer

### 1. Uses Only as Much Memory as a 720p YouTube Video (~255 MB)
- **Pure Model Footprint in Memory**: **254.6 MB** (`257.3 MB` total on disk including tokenizer).
- **Real-World Perspective**: Playing a single **720p HD YouTube video** in a web browser uses **~180 – 260 MB** of memory, and a **1080p Full HD video** uses **~350 – 450 MB**.
- That means the entire **Affan AI Engine 2 Mini** neural network lives inside the exact same memory footprint as a **720p video stream**—and even with its full 2048-token active context window (`~371 MB` total), it uses less memory than a single 1080p video tab!
- **8.4x Smaller Core Brain**: The 28 core Transformer layers occupy just **105.5 MB** in 1.58-bit ternary precision, compared to **~884 MB – 1.20 GB** for standard unquantized 16-bit models.

### 2. Zero External Dependencies — 620 KB Standalone Native Engine
- No Python. No PyTorch. No 4 GB CUDA Toolkit installation. No external DLL clutter.
- The entire desktop studio, HTTP server, UI, tokenizer, and 3-tier hardware compute engine compile into a single **~620 KB native `.exe`** that dynamically binds to your hardware at launch.

### 3. Dynamic 3-Tier Universal Hardware Acceleration (`CUDA` $\to$ `DirectML / D3D12` $\to$ `CPU AVX2`)
Affan AI Engine 2 Mini automatically detects and cascades across three custom-engineered hardware backends out of the box:

1. **`CUDA | GPU` (Tier 1 — NVIDIA CUDA Graph + Fused PTX Engine)**
   - Uses carry-safe register ternary unpacking, shared-memory input caching, 4-row-per-warp `__dp4a` integer dot-products, fused single-launch `Q+K+V` and `Gate+Up SwiGLU` kernels, and **Full 28-Layer CUDA Graph Capture** (executing an entire token forward pass in a **single GPU driver launch**).
2. **`DirectML | GPU` (Tier 2 — Universal DirectX 12 Compute Engine)**
   - Built with pure Shader Model 5.1 (`cs_5_1`) integer bitfield math so it runs at high speed on **any DirectX 12 GPU**—including older GPUs without Tensor Cores like the **AMD Radeon RX 580**, modern **AMD Radeon RX 6000/7000**, **Intel Arc / Iris Xe**, and **NVIDIA GPUs**.
3. **`Vanilla | CPU` (Tier 3 — AVX2 4x4 & Dynamic Core Affinity Engine)**
   - Uses 256-bit AVX2 4x4 register-tiled ternary matrix multiplication and intelligent P-core/E-core thread pinning so even PCs with **no dedicated GPU** run smoothly and responsively.

---

## Verified Performance Benchmarks

| Execution Tier | Time-To-First-Token (`TTFT`) | Live Decode Speed (`tok/s`) | Peak Prefill Speed (`tok/s`) | Compatible Hardware |
| :--- | :---: | :---: | :---: | :--- |
| **`CUDA | GPU`** *(Tier 1)* | **14 ms** | **335+ tok/s** | **4,050+ tok/s** | NVIDIA GTX / RTX GPUs |
| **`DirectML | GPU`** *(Tier 2)* | **48 ms** | **283 – 293 tok/s** | **1,048+ tok/s** | **AMD RX 580**, AMD RDNA, Intel Arc/Iris, NVIDIA |
| **`Vanilla | CPU`** *(Tier 3)* | **323 ms** | **32 – 36 tok/s** | **130+ tok/s** | Any x86-64 CPU (Intel / AMD) |

*Up to **5.5x faster TTFT**, **3.5x – 6.0x faster token generation**, and **~5.0x lower RAM/VRAM usage** than conventional unquantized 16-bit models.*

---

## Quick Start (How to Run)

1. Ensure the folder structure stays together:
   ```text
   Affan AI Engine 2 Mini/
   ├── Affan AI Engine 2 Mini.exe          # Standalone Native Studio & 3-Tier Inference Engine
   ├── models/
   │   └── affan_ai_engine_2_mini.affa     # Proprietary 1.58-Bit Ternary Model (257 MB)
   ├── LICENSE.txt                         # Official MABA Studios License
   └── README.md                           # Documentation & Architecture Guide
   ```
2. Double-click **`Affan AI Engine 2 Mini.exe`**.
3. The engine will automatically load `models/affan_ai_engine_2_mini.affa`, detect your fastest hardware tier (`CUDA | GPU`, `DirectML | GPU`, or `Vanilla | CPU`), and open the **Affan AI Engine 2 Mini** interface.
4. To switch hardware backends live or run a real-time hardware benchmark on your machine, click **Settings $\to$ Engine & Speed**.

---

## Credits & Authorship

- **Creator & Lead Architect**: **Muhammad Affan Bin Adnan**
- **Studio**: **MABA Studios**
- **Model**: **Affan AI Engine 2 Mini** (`.affa` 1.58-Bit Ternary Architecture)

© 2026 Muhammad Affan Bin Adnan · MABA Studios. All Rights Reserved.
