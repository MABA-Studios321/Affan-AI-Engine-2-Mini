import subprocess, os

fxc = r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\fxc.exe"
hlsl = r"D:\affan_ai_engine_2_mini\engine_cpp\affa_d3d12_shaders.hlsl"
out_h = r"D:\affan_ai_engine_2_mini\engine_cpp\affa_d3d12_bytecode.h"

entries = ["cs_embed", "cs_norm_quant", "cs_matmul", "cs_qk_rope", "cs_attn", "cs_logits"]

with open(out_h, "w", encoding="utf-8") as hf:
    hf.write("#pragma once\n#include <stdint.h>\n// Precompiled SM 5.1 DXBC Compute Shaders (RX 580 / AMD / Intel / NVIDIA Universal GPU)\n\n")
    for ep in entries:
        tmp_cso = f"D:\\affan_ai_engine_2_mini\\build\\{ep}.cso"
        cmd = [fxc, "/T", "cs_5_1", "/E", ep, "/O3", "/nologo", "/Fo", tmp_cso, hlsl]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"FAILED {ep}:\n{res.stdout}\n{res.stderr}")
            raise SystemExit(1)
        with open(tmp_cso, "rb") as bf:
            data = bf.read()
        print(f"Compiled {ep}: {len(data)} bytes")
        hf.write(f"static const uint8_t kDxbc_{ep}[{len(data)}] = {{\n  ")
        for i, b in enumerate(data):
            hf.write(f"0x{b:02x},")
            if (i + 1) % 24 == 0:
                hf.write("\n  ")
        hf.write("\n};\n\n")

print("SUCCESS! Wrote", out_h)
