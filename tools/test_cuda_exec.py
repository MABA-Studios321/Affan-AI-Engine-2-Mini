import ctypes

# Let's check kernel launch parameters and execution on RTX 5050
# We can check what cuLaunchKernel returns for each kernel
cuda = ctypes.windll.LoadLibrary('nvcuda.dll')
cuda.cuInit(0)
dev = ctypes.c_int()
cuda.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.c_void_p()
cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value)

with open(r'D:\affan_ai_engine_2_mini\engine_cpp\affa_ptx.h', 'r') as f:
    text = f.read()

# Parse the PTX
lines = []
for line in text.splitlines():
    if line.startswith('  "'):
        content = line[3:-1].encode().decode('unicode_escape')
        lines.append(content)
ptx = "".join(lines).encode()

module = ctypes.c_void_p()
res = cuda.cuModuleLoadData(ctypes.byref(module), ptx)
print('cuModuleLoadData res:', res)

# Check maximum dynamic shared memory per block on device
max_smem = ctypes.c_int()
cuda.cuDeviceGetAttribute(ctypes.byref(max_smem), 8, dev.value) # CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
print('Max shared memory per block:', max_smem.value)
max_opt_smem = ctypes.c_int()
cuda.cuDeviceGetAttribute(ctypes.byref(max_opt_smem), 97, dev.value) # CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
print('Max opt-in shared memory per block:', max_opt_smem.value)

# Look at smem_attn in affa_cuda_backend.cpp:
# s->smem_attn = ((size_t)c.n_ctx + c.hd) * 4;
# For n_ctx = 2048, hd = 64:
# smem_attn = (2048 + 64) * 4 = 8448 bytes
print('smem_attn for 2048 ctx:', (2048 + 64) * 4)

fn_attn = ctypes.c_void_p()
cuda.cuModuleGetFunction(ctypes.byref(fn_attn), module, b'k_attn')
print('k_attn func:', fn_attn.value)
