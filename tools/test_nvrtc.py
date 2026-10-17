import ctypes, os

torch_lib = r'C:\Users\AFFAN\AppData\Roaming\Python\Python311\site-packages\torch\lib'
os.environ["PATH"] = torch_lib + os.pathsep + os.environ["PATH"]
os.add_dll_directory(torch_lib)
nvrtc = ctypes.windll.LoadLibrary(os.path.join(torch_lib, 'nvrtc64_120_0.dll'))

cu_src = b'''
extern "C" __global__ void saxpy(float a, float *x, float *y, float *out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) out[tid] = a * x[tid] + y[tid];
}
'''
prog = ctypes.c_void_p()
res = nvrtc.nvrtcCreateProgram(ctypes.byref(prog), cu_src, b'test.cu', 0, None, None)

opts = [b'--gpu-architecture=compute_75']
c_opts = (ctypes.c_char_p * len(opts))(*opts)
cres = nvrtc.nvrtcCompileProgram(prog, len(opts), c_opts)
print('cres:', cres)

if cres == 0:
    ptx_size = ctypes.c_size_t()
    nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
    ptx = ctypes.create_string_buffer(ptx_size.value)
    nvrtc.nvrtcGetPTX(prog, ptx)
    print('Generated PTX successfully! Size:', ptx_size.value)

    cuda = ctypes.windll.LoadLibrary('nvcuda.dll')
    cuda.cuInit(0)
    dev = ctypes.c_int()
    cuda.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p()
    cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value)

    module = ctypes.c_void_p()
    lres = cuda.cuModuleLoadData(ctypes.byref(module), ptx.raw)
    print('cuModuleLoadData on RTX 5050 =', lres)
    fn = ctypes.c_void_p()
    fres = cuda.cuModuleGetFunction(ctypes.byref(fn), module, b'saxpy')
    print('cuModuleGetFunction =', fres)
else:
    log_size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
    log = ctypes.create_string_buffer(log_size.value)
    nvrtc.nvrtcGetProgramLog(prog, log)
    print('Log:', log.value.decode())
