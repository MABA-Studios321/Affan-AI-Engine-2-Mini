#include <windows.h>
#include <iostream>
#include <string>

typedef int (*cuInit_t)(unsigned int);
typedef int (*cuDeviceGet_t)(int*, int);
typedef int (*cuDeviceGetName_t)(char*, int, int);
typedef int (*cuDeviceGetAttribute_t)(int*, int, int);

int main() {
    HMODULE hCuda = LoadLibraryA("nvcuda.dll");
    if (!hCuda) {
        std::cout << "nvcuda.dll not found." << std::endl;
        return 1;
    }
    cuInit_t cuInit = (cuInit_t)GetProcAddress(hCuda, "cuInit");
    cuDeviceGet_t cuDeviceGet = (cuDeviceGet_t)GetProcAddress(hCuda, "cuDeviceGet");
    cuDeviceGetName_t cuDeviceGetName = (cuDeviceGetName_t)GetProcAddress(hCuda, "cuDeviceGetName");
    cuDeviceGetAttribute_t cuDeviceGetAttribute = (cuDeviceGetAttribute_t)GetProcAddress(hCuda, "cuDeviceGetAttribute");

    if (!cuInit || !cuDeviceGet || !cuDeviceGetName) {
        std::cout << "Failed to find CUDA functions in nvcuda.dll" << std::endl;
        return 1;
    }

    if (cuInit(0) != 0) {
        std::cout << "cuInit failed" << std::endl;
        return 1;
    }

    int dev = 0;
    if (cuDeviceGet(&dev, 0) != 0) {
        std::cout << "cuDeviceGet failed" << std::endl;
        return 1;
    }

    char name[256] = {0};
    cuDeviceGetName(name, sizeof(name), dev);
    int major = 0, minor = 0;
    if (cuDeviceGetAttribute) {
        cuDeviceGetAttribute(&major, 75, dev); // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        cuDeviceGetAttribute(&minor, 76, dev); // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
    }

    std::cout << "CUDA Device from C++: " << name << " (Compute " << major << "." << minor << ")" << std::endl;
    return 0;
}
