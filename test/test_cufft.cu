// 简单测试程序，验证 cuFFT 距离脉压工作正常
// 编译: nvcc -o test_cufft test_cufft.cu -lcufft

#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <complex>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while (0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult status = call; \
        if (status != CUFFT_SUCCESS) { \
            std::cerr << "cuFFT Error: " << status << std::endl; \
            exit(1); \
        } \
    } while (0)

// Kernel: 频域乘法
__global__ void multiplyHfSpecKernel(cuDoubleComplex* D_X,
                                     const cuDoubleComplex* D_HfSpec,
                                     int W, int Lraw)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < W * Lraw) {
        int f = i % Lraw;
        cuDoubleComplex X_kf = D_X[i];
        cuDoubleComplex H_f = D_HfSpec[f];
        D_X[i] = cuCmul(X_kf, H_f);
    }
}

// Kernel: 归一化 + 裁剪
__global__ void normalizeAndCropKernel(const cuDoubleComplex* D_X,
                                       cuDoubleComplex* D_rcOut,
                                       int W, int Lraw, int M,
                                       int Lraw2M, double scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < W * M) {
        int k = i / M;
        int m = i % M;
        int j = k * Lraw + m * Lraw2M;
        cuDoubleComplex X_jm = D_X[j];
        D_rcOut[i] = cuCmul(X_jm, make_cuDoubleComplex(scale, 0.0));
    }
}

int main() {
    const int W = 4;        // 脉冲数
    const int Lraw = 1024;  // FFT 长度
    const int M = 256;      // 输出距离单元数
    const int Lraw2M = Lraw / M;
    const double scale = 1.0 / double(Lraw);

    std::cout << "Testing cuFFT Range Compression" << std::endl;
    std::cout << "W=" << W << ", Lraw=" << Lraw << ", M=" << M << std::endl;

    size_t size_in = (size_t)W * Lraw * sizeof(cuDoubleComplex);
    size_t size_hfspec = (size_t)Lraw * sizeof(cuDoubleComplex);
    size_t size_out = (size_t)W * M * sizeof(cuDoubleComplex);

    // Host 数据
    std::vector<std::complex<double>> h_inData(W * Lraw);
    std::vector<std::complex<double>> h_HfSpec(Lraw, 1.0); // 简单参考函数
    std::vector<std::complex<double>> h_rcOut(W * M);

    // 初始化输入为简单测试数据
    for (size_t i = 0; i < W * Lraw; ++i) {
        h_inData[i] = std::complex<double>(1.0, 0.0);
    }

    // Device 内存
    cuDoubleComplex *D_inOut, *D_HfSpec, *D_rcOut;
    CUDA_CHECK(cudaMalloc(&D_inOut, size_in));
    CUDA_CHECK(cudaMalloc(&D_HfSpec, size_hfspec));
    CUDA_CHECK(cudaMalloc(&D_rcOut, size_out));

    // Host → Device
    std::cout << "Copying data to GPU..." << std::endl;
    std::vector<cuDoubleComplex> h_in_cuda(W * Lraw);
    std::vector<cuDoubleComplex> h_hf_cuda(Lraw);
    for (int i = 0; i < W * Lraw; ++i)
        h_in_cuda[i] = make_cuDoubleComplex(h_inData[i].real(), h_inData[i].imag());
    for (int i = 0; i < Lraw; ++i)
        h_hf_cuda[i] = make_cuDoubleComplex(h_HfSpec[i].real(), h_HfSpec[i].imag());

    CUDA_CHECK(cudaMemcpy(D_inOut, h_in_cuda.data(), size_in, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(D_HfSpec, h_hf_cuda.data(), size_hfspec, cudaMemcpyHostToDevice));

    // cuFFT Plan
    std::cout << "Creating cuFFT plan..." << std::endl;
    cufftHandle plan;
    int rank = 1, n[1] = {Lraw};
    CUFFT_CHECK(cufftPlanMany(&plan, rank, n,
                              NULL, 1, Lraw,
                              NULL, 1, Lraw,
                              CUFFT_Z2Z, W));

    // FFT
    std::cout << "Executing FFT..." << std::endl;
    CUFFT_CHECK(cufftExecZ2Z(plan, D_inOut, D_inOut, CUFFT_FORWARD));

    // 频域乘法
    std::cout << "Executing multiply kernel..." << std::endl;
    int threadsPerBlock = 256;
    int blocksPerGrid = (W * Lraw + threadsPerBlock - 1) / threadsPerBlock;
    multiplyHfSpecKernel<<<blocksPerGrid, threadsPerBlock>>>(D_inOut, D_HfSpec, W, Lraw);
    CUDA_CHECK(cudaGetLastError());

    // IFFT
    std::cout << "Executing IFFT..." << std::endl;
    CUFFT_CHECK(cufftExecZ2Z(plan, D_inOut, D_inOut, CUFFT_INVERSE));

    // 后处理
    std::cout << "Executing normalize kernel..." << std::endl;
    blocksPerGrid = (W * M + threadsPerBlock - 1) / threadsPerBlock;
    normalizeAndCropKernel<<<blocksPerGrid, threadsPerBlock>>>(
        D_inOut, D_rcOut, W, Lraw, M, Lraw2M, scale);
    CUDA_CHECK(cudaGetLastError());

    // 同步
    CUDA_CHECK(cudaDeviceSynchronize());

    // Device → Host
    std::cout << "Copying results back..." << std::endl;
    std::vector<cuDoubleComplex> h_rc_cuda(W * M);
    CUDA_CHECK(cudaMemcpy(h_rc_cuda.data(), D_rcOut, size_out, cudaMemcpyDeviceToHost));

    // 验证输出
    std::cout << "\nOutput verification:" << std::endl;
    std::cout << "rc_out[0] = " << h_rc_cuda[0].x << " + " << h_rc_cuda[0].y << "i" << std::endl;
    std::cout << "rc_out[M] = " << h_rc_cuda[M].x << " + " << h_rc_cuda[M].y << "i" << std::endl;
    std::cout << "rc_out[W*M-1] = " << h_rc_cuda[W*M-1].x << " + " << h_rc_cuda[W*M-1].y << "i" << std::endl;

    // 清理
    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(D_inOut));
    CUDA_CHECK(cudaFree(D_HfSpec));
    CUDA_CHECK(cudaFree(D_rcOut));

    std::cout << "\nTest completed successfully!" << std::endl;
    return 0;
}
