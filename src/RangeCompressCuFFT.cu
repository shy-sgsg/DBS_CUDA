#include "RangeCompressCuFFT.hpp"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <iostream>
#include <cstdlib>

// CUDA 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            return false; \
        } \
    } while (0)

// cuFFT 错误检查宏
#define CUFFT_CHECK(call) \
    do { \
        cufftResult status = call; \
        if (status != CUFFT_SUCCESS) { \
            std::cerr << "cuFFT Error at " << __FILE__ << ":" << __LINE__ \
                      << ": cuFFT status code " << status << std::endl; \
            return false; \
        } \
    } while (0)

// --------------------------------------------------
// Kernel: 频域乘法 (X *= HfSpec[f])
// --------------------------------------------------
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

// --------------------------------------------------
// Kernel: IFFT 后处理（归一化 + 裁剪 M 距离单元）
// --------------------------------------------------
__global__ void normalizeAndCropKernel(const cuDoubleComplex* D_X,
                                       cuDoubleComplex* D_rcOut,
                                       int W, int Lraw, int M,
                                       int Lraw2M,
                                       double scale)
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

// --------------------------------------------------
// 主函数：cuFFT 距离脉压
// --------------------------------------------------
bool rangeCompressCuFFT(const Params &P,
                        const BeamRaw<double> &in,
                        const std::vector<std::complex<double>> &HfSpec,
                        std::vector<std::complex<double>> &rc_out)
{
    const int W = in.W;
    const int Lraw = in.Lraw;
    const int M = P.range_samp_used;

    if ((int)HfSpec.size() != Lraw || (int)in.s.size() != W * Lraw) {
        std::cerr << "Size mismatch in rangeCompressCuFFT" << std::endl;
        return false;
    }

    if (Lraw % M != 0) {
        std::cerr << "Lraw must be an integer multiple of M" << std::endl;
        return false;
    }

    size_t size_in_bytes = (size_t)W * Lraw * sizeof(cuDoubleComplex);
    size_t size_out_bytes = (size_t)W * M * sizeof(cuDoubleComplex);
    size_t size_hfspec_bytes = (size_t)Lraw * sizeof(cuDoubleComplex);

    // --- 1. Device 内存分配 ---
    cuDoubleComplex *D_inOut = nullptr;
    cuDoubleComplex *D_HfSpec = nullptr;
    cuDoubleComplex *D_rcOut = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&D_inOut, size_in_bytes));
    CUDA_CHECK(cudaMalloc((void**)&D_HfSpec, size_hfspec_bytes));
    CUDA_CHECK(cudaMalloc((void**)&D_rcOut, size_out_bytes));

    // --- 2. Host -> Device 数据传输 ---
    // 将输入数据转换为 cuDoubleComplex
    std::vector<cuDoubleComplex> h_inData(W * Lraw);
    for (int i = 0; i < W * Lraw; ++i) {
        h_inData[i] = make_cuDoubleComplex(in.s[i].real(), in.s[i].imag());
    }

    std::vector<cuDoubleComplex> h_HfSpec(Lraw);
    for (int i = 0; i < Lraw; ++i) {
        h_HfSpec[i] = make_cuDoubleComplex(HfSpec[i].real(), HfSpec[i].imag());
    }

    CUDA_CHECK(cudaMemcpy(D_inOut, h_inData.data(), size_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(D_HfSpec, h_HfSpec.data(), size_hfspec_bytes, cudaMemcpyHostToDevice));

    // --- 3. cuFFT Plan 创建 ---
    cufftHandle plan;
    int rank = 1;
    int n[1] = {Lraw};
    int howmany = W;

    CUFFT_CHECK(cufftPlanMany(&plan, rank, n,
                              NULL, 1, Lraw,
                              NULL, 1, Lraw,
                              CUFFT_Z2Z,
                              howmany));

    // --- 4. 前向 FFT ---
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUFFT_CHECK(cufftExecZ2Z(plan, D_inOut, D_inOut, CUFFT_FORWARD));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "cuFFT Forward FFT execution time: " << milliseconds << " ms" << std::endl;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // --- 5. 频域乘法 Kernel ---
    int threadsPerBlock = 256;
    int blocksPerGrid = (W * Lraw + threadsPerBlock - 1) / threadsPerBlock;
    multiplyHfSpecKernel<<<blocksPerGrid, threadsPerBlock>>>(D_inOut, D_HfSpec, W, Lraw);
    CUDA_CHECK(cudaGetLastError());

    // --- 6. 逆向 FFT ---
    CUFFT_CHECK(cufftExecZ2Z(plan, D_inOut, D_inOut, CUFFT_INVERSE));

    // --- 7. 后处理 Kernel（归一化 + 裁剪） ---
    const int Lraw2M = Lraw / M;
    const double scale = 1.0 / double(Lraw);
    blocksPerGrid = (W * M + threadsPerBlock - 1) / threadsPerBlock;
    normalizeAndCropKernel<<<blocksPerGrid, threadsPerBlock>>>(
        D_inOut, D_rcOut, W, Lraw, M, Lraw2M, scale);
    CUDA_CHECK(cudaGetLastError());

    // 等待所有操作完成
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- 8. Device -> Host 结果传输 ---
    std::vector<cuDoubleComplex> h_rcOut(W * M);
    CUDA_CHECK(cudaMemcpy(h_rcOut.data(), D_rcOut, size_out_bytes, cudaMemcpyDeviceToHost));

    // 转换回 std::complex<double>
    rc_out.resize(W * M);
    for (int i = 0; i < W * M; ++i) {
        rc_out[i] = std::complex<double>(h_rcOut[i].x, h_rcOut[i].y);
    }

    // --- 9. 清理 ---
    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(D_inOut));
    CUDA_CHECK(cudaFree(D_HfSpec));
    CUDA_CHECK(cudaFree(D_rcOut));

    return true;
}

// --------------------------------------------------
// Kernel: convert cuDoubleComplex -> cufftComplex (double->float)
// --------------------------------------------------
__global__ void convertDoubleToFloatKernel(const cuDoubleComplex* src,
                                           cufftComplex* dst,
                                           size_t nelem)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nelem) {
        dst[i].x = static_cast<float>(src[i].x);
        dst[i].y = static_cast<float>(src[i].y);
    }
}

// Device-oriented implementation that operates on preallocated device buffers
bool rangeCompressCuFFT_device_buffers(const Params &P,
                                      void *d_inOut_void,
                                      const void *d_HfSpec_void,
                                      void *d_rcOut_void,
                                      int W, int Lraw, int M,
                                      void *stream_void)
{
    cuDoubleComplex *D_inOut = reinterpret_cast<cuDoubleComplex *>(d_inOut_void);
    const cuDoubleComplex *D_HfSpec = reinterpret_cast<const cuDoubleComplex *>(d_HfSpec_void);
    cuDoubleComplex *D_rcOut = reinterpret_cast<cuDoubleComplex *>(d_rcOut_void);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);

    if (Lraw % M != 0) {
        std::cerr << "Lraw must be an integer multiple of M" << std::endl;
        return false;
    }

    // create cuFFT plan and bind to stream
    cufftHandle plan;
    int rank = 1;
    int n[1] = { Lraw };
    int howmany = W;
    CUFFT_CHECK(cufftPlanMany(&plan, rank, n,
                              NULL, 1, Lraw,
                              NULL, 1, Lraw,
                              CUFFT_Z2Z,
                              howmany));

    if (stream)
        CUFFT_CHECK(cufftSetStream(plan, stream));

    // forward FFT
    CUFFT_CHECK(cufftExecZ2Z(plan, D_inOut, D_inOut, CUFFT_FORWARD));

    // multiply in frequency
    int threadsPerBlock = 256;
    int blocksPerGrid = (W * Lraw + threadsPerBlock - 1) / threadsPerBlock;
    multiplyHfSpecKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(D_inOut, D_HfSpec, W, Lraw);
    CUDA_CHECK(cudaGetLastError());

    // inverse FFT
    CUFFT_CHECK(cufftExecZ2Z(plan, D_inOut, D_inOut, CUFFT_INVERSE));

    // normalize and crop
    const int Lraw2M = Lraw / M;
    const double scale = 1.0 / double(Lraw);
    blocksPerGrid = (W * M + threadsPerBlock - 1) / threadsPerBlock;
    normalizeAndCropKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(D_inOut, D_rcOut, W, Lraw, M, Lraw2M, scale);
    CUDA_CHECK(cudaGetLastError());

    // ensure completion on requested stream
    if (stream)
        CUDA_CHECK(cudaStreamSynchronize(stream));
    else
        CUDA_CHECK(cudaDeviceSynchronize());

    CUFFT_CHECK(cufftDestroy(plan));

    return true;
}

bool convertDoubleToFloatDevice(const void *d_in_double_void,
                                 void *d_out_float_void,
                                 size_t nelem,
                                 void *stream_void)
{
    const cuDoubleComplex *d_in_double = reinterpret_cast<const cuDoubleComplex *>(d_in_double_void);
    cufftComplex *d_out_float = reinterpret_cast<cufftComplex *>(d_out_float_void);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);

    const int threadsPerBlock = 256;
    const int blocks = (int)((nelem + threadsPerBlock - 1) / threadsPerBlock);
    convertDoubleToFloatKernel<<<blocks, threadsPerBlock, 0, stream>>>(d_in_double, d_out_float, nelem);
    CUDA_CHECK(cudaGetLastError());

    if (stream)
        CUDA_CHECK(cudaStreamSynchronize(stream));
    else
        CUDA_CHECK(cudaDeviceSynchronize());

    return true;
}
