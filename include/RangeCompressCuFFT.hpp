#pragma once

#include "DbsTypes.hpp"
#include <vector>
#include <complex>

// 使用 cuFFT 进行距离脉压
// 输入: in - BeamRaw<double> (W x Lraw)，HfSpec - 参考函数 (Lraw)
// 输出: rc_out - W x M 的复数矩阵（行主）
bool rangeCompressCuFFT(const Params &P,
                        const BeamRaw<double> &in,
                        const std::vector<std::complex<double>> &HfSpec,
                        std::vector<std::complex<double>> &rc_out);

// Device-oriented API: operate on device buffers (no host allocations inside).
// All device pointers are represented as `void*` to avoid forcing CUDA headers
// into every translation unit that includes this header. Cast to the proper
// device pointer types in the implementation.
// - `d_inOut` : device pointer to input buffer (cuDoubleComplex), size W*Lraw
// - `d_HfSpec`: device pointer to HfSpec (cuDoubleComplex), size Lraw
// - `d_rcOut` : device pointer to output (cuDoubleComplex), size W*M
// - `stream`  : pointer to `cudaStream_t` (pass as void*) or nullptr for default
bool rangeCompressCuFFT_device_buffers(const Params &P,
                                      void *d_inOut,
                                      const void *d_HfSpec,
                                      void *d_rcOut,
                                      int W, int Lraw, int M,
                                      void *stream);

// Convert device double-complex buffer to single-precision complex buffer
// - `d_in_double` : cuDoubleComplex* device input
// - `d_out_float` : cufftComplex* device output
// - `nelem`       : number of elements to convert
// - `stream`      : cudaStream_t as void*
bool convertDoubleToFloatDevice(const void *d_in_double,
                                 void *d_out_float,
                                 size_t nelem,
                                 void *stream);
