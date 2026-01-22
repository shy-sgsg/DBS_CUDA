// GPU-accelerated azimuth FFT + shift/recentre using cuFFT
#pragma once
#include "DbsTypes.hpp"

// Perform batched FFT along rows (pulse axis) for an input WxM matrix `rc`.
// The output `azOut` will have the same size. This function performs:
//   1) batched FFT (length W) over each column (M batches),
//   2) fftshift by W/2,
//   3) recentre according to `fd_ctr` and `PRF`.
// All heavy work is done on the GPU (cuFFT + a small kernel for circshift).
bool azFftShiftAndRecenterCuFFT(const Image2D<std::complex<float>> &rc,
                                 float PRF, float fd_ctr,
                                 Image2D<std::complex<float>> &azOut);

// Device-oriented API: operate on device buffers (single-precision complex)
// - `d_rcFloat` : device pointer to input data (cufftComplex), size W*M
// - `W`, `M`    : dimensions
// - `PRF`, `fd_ctr` : parameters for recentring
// - `d_azOut`   : device pointer for output (cufftComplex), may be same as input
// - `stream`    : cudaStream_t as void* or nullptr
bool azFftShiftAndRecenterCuFFT_device(void *d_rcFloat,
                                       int W, int M,
                                       float PRF, float fd_ctr,
                                       void *d_azOut,
                                       void *stream);
