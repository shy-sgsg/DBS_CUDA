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
