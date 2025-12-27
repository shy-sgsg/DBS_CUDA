#include "AzFftCuFFT.hpp"
#include <cufft.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Simple CUDA error check
#define CUDA_CHECK(call) do { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    std::fprintf(stderr, "[CUDA ERR] %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    return false; \
  } \
} while(0)

// cuFFT error check
#define CUFFT_CHECK(call) do { \
  cufftResult res = call; \
  if (res != CUFFT_SUCCESS) { \
    std::fprintf(stderr, "[CUFFT ERR] %s:%d code=%d\n", __FILE__, __LINE__, (int)res); \
    return false; \
  } \
} while(0)

// Kernel: circshift rows. Writes d_out[r*M + c] = d_in[((r - shift) mod W)*M + c]
__global__ static void circshift_rows_kernel(cufftComplex *d_in, cufftComplex *d_out,
                                            int W, int M, int shift)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (c >= M || r >= W) return;
  int src_r = r - shift;
  src_r %= W; if (src_r < 0) src_r += W;
  int dst_idx = r * M + c;
  int src_idx = src_r * M + c;
  d_out[dst_idx] = d_in[src_idx];
}

bool azFftShiftAndRecenterCuFFT(const Image2D<std::complex<float>> &rc,
                                 float PRF, float fd_ctr,
                                 Image2D<std::complex<float>> &azOut)
{
  if (rc.rows <= 0 || rc.cols <= 0) return false;
  const int W = rc.rows;
  const int M = rc.cols;

  // allocate output on host
  azOut = Image2D<std::complex<float>>(W, M);
  if (rc.size() != azOut.size()) return false;

  const size_t nelem = static_cast<size_t>(W) * static_cast<size_t>(M);
  const size_t bytes = nelem * sizeof(cufftComplex);

  // allocate device buffers
  cufftComplex *d_buf = nullptr;
  cufftComplex *d_tmp = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_buf), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_tmp), bytes));

  // copy input to device
  CUDA_CHECK(cudaMemcpy(d_buf, rc.buf.data(), bytes, cudaMemcpyHostToDevice));

  // prepare plan
  cufftHandle plan;
  int n[1] = { W };
  int inembed[1] = { W };
  int onembed[1] = { W };
  int istride = M;
  int ostride = M;
  int idist = 1;
  int odist = 1;
  int batch = M;

  CUFFT_CHECK(cufftPlanMany(&plan, 1, n,
                            inembed, istride, idist,
                            onembed, ostride, odist,
                            CUFFT_C2C, batch));

  // execute forward FFT in-place
  CUFFT_CHECK(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_buf),
                           reinterpret_cast<cufftComplex*>(d_buf), CUFFT_FORWARD));

  CUFFT_CHECK(cufftDestroy(plan));

  // compute combined shift: fftshift W/2 then recentre
  int k0 = 0;
  if (W > 1) {
    const float df = PRF / float(W - 1);
    float pos = (fd_ctr + 0.5f * PRF) / df;
    // clamp
    if (pos < 0.0f) pos = 0.0f;
    if (pos > (float)(W - 1)) pos = (float)(W - 1);
    k0 = int((pos >= 0.0f) ? pos + 0.5f : pos - 0.5f);
  }
  const int center0 = int(std::ceil(W / 2.0)) - 1;
  const int shift_recentre = center0 - k0;
  const int shift_fftshift = W / 2;
  int total_shift = (shift_fftshift + shift_recentre) % W;
  if (total_shift < 0) total_shift += W;

  // circshift rows using kernel: write into d_tmp
  const int TX = 32, TY = 8;
  dim3 block(TX, TY);
  dim3 grid((M + TX - 1) / TX, (W + TY - 1) / TY);
  circshift_rows_kernel<<<grid, block>>>(d_buf, d_tmp, W, M, total_shift);
  CUDA_CHECK(cudaGetLastError());

  // copy back
  CUDA_CHECK(cudaMemcpy(azOut.buf.data(), d_tmp, bytes, cudaMemcpyDeviceToHost));

  cudaFree(d_buf);
  cudaFree(d_tmp);

  return true;
}
