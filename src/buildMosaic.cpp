#include "DbsStitcher.hpp"
#include <numeric>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm> // std::min/std::max
#include "rotation_xy.hpp"
#include "stitcher_kernel.hpp"
#include <cuda_runtime.h>
#include <iostream>


// 简易 clamp（C++11 里没有 std::clamp）
static inline int clamp_int(int v, int lo, int hi)
{
  return v < lo ? lo : (v > hi ? hi : v);
}
static inline float clamp_float(float v, float lo, float hi)
{
  return v < lo ? lo : (v > hi ? hi : v);
}
// 光速（避免外部依赖）
static inline float c0f_local() { return 299792458.0f; }

bool DbsStitcher::buildMosaic(const Params &P,
                              const RDData &RD,
                              const MetaPack &meta,
                              const Grid &grid,
                              Mosaic &mosaic)
{
  const int B = (int)meta.beams.size();                   // bowei_num
  const int M = (RD.amp.empty() ? 0 : RD.amp[0].cols);    // range_samp_used
  const int nEff = (RD.amp.empty() ? 0 : RD.amp[0].rows); // f_youxiao
  const int nx = (int)grid.x.size();
  const int ny = (int)grid.y.size();
  if (B <= 0 || M <= 0 || nEff <= 0 || nx <= 0 || ny <= 0)
    return false;

  const float R_bin = c0f_local() / (2.0f * (float)P.fs_hz);
  const float lambda = c0f_local() / (float)P.fc_hz;
  const float R_min = (float)P.Rmin_m;

  // —— Height：统一高度（mean(meta.z) - P.mean_ground_h）
  double zmean = 0.0;
  for (int i = 0; i < B; ++i)
    zmean += meta.beams[(size_t)i].z;
  zmean = (B > 0) ? (zmean / B) : 0.0;
  const float Height = (float)(zmean - P.mean_ground_h);

  // —— 每波位 fd 轴 min/max/delta（按波位并行）
  std::vector<float> min_RD_y(B), max_RD_y(B), delta_RD_y(B);
  for (int b = 0; b < B; ++b)
  {
    const std::vector<float> &frow = RD.fd_axis[b]; // size=nEff
    float mn = std::numeric_limits<float>::infinity();
    float mx = -std::numeric_limits<float>::infinity();
    for (size_t k = 0; k < frow.size(); ++k)
    {
      const float v = frow[k];
      mn = v < mn ? v : mn;
      mx = v > mx ? v : mx;
    }
    min_RD_y[b] = (frow.empty() ? 0.0f : mn);
    max_RD_y[b] = (frow.empty() ? 0.0f : mx);
    delta_RD_y[b] = (nEff > 1) ? (max_RD_y[b] - min_RD_y[b]) / (float)(nEff - 1) : 1e-9f;
  }

  // —— 速度与航迹角（按波位并行）
  std::vector<float> vN(B), vE(B), V_feiji(B), sin_jiaodu(B), cos_jiaodu(B);
  int squint_side = P.squint_side; // 0 for left, 1 for right
  for (int b = 0; b < B; ++b)
  {
    vN[b] = meta.beams[b].vN;
    vE[b] = meta.beams[b].vE;
    V_feiji[b] = std::sqrt(vN[b] * vN[b] + vE[b] * vE[b]);
    const float denom = (std::fabs(vE[b]) > 1e-12f) ? std::fabs(vE[b]) : 1e-12f;
    const float jiaodu = std::atan(std::fabs(vN[b] / denom));
    sin_jiaodu[b] = std::sin(jiaodu);
    cos_jiaodu[b] = std::cos(jiaodu);
  }

  // —— 输出缓存
  mosaic.amp = Image2D<float>(ny, nx);
  mosaic.which_beam = Image2D<uint16_t>(ny, nx);
  mosaic.r_save = Image2D<float>(ny, nx);
  mosaic.fd_save = Image2D<float>(ny, nx);
  mosaic.x_rel_save = Image2D<float>(ny, nx);
  mosaic.y_rel_save = Image2D<float>(ny, nx);

  // 清零（可并行按元素写）
  const size_t Npix = (size_t)nx * ny;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < Npix; ++idx)
  {
    mosaic.amp.buf[idx] = 0.0f;
    mosaic.which_beam.buf[idx] = (uint16_t)0;
    mosaic.r_save.buf[idx] = 0.0f;
    mosaic.fd_save.buf[idx] = 0.0f;
    mosaic.x_rel_save.buf[idx] = 0.0f;
    mosaic.y_rel_save.buf[idx] = 0.0f;
  }

  // —— 旋转到飞机坐标
  // const int flag = P.flight_dir_flag;
  int flag = infer_flight_dir_flag(vN, vE); // 1/2/3/4
  DBG("Inferred flight_dir_flag = " << flag);

  flag = flag + squint_side * 4;

  // —— 主循环：逐像素 (i, j) ；按列并行，列内顺序，避免 prev_pos 跨线程
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < nx; ++i)
  {
    const float x_label = (float)grid.x[i];

    for (int j = 0; j < ny; ++j)
    {
      const float y_label = (float)grid.y[j];

      // ===== 粗投影：使用第1波位判距 =====
      float tmp_x0 = x_label - meta.beams[0].x;
      float tmp_y0 = y_label - meta.beams[0].y;

      float xl0, yl0;
      {
        const float c = cos_jiaodu[0], s = sin_jiaodu[0];
        rotation_xy_inv(tmp_x0, tmp_y0, flag, c, s, xl0, yl0);
      }

      const float R_geo = std::sqrt(xl0 * xl0 + yl0 * yl0 + Height * Height);
      const float R_idx0 = ((R_geo - R_min) / R_bin) + 1.0f;

      // 上一行 position（邻域约束）—— 同一列内顺序访问，线程安全
      uint16_t prev_pos = 0;
      if (j > 0)
        prev_pos = mosaic.which_beam.at(j - 1, i);

      // ===== 遍历每个波位（顺序即可；对每个像素内部不存在共享写） =====
      for (int n = 0; n < B; ++n)
      {
        // 邻域约束
        bool neighbor_ok = (j == 0) ? true : (prev_pos == 0 || std::abs(int(prev_pos) - (n + 1)) <= 15);
        if (!neighbor_ok)
          continue;

        if (!(R_idx0 > 1.0f && R_idx0 < (float)(M - 1)))
          continue;

        // === 精确投影到第 n 波位 ===
        const float tmp_x = x_label - meta.beams[n].x;
        const float tmp_y = y_label - meta.beams[n].y;

        float x_tmp, y_tmp;
        {
          const float c = cos_jiaodu[n], s = sin_jiaodu[n];
          rotation_xy_inv(tmp_x, tmp_y, flag, c, s, x_tmp, y_tmp);
        }

        const float R = std::sqrt(x_tmp * x_tmp + y_tmp * y_tmp + Height * Height);
        const float fd = (R > 0.0f) ? (2.0f * V_feiji[n] * x_tmp / (R * lambda)) : 0.0f;

        // —— 浮点索引（MATLAB 1-based）
        float R_idx = ((R - R_min) / R_bin) + 1.0f;
        float fd_idx = ((fd - min_RD_y[n]) /
                        (delta_RD_y[n] > 1e-9f ? delta_RD_y[n] : 1e-9f)) +
                       1.0f;

        // —— 有效性判定
        if (!(R_idx > 1.0f && R_idx < (float)(M - 1)))
          continue;
        if (!(fd_idx > 1.0f && fd_idx < (float)(nEff)))
          continue;

        // —— 最近/双线性插值
        int R_lo = (int)std::floor(R_idx);
        int R_hi = (int)std::ceil(R_idx);
        int fd_lo = (int)std::floor(fd_idx);
        int fd_hi = (int)std::ceil(fd_idx);

        R_lo = clamp_int(R_lo, 1, M - 1);
        R_hi = clamp_int(R_hi, 1, M - 1);
        fd_lo = clamp_int(fd_lo, 1, nEff - 1);
        fd_hi = clamp_int(fd_hi, 1, nEff - 1);

        const Image2D<float> &A = RD.amp[n]; // nEff x M（幅度，已为正）

        float amp_tmp = 0.0f;
        int R_pick = R_lo, fd_pick = fd_lo;

        if (P.interp_mode == 1)
        {
          // 最近点
          R_pick = ((R_hi - R_idx) > (R_idx - R_lo)) ? R_lo : R_hi;
          fd_pick = ((fd_hi - fd_idx) > (fd_idx - fd_lo)) ? fd_lo : fd_hi;
          amp_tmp = A.at(fd_pick - 1, R_pick - 1);
        }
        else
        {
          // 双线性（按原逻辑对幅度线性组合）
          const float a11 = std::fabs(A.at(fd_lo - 1, R_lo - 1));
          const float a12 = std::fabs(A.at(fd_lo - 1, R_hi - 1));
          const float a21 = std::fabs(A.at(fd_hi - 1, R_lo - 1));
          const float a22 = std::fabs(A.at(fd_hi - 1, R_hi - 1));

          const float wR1 = (R_hi - R_idx), wR2 = (R_idx - R_lo);
          const float wF1 = (fd_hi - fd_idx), wF2 = (fd_idx - fd_lo);

          amp_tmp = a11 * wR1 * wF1 + a12 * wR2 * wF1 + a21 * wR1 * wF2 + a22 * wR2 * wF2;
          const float denom = (float)((R_hi - R_lo) * (fd_hi - fd_lo));
          amp_tmp = (denom != 0.0f) ? (amp_tmp / denom) : a11;

          // 最近邻用于保存 R/fd 的物理值
          R_pick = ((R_hi - R_idx) > (R_idx - R_lo)) ? R_lo : R_hi;
          fd_pick = ((fd_hi - fd_idx) > (fd_idx - fd_lo)) ? fd_lo : fd_hi;
        }

        // —— 更新像素（取最大幅值）：同一像素仅当前线程写入，线程安全
        float &cur = mosaic.amp.at(j, i);
        if (std::fabs(cur) < std::fabs(amp_tmp))
        {
          cur = amp_tmp;
          mosaic.which_beam.at(j, i) = (uint16_t)(n + 1); // 1-based

          mosaic.x_rel_save.at(j, i) = x_tmp;
          mosaic.y_rel_save.at(j, i) = y_tmp;

          // 最近原则保存 R/fd 的物理值
          mosaic.r_save.at(j, i) = (float)(R_pick - 1) * R_bin + R_min;
          mosaic.fd_save.at(j, i) = (float)(fd_pick - 1) * delta_RD_y[n] + min_RD_y[n];
        }
      } // for n
    } // for j
  } // for i

  return true;
}

// GPU-accelerated variant: batched queries + device-resident amplitude buffers.
bool DbsStitcher::buildMosaicGPU(const Params &P,
                                 const RDData &RD,
                                 const MetaPack &meta,
                                 const Grid &grid,
                                 Mosaic &mosaic)
{
  const int B = (int)meta.beams.size();
  const int M = (RD.amp.empty() ? 0 : RD.amp[0].cols);
  const int nEff = (RD.amp.empty() ? 0 : RD.amp[0].rows);
  const int nx = (int)grid.x.size();
  const int ny = (int)grid.y.size();
  if (B <= 0 || M <= 0 || nEff <= 0 || nx <= 0 || ny <= 0)
    return false;

  const float R_bin = c0f_local() / (2.0f * (float)P.fs_hz);
  const float lambda = c0f_local() / (float)P.fc_hz;
  const float R_min = (float)P.Rmin_m;

  double zmean = 0.0;
  for (int i = 0; i < B; ++i) zmean += meta.beams[(size_t)i].z;
  zmean = (B > 0) ? (zmean / B) : 0.0;
  const float Height = (float)(zmean - P.mean_ground_h);

  std::vector<float> min_RD_y(B), max_RD_y(B), delta_RD_y(B);
  for (int b = 0; b < B; ++b)
  {
    const std::vector<float> &frow = RD.fd_axis[b];
    float mn = std::numeric_limits<float>::infinity();
    float mx = -std::numeric_limits<float>::infinity();
    for (size_t k = 0; k < frow.size(); ++k)
    {
      const float v = frow[k];
      mn = v < mn ? v : mn;
      mx = v > mx ? v : mx;
    }
    min_RD_y[b] = (frow.empty() ? 0.0f : mn);
    max_RD_y[b] = (frow.empty() ? 0.0f : mx);
    delta_RD_y[b] = (nEff > 1) ? (max_RD_y[b] - min_RD_y[b]) / (float)(nEff - 1) : 1e-9f;
  }

  std::vector<float> vN(B), vE(B), V_feiji(B), sin_jiaodu(B), cos_jiaodu(B);
  int squint_side = P.squint_side;
  for (int b = 0; b < B; ++b)
  {
    vN[b] = meta.beams[b].vN;
    vE[b] = meta.beams[b].vE;
    V_feiji[b] = std::sqrt(vN[b] * vN[b] + vE[b] * vE[b]);
    const float denom = (std::fabs(vE[b]) > 1e-12f) ? std::fabs(vE[b]) : 1e-12f;
    const float jiaodu = std::atan(std::fabs(vN[b] / denom));
    sin_jiaodu[b] = std::sin(jiaodu);
    cos_jiaodu[b] = std::cos(jiaodu);
  }

  // output
  mosaic.amp = Image2D<float>(ny, nx);
  mosaic.which_beam = Image2D<uint16_t>(ny, nx);
  mosaic.r_save = Image2D<float>(ny, nx);
  mosaic.fd_save = Image2D<float>(ny, nx);
  mosaic.x_rel_save = Image2D<float>(ny, nx);
  mosaic.y_rel_save = Image2D<float>(ny, nx);

  const size_t Npix = (size_t)nx * ny;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < Npix; ++idx)
  {
    mosaic.amp.buf[idx] = 0.0f;
    mosaic.which_beam.buf[idx] = (uint16_t)0;
    mosaic.r_save.buf[idx] = 0.0f;
    mosaic.fd_save.buf[idx] = 0.0f;
    mosaic.x_rel_save.buf[idx] = 0.0f;
    mosaic.y_rel_save.buf[idx] = 0.0f;
  }

  int flag = infer_flight_dir_flag(vN, vE);
  flag = flag + squint_side * 4;

  // Full GPU approach: pack all beam amplitude matrices into one contiguous buffer
  // and run a 2D kernel where each thread computes one output pixel.

  // 1) Prepare BeamDevParams and combined amplitude buffer
  std::vector<BeamDevParams> h_beams(B);
  std::vector<float> h_all_amps;
  h_all_amps.reserve((size_t)B * (size_t)M * (size_t)nEff);

  for (int b = 0; b < B; ++b)
  {
    const auto &mb = meta.beams[b];
    float vN_b = mb.vN;
    float vE_b = mb.vE;
    float V = std::sqrt(vN_b * vN_b + vE_b * vE_b);
    float denom_local = (std::fabs(vE_b) > 1e-12f) ? std::fabs(vE_b) : 1e-12f;
    float angle = std::atan(std::fabs(vN_b / denom_local));

    BeamDevParams bp;
    bp.x = (float)mb.x;
    bp.y = (float)mb.y;
    bp.z = (float)mb.z;
    bp.vN = vN_b;
    bp.vE = vE_b;
    bp.V_feiji = V;
    bp.sin_j = std::sin(angle);
    bp.cos_j = std::cos(angle);
    bp.min_fd = RD.fd_axis[b].empty() ? 0.0f : RD.fd_axis[b].front();
    bp.delta_fd = (nEff > 1) ? (RD.fd_axis[b].back() - RD.fd_axis[b].front()) / (float)(nEff - 1) : 1e-9f;
    bp.offset = (uint64_t)h_all_amps.size();

    h_beams[b] = bp;

    // append amplitude matrix (row-major: f * M + r)
    const Image2D<float> &A = RD.amp[b];
    h_all_amps.insert(h_all_amps.end(), A.buf.begin(), A.buf.end());
  }

  // 2) allocate device memory
  float *d_all_amps = nullptr;
  BeamDevParams *d_beams = nullptr;
  float *d_grid_x = nullptr;
  float *d_grid_y = nullptr;
  float *d_amp_mosaic = nullptr;
  uint16_t *d_which_beam = nullptr;

  size_t all_amps_bytes = h_all_amps.size() * sizeof(float);
  cudaMalloc((void**)&d_all_amps, all_amps_bytes);
  cudaMalloc((void**)&d_beams, sizeof(BeamDevParams) * (size_t)B);
  cudaMalloc((void**)&d_grid_x, sizeof(float) * (size_t)nx);
  cudaMalloc((void**)&d_grid_y, sizeof(float) * (size_t)ny);
  cudaMalloc((void**)&d_amp_mosaic, sizeof(float) * (size_t)nx * (size_t)ny);
  cudaMalloc((void**)&d_which_beam, sizeof(uint16_t) * (size_t)nx * (size_t)ny);

  // 3) copy to device
  cudaMemcpy(d_all_amps, h_all_amps.data(), all_amps_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_beams, h_beams.data(), sizeof(BeamDevParams) * (size_t)B, cudaMemcpyHostToDevice);
  cudaMemcpy(d_grid_x, grid.x.data(), sizeof(float) * (size_t)nx, cudaMemcpyHostToDevice);
  cudaMemcpy(d_grid_y, grid.y.data(), sizeof(float) * (size_t)ny, cudaMemcpyHostToDevice);

  // 4) launch kernel
  dim3 block(16, 16);
  dim3 gsize((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

  int flag_base = flag;

  launchBuildMosaicKernel(
      d_amp_mosaic, d_which_beam, d_all_amps,
      d_grid_x, d_grid_y, d_beams,
      nx, ny, B, M, nEff,
      (float)P.Rmin_m, R_bin, lambda, Height, flag_base
  );

  cudaDeviceSynchronize();

  // 5) copy results back
  mosaic.amp = Image2D<float>(ny, nx);
  mosaic.which_beam = Image2D<uint16_t>(ny, nx);
  cudaMemcpy(mosaic.amp.buf.data(), d_amp_mosaic, sizeof(float) * (size_t)nx * (size_t)ny, cudaMemcpyDeviceToHost);
  cudaMemcpy(mosaic.which_beam.buf.data(), d_which_beam, sizeof(uint16_t) * (size_t)nx * (size_t)ny, cudaMemcpyDeviceToHost);

  // 6) cleanup
  cudaFree(d_all_amps);
  cudaFree(d_beams);
  cudaFree(d_grid_x);
  cudaFree(d_grid_y);
  cudaFree(d_amp_mosaic);
  cudaFree(d_which_beam);

  return true;
}