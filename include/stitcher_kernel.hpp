#pragma once
#include <cstdint>

struct BeamDevParams {
  float x;
  float y;
  float z;
  float vN;
  float vE;
  float V_feiji;
  float sin_j;
  float cos_j;
  float min_fd;
  float delta_fd;
  uint64_t offset; // offset in floats into d_all_amps
  int flag;        // flight dir flag (matches CPU infer_flight_dir_flag + squint_side*4)
};

// 定义一个普通的 C++ 函数作为包装器

// 支持 texture object 的 kernel 启动器
#include <cuda_runtime.h>
void launchBuildMosaicKernel(
  float* d_amp_mosaic, uint16_t* d_which_beam,
  const float* d_all_amps,
  const float* d_grid_x, const float* d_grid_y,
  const void* d_beams,
  int nx, int ny, int B, int M, int nEff,
  float R_min, float R_bin, float lambda, float Height, int flag_base,
  bool useTexInterp = false // 是否用纹理内存
);
