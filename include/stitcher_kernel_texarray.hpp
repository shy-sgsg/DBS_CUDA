#pragma once
#include <cuda_runtime.h>

// 纹理对象数组 kernel 启动
__global__ void buildMosaicFullKernelTexArray(
    float* d_amp_mosaic,
    uint16_t* d_which_beam,
    const float* d_all_amps,
    const float* d_grid_x,
    const float* d_grid_y,
    const BeamDevParams* d_beams,
    int nx, int ny, int B, int M, int nEff,
    float R_min, float R_bin, float lambda, float Height, int flag_base,
    cudaTextureObject_t* texObjArr // B 个纹理对象
);

// host 启动器
void launchBuildMosaicKernelTexArray(
    float* d_amp_mosaic, uint16_t* d_which_beam,
    const float* d_all_amps,
    const float* d_grid_x, const float* d_grid_y,
    const void* d_beams,
    int nx, int ny, int B, int M, int nEff,
    float R_min, float R_bin, float lambda, float Height, int flag_base
);
