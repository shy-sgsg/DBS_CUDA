#include "stitcher_kernel.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

__global__ void buildMosaicFullKernel(
    float* d_amp_mosaic,
    uint16_t* d_which_beam,
    const float* d_all_amps,
    const float* d_grid_x,
    const float* d_grid_y,
    const BeamDevParams* d_beams,
    int nx, int ny, int B, int M, int nEff,
    float R_min, float R_bin, float lambda, float Height, int flag_base)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    float x_label = d_grid_x[i];
    float y_label = d_grid_y[j];

    float best_amp = 0.0f;
    uint16_t best_beam_idx = 0;

    for (int b = 0; b < B; ++b) {
        BeamDevParams bp = d_beams[b];

        // relative coords
        float dx = x_label - bp.x;
        float dy = y_label - bp.y;

        // rotation using the same "rotation_xy_inv" convention from CPU
        float c = bp.cos_j;
        float s = bp.sin_j;
        int flag = bp.flag; // per-beam flag (matches CPU)

        float x_tmp = 0.0f, y_tmp = 0.0f;
        switch (flag)
        {
        case 1:
            x_tmp = -dx * c - dy * s;
            y_tmp = -dx * s + dy * c;
            break;
        case 2:
            x_tmp = dx * c + dy * s;
            y_tmp = dx * s - dy * c;
            break;
        case 3:
            x_tmp = -dx * c + dy * s;
            y_tmp = dx * s + dy * c;
            break;
        case 4:
            x_tmp = dx * c - dy * s;
            y_tmp = -dx * s - dy * c;
            break;
        case 6:
            x_tmp = dx * c + dy * s;
            y_tmp = -dx * s + dy * c;
            break;
        case 5:
            x_tmp = -dx * c - dy * s;
            y_tmp = dx * s - dy * c;
            break;
        case 8:
            x_tmp = dx * c - dy * s;
            y_tmp = dx * s + dy * c;
            break;
        case 7:
            x_tmp = -dx * c + dy * s;
            y_tmp = -dx * s - dy * c;
            break;
        default:
            x_tmp = dx;
            y_tmp = dy;
            break;
        }

        float R = sqrtf(x_tmp * x_tmp + y_tmp * y_tmp + Height * Height);
        float fd = (R > 1e-6f) ? (2.0f * bp.V_feiji * x_tmp / (R * lambda)) : 0.0f;

        float R_idx = (R - R_min) / R_bin;
        float fd_idx = (fd - bp.min_fd) / (bp.delta_fd > 1e-9f ? bp.delta_fd : 1e-9f);

        if (R_idx >= 0.0f && R_idx < (float)(M - 1) && fd_idx >= 0.0f && fd_idx < (float)(nEff - 1)) {
            int r0 = (int)R_idx;
            int f0 = (int)fd_idx;
            float wr = R_idx - r0;
            float wf = fd_idx - f0;

            const float* b_data = d_all_amps + (size_t)bp.offset;

            float v00 = b_data[f0 * M + r0];
            float v01 = b_data[f0 * M + (r0 + 1)];
            float v10 = b_data[(f0 + 1) * M + r0];
            float v11 = b_data[(f0 + 1) * M + (r0 + 1)];

            float amp_val = (1.0f - wf) * (1.0f - wr) * v00 + (1.0f - wf) * wr * v01
                            + wf * (1.0f - wr) * v10 + wf * wr * v11;

            if (fabsf(amp_val) > fabsf(best_amp)) {
                best_amp = amp_val;
                best_beam_idx = (uint16_t)(b + 1);
            }
        }
    }

    int out_idx = j * nx + i;
    d_amp_mosaic[out_idx] = best_amp;
    d_which_beam[out_idx] = best_beam_idx;
}

// 包装函数实现
void launchBuildMosaicKernel(
    float* d_amp_mosaic, uint16_t* d_which_beam, 
    const float* d_all_amps, 
    const float* d_grid_x, const float* d_grid_y,
    const void* d_beams, // 使用 void* 或具体的结构体指针
    int nx, int ny, int B, int M, int nEff,
    float R_min, float R_bin, float lambda, float Height, int flag_base)    
{
    dim3 block(16, 16);
    dim3 gsize((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // 只有在 .cu 文件里才能写 <<< >>>
    buildMosaicFullKernel<<<gsize, block>>>(
        d_amp_mosaic, d_which_beam, d_all_amps,
        d_grid_x, d_grid_y, (const BeamDevParams*)d_beams,
        nx, ny, B, M, nEff,
        R_min, R_bin, lambda, Height, flag_base
    );
    
    // 检查是否启动成功
    cudaDeviceSynchronize();
}