#include "stitcher_kernel.hpp"
#include "stitcher_kernel_texarray.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

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
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 0 || i >= nx || j < 0 || j >= ny) return;
    float x_label = d_grid_x[i];
    float y_label = d_grid_y[j];
    float best_amp = 0.0f;
    uint16_t best_beam_idx = 0;
    for (int b = 0; b < B; ++b) {
        BeamDevParams bp = d_beams[b];
        float dx = x_label - bp.x;
        float dy = y_label - bp.y;
        float c = bp.cos_j;
        float s = bp.sin_j;
        int flag = bp.flag;
        float x_tmp = 0.0f, y_tmp = 0.0f;
        switch (flag) {
        case 1: x_tmp = -dx * c - dy * s; y_tmp = -dx * s + dy * c; break;
        case 2: x_tmp = dx * c + dy * s; y_tmp = dx * s - dy * c; break;
        case 3: x_tmp = -dx * c + dy * s; y_tmp = dx * s + dy * c; break;
        case 4: x_tmp = dx * c - dy * s; y_tmp = -dx * s - dy * c; break;
        case 6: x_tmp = dx * c + dy * s; y_tmp = -dx * s + dy * c; break;
        case 5: x_tmp = -dx * c - dy * s; y_tmp = dx * s - dy * c; break;
        case 8: x_tmp = dx * c - dy * s; y_tmp = dx * s + dy * c; break;
        case 7: x_tmp = -dx * c + dy * s; y_tmp = -dx * s - dy * c; break;
        default: x_tmp = dx; y_tmp = dy; break;
        }
        float R = sqrtf(x_tmp * x_tmp + y_tmp * y_tmp + Height * Height);
        float fd = (R > 1e-6f) ? (2.0f * bp.V_feiji * x_tmp / (R * lambda)) : 0.0f;
        float R_max = R_min + (M - 1) * R_bin;
        float fd_max = bp.min_fd + (nEff - 1) * bp.delta_fd;
        if (R < R_min || R > R_max || fd < bp.min_fd || fd > fd_max) continue;
        float R_idx = (R - R_min) / R_bin;
        float fd_idx = (fd - bp.min_fd) / (bp.delta_fd > 1e-9f ? bp.delta_fd : 1e-9f);
        R_idx = fmaxf(0.0f, fminf(R_idx, (float)(M - 1)));
        fd_idx = fmaxf(0.0f, fminf(fd_idx, (float)(nEff - 1)));
        if (R_idx >= 0.0f && R_idx < (float)(M - 1) && fd_idx >= 0.0f && fd_idx < (float)(nEff - 1)) {
            float amp_val = tex2D<float>(texObjArr[b], R_idx + 0.5f, fd_idx + 0.5f);
            if (fabsf(amp_val) > fabsf(best_amp)) {
                best_amp = amp_val;
                best_beam_idx = (uint16_t)(b + 1);
            }
        }
    }
    int out_idx = j * nx + i;
    if (out_idx < 0 || out_idx >= nx * ny) return;
    d_amp_mosaic[out_idx] = best_amp;
    d_which_beam[out_idx] = best_beam_idx;
}

void launchBuildMosaicKernelTexArray(
    float* d_amp_mosaic, uint16_t* d_which_beam,
    const float* d_all_amps,
    const float* d_grid_x, const float* d_grid_y,
    const void* d_beams,
    int nx, int ny, int B, int M, int nEff,
    float R_min, float R_bin, float lambda, float Height, int flag_base
) {
    dim3 block(16, 16);
    dim3 gsize((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    cudaError_t err;
    // 先将 device 上的 d_beams 拷贝到 host
    BeamDevParams* h_beams = (BeamDevParams*)malloc(B * sizeof(BeamDevParams));
    if (!h_beams) {
        printf("[HOST] malloc h_beams failed!\n");
        return;
    }
    err = cudaMemcpy(h_beams, d_beams, B * sizeof(BeamDevParams), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("[HOST] cudaMemcpy d_beams to h_beams failed: %s\n", cudaGetErrorString(err));
        free(h_beams);
        return;
    }
    // 分配 B 个纹理对象
    cudaTextureObject_t* h_texObjArr = (cudaTextureObject_t*)malloc(B * sizeof(cudaTextureObject_t));
    cudaTextureObject_t* d_texObjArr = nullptr;
    err = cudaMalloc(&d_texObjArr, B * sizeof(cudaTextureObject_t));
    if (err != cudaSuccess) {
        printf("[HOST] cudaMalloc d_texObjArr failed: %s\n", cudaGetErrorString(err));
        free(h_beams);
        free(h_texObjArr);
        return;
    }
    cudaArray_t* h_cuArrayArr = (cudaArray_t*)malloc(B * sizeof(cudaArray_t));
    for (int b = 0; b < B; ++b) {
        size_t offset = h_beams[b].offset;
        const float* b_data = d_all_amps + offset;
        float* temp_dev = nullptr;
        err = cudaMalloc(&temp_dev, M * nEff * sizeof(float));
        if (err != cudaSuccess) {
            printf("[HOST] Beam %d: cudaMalloc temp_dev failed: %s\n", b, cudaGetErrorString(err));
            h_texObjArr[b] = 0;
            h_cuArrayArr[b] = nullptr;
            continue;
        }
        cudaMemcpyKind copyKind = cudaMemcpyHostToDevice;
        cudaPointerAttributes attr;
        if (cudaPointerGetAttributes(&attr, b_data) == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
            copyKind = cudaMemcpyDeviceToDevice;
        }
        err = cudaMemcpy(temp_dev, b_data, M * nEff * sizeof(float), copyKind);
        if (err != cudaSuccess) {
            printf("[HOST] Beam %d: cudaMemcpy to temp_dev failed: %s\n", b, cudaGetErrorString(err));
            cudaFree(temp_dev);
            h_texObjArr[b] = 0;
            h_cuArrayArr[b] = nullptr;
            continue;
        }
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaArray_t cuArray;
        err = cudaMallocArray(&cuArray, &channelDesc, M, nEff);
        if (err != cudaSuccess) {
            printf("[HOST] Beam %d: cudaMallocArray failed: %s\n", b, cudaGetErrorString(err));
            cudaFree(temp_dev);
            h_texObjArr[b] = 0;
            h_cuArrayArr[b] = nullptr;
            continue;
        }
        err = cudaMemcpy2DToArray(cuArray, 0, 0, temp_dev, M * sizeof(float), M * sizeof(float), nEff, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            printf("[HOST] Beam %d: cudaMemcpy2DToArray failed: %s\n", b, cudaGetErrorString(err));
            cudaFree(temp_dev);
            cudaFreeArray(cuArray);
            h_texObjArr[b] = 0;
            h_cuArrayArr[b] = nullptr;
            continue;
        }
        cudaFree(temp_dev);
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;
        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;
        cudaTextureObject_t texObj = 0;
        err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
        if (err != cudaSuccess) {
            printf("[HOST] Beam %d: cudaCreateTextureObject failed: %s\n", b, cudaGetErrorString(err));
            cudaFreeArray(cuArray);
            h_texObjArr[b] = 0;
            h_cuArrayArr[b] = nullptr;
            continue;
        }
        h_texObjArr[b] = texObj;
        h_cuArrayArr[b] = cuArray;
    }
    // 拷贝纹理对象数组到 device
    err = cudaMemcpy(d_texObjArr, h_texObjArr, B * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("[HOST] cudaMemcpy texObjArr to device failed: %s\n", cudaGetErrorString(err));
        free(h_beams);
        free(h_texObjArr);
        free(h_cuArrayArr);
        cudaFree(d_texObjArr);
        return;
    }
    buildMosaicFullKernelTexArray<<<gsize, block>>>(
        d_amp_mosaic, d_which_beam, d_all_amps,
        d_grid_x, d_grid_y, (const BeamDevParams*)d_beams,
        nx, ny, B, M, nEff,
        R_min, R_bin, lambda, Height, flag_base,
        d_texObjArr
    );
    cudaDeviceSynchronize();
    // 清理
    for (int b = 0; b < B; ++b) {
        if (h_texObjArr[b]) cudaDestroyTextureObject(h_texObjArr[b]);
        if (h_cuArrayArr[b]) cudaFreeArray(h_cuArrayArr[b]);
    }
    cudaFree(d_texObjArr);
    free(h_beams);
    free(h_texObjArr);
    free(h_cuArrayArr);
}
