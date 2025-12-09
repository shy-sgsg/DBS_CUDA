#pragma once
#include "DbsTypes.hpp"
#include <array>

// MATLAB: rec = read_beam_and_range_compress(P, beamIdx, Hf)
// C++: 拆成两步
bool readBeamRaw(const Params &P, int beamIdx, BeamRaw<double> &out);

// FFTW 版本（保留以兼容）
bool rangeCompressFFT(const Params &P,
                      const BeamRaw<double> &in,
                      const std::vector<std::complex<double>> &HfSpec,
                      std::vector<std::complex<double>> &rc_out); // 输出：W x M, CV_32FC2（复数）, 已裁剪到 M = upulse_len

std::vector<double> estimateFdCenter(const std::vector<std::complex<double>> &data,
                              int Na, int Nr,
                              double PRF,
                              double vmean,
                              const std::vector<double> &fw_angle_deg,
                              double lambda,
                              int k );