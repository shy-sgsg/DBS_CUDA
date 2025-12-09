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
