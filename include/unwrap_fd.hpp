// unwrap_prf_to_model.hpp
#ifndef UNWRAP_PRF_TO_MODEL_HPP
#define UNWRAP_PRF_TO_MODEL_HPP

#include <cmath>

// 兼容 g++4.5：自定义“就近取整”到整数
static inline int nearest_int(double x) {
    // 等价于 round(x)，但避免依赖 C++11
    return (x >= 0.0) ? int(std::floor(x + 0.5)) : int(std::ceil(x - 0.5));
}

// 标量版解混叠：输入均为 double，返回 float
// d_est     : 观测到的（被 PRF 混叠后的）多普勒中心，单位 Hz，通常落在 (-PRF/2, PRF/2]
// PRF       : 脉冲重复频率，Hz
// theta_deg : 斜视角，度
// v         : 平台速度，m/s
// fc        : 载频，Hz  (例如 17e9)
//
// 策略：在两个候选族 S = { +d_est + n*PRF } 和 {-d_est + n*PRF } 中
//      各自选取与理论值 f_theory 最近的成员；两族再比较误差，取误差更小者。
//      中心角 |theta|<=1 度按你的规则“应当无模糊”，强制使用 +d_est（n=0）。
static inline float unwrap_prf_to_model(double d_est,
                                        double PRF,
                                        double theta_deg,
                                        double v,
                                        double fc)
{
    // 常量与理论值
    const double c  = 299792458.0;             // m/s
    const double pi = 3.14159265358979323846;
    const double lambda = c / fc;              // 波长
    const double theta = theta_deg * (pi / 180.0);
    const double f_theory = -2.0 * v / lambda * std::sin(theta);

    // 保护：PRF 非法时直接返回观测值
    if (PRF <= 0.0 || !std::isfinite(PRF)) {
        return static_cast<float>(d_est);
    }

    // 规则：中心几个位置（约 ±1°）认定“无模糊”
    if (std::fabs(theta_deg) <= 1.0) {
        // 返回 +d_est（不加拍），以匹配你的实际约束
        return static_cast<float>(d_est);
    }

    // 族一：+d_est + n*PRF
    const double n1_base = (f_theory - (+d_est)) / PRF;
    int n1 = nearest_int(n1_base);

    // 族二：-d_est + n*PRF   （用于你提到的案例：例如 2*PRF - d_est 更接近理论）
    const double n2_base = (f_theory - (-d_est)) / PRF;
    int n2 = nearest_int(n2_base);

    // 为稳健起见，各自再在邻域 ±2 搜索最优（防止边界取整误差）
    double best_val = 0.0;
    double best_err = 1e300;

    // 搜索族一
    {
        const int n0 = n1;
        for (int dn = -2; dn <= 2; ++dn) {
            const int n = n0 + dn;
            const double cand = (+d_est) + n * PRF;
            const double err  = std::fabs(cand - f_theory);
            if (err < best_err) {
                best_err = err;
                best_val = cand;
            }
        }
    }

    // 搜索族二
    {
        const int n0 = n2;
        for (int dn = -2; dn <= 2; ++dn) {
            const int n = n0 + dn;
            const double cand = (-d_est) + n * PRF;
            const double err  = std::fabs(cand - f_theory);
            if (err < best_err) {
                best_err = err;
                best_val = cand;
            }
        }
    }

    // 可选的物理合理性裁剪（不强制）：限制到 |f| <= 2v/lambda + 0.5*PRF
    // 你的数据若已干净可去掉
    // const double fmax = 2.0 * v / lambda + 0.5 * PRF;
    // if (std::fabs(best_val) > fmax) {
    //     best_val = (best_val > 0 ? fmax : -fmax);
    // }

    return static_cast<float>(best_val);
}

#endif // UNWRAP_PRF_TO_MODEL_HPP
