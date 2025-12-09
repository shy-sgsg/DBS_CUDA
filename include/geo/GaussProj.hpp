#pragma once
#include <cmath>
#include <limits>

// ------------------------------------------------------------
// Gaussp3 / Gaussp3RV
// 说明：
//  - 输入/输出单位都与原 MATLAB 一致：B、L、L0 用“度”；x(东)、y(北) 用“米”
//  - 椭球：WGS-84
//  - 数学常量和公式完全按你提供的版本翻译
// ------------------------------------------------------------

// 经纬度(度) -> 高斯平面直角坐标(米)
inline void Gaussp3(double B_deg, double L_deg, double L0_deg,
                    double& xout, double& yout)
{
    const double c  = 6399593.6258;
    const double E1 = 6.73949674227e-03;
    const double p0 = 57.29577951308232; // 180/pi
    const double a0 = 111132.9525494;
    const double a2 = -16038.50840;
    const double a4 = 16.83260;
    const double a6 = -2.198e-02;
    const double a8 = 3e-05;

    const double B1   = B_deg;
    const double L1   = L_deg;
    const double Brad = B1 * M_PI / 180.0;

    const double t  = std::tan(Brad);
    const double k  = E1 * std::cos(Brad) * std::cos(Brad);
    const double v  = 1.0 + k;
    const double N  = c / std::sqrt(v);
    const double l0 = L1 - L0_deg;
    const double p  = std::cos(Brad) * l0 / p0;

    const double X = a0*B1
                   + a2*std::sin(2*Brad)
                   + a4*std::sin(4*Brad)
                   + a6*std::sin(6*Brad)
                   + a8*std::sin(8*Brad);

    // y (北)
    {
        const double p2 = p*p;
        const double t2 = t*t;
        const double term1 = 1.0
          + ((5 - t2 + (9 + 4*k)*k)
          + ((61 + (t2 - 58)*t2 + (9 - 11*t2)*30*k)
          + (1385 + (-3111 + (543 - t2)*t2)*t2)*p2/56.0)*p2/30.0)*p2/12.0;
        yout = X + N * t * term1 * p2 / 2.0;
    }

    // x (东)
    {
        const double p2 = p*p;
        const double t2 = t*t;
        const double term1 = 1.0
          + ((1 - t2 + k)
          + ((5 + t2*(t2 - 18 - 58*k) + 14*k)
          + (61 + (-479 + (179 - t2)*t2)*t2)*p2/42.0)*p2/20.0)*p2/6.0;
        xout = 500000.0 + N * term1 * p;
    }
}

// 高斯平面直角坐标(米) -> 经纬度(度)
inline bool Gaussp3RV(double x, double y, double L0_deg,
                      double& B_deg_out, double& L_deg_out)
{
    // 常量
    const double c  = 6399593.6258;
    const double E1 = 6.73949674227e-03;
    const double E  = 6.6943799013e-03;      (void)E;      // 这个常量在原式中未直接使用
    const double p0 = 57.29577951308232;     // 180/pi
    const double a  = 6378137.0;             (void)a;      // 未直接使用
    const double a0 = 111132.9525494;        (void)a0;     // 逆算式未直接使用
    const double a2 = -16038.50840;          (void)a2;     // 逆算式未直接使用
    const double a4 = 16.83260;              (void)a4;
    const double a6 = -2.198e-02;            (void)a6;
    const double a8 = 3e-05;                 (void)a8;

    const double q0 = 157048761.142065e-15;
    const double q2 = 2526250855.8867e-12;
    const double q4 = -14923621.5362e-12;
    const double q6 = 120769.6828e-12;
    const double q8 = -1075.7667e-12;

    // 按“x 指东, y 指北”的约定：
    const double m_X2 = y;
    const double m_Y2 = x;

    // 反算
    const double B0 = m_X2 * q0;
    const double sB0 = std::sin(B0);
    const double Bf  = B0 + std::sin(2*B0) * ( q2
                    + sB0*sB0 * ( q4
                    + sB0*sB0 * ( q6 + q8*sB0*sB0 ) ) );

    const double Y   = m_Y2 - 500000.0;
    const double tf  = std::tan(Bf);
    const double k   = E1 * std::cos(Bf) * std::cos(Bf);
    const double vf  = 1.0 + k;
    const double Nf  = c / std::sqrt(vf);
    const double q   = Y / Nf;

    // B (纬度, 度)
    {
        const double tf2 = tf*tf;
        const double q2v = q*q;

        const double inner2 =
            (5 + 3*tf2*(1 + (-2 - 3*k)*k) + 3*k*(2 - k))
          + ( -(61 + 45*tf2*(2 + tf2) + (107 + (-162 - 45*tf2)*tf2)*k)
              + (1385 + (3633 + (4095 + 1575*tf2)*tf2)*tf2) * q2v / 56.0
            ) * q2v / 30.0;

        const double termB =
            p0 * tf * ( -vf + inner2 * q2v / 12.0 ) * q2v / 2.0;

        B_deg_out = Bf * 180.0 / M_PI + termB;
    }

    // L (经度, 度)
    {
        const double tf2 = tf*tf;
        const double q2v = q*q;

        const double innerL =
            -(1 + 2*tf2 + k)
            + ( (5 + 4*tf2*(7 + 6*tf2) + 2*k*(3 + 4*tf2))
                - (61 + (662 + (1320 + 720*tf2)*tf2)*tf2) * q2v / 42.0
              ) * q2v / 20.0;

        const double l0 = p0 * q / std::cos(Bf) * ( 1 + innerL * q2v / 6.0 );
        L_deg_out = L0_deg + l0;
    }

    return std::isfinite(B_deg_out) && std::isfinite(L_deg_out);
}
