#pragma once

#include <cmath>
#include <iostream>
#include <vector>
#include <cstdio>

inline void rotation_xy_inv(float tmp_x0, float tmp_y0,
                            int flag,
                            float c, float s,
                            float &xl0, float &yl0)
{
    switch (flag)
    {
    case 1:
        xl0 = -tmp_x0 * c - tmp_y0 * s;
        yl0 = -tmp_x0 * s + tmp_y0 * c;
        break;
    case 2:
        xl0 = tmp_x0 * c + tmp_y0 * s;
        yl0 = tmp_x0 * s - tmp_y0 * c;
        break;
    case 3:
        xl0 = -tmp_x0 * c + tmp_y0 * s;
        yl0 = tmp_x0 * s + tmp_y0 * c;
        break;
    case 4:
        xl0 = tmp_x0 * c - tmp_y0 * s;
        yl0 = -tmp_x0 * s - tmp_y0 * c;
        break;
    case 6:
        xl0 = tmp_x0 * c + tmp_y0 * s;
        yl0 = -tmp_x0 * s + tmp_y0 * c;
        break;
    case 5:
        xl0 = -tmp_x0 * c - tmp_y0 * s;
        yl0 = tmp_x0 * s - tmp_y0 * c;
        break;
    case 8:
        xl0 = tmp_x0 * c - tmp_y0 * s;
        yl0 = tmp_x0 * s + tmp_y0 * c;
        break;
    case 7:
        xl0 = -tmp_x0 * c + tmp_y0 * s;
        yl0 = -tmp_x0 * s - tmp_y0 * c;
        break;
    default:
        xl0 = tmp_x0;
        yl0 = tmp_y0;
        break;
    }
}

inline void rotation_xy(float tmp_x0, float tmp_y0,
                        int flag,
                        float c, float s,
                        float &xl0, float &yl0)
{
    switch (flag)
    {
    case 1:
        xl0 = -tmp_x0 * c - tmp_y0 * s;
        yl0 = -tmp_x0 * s + tmp_y0 * c;
        break;
    case 2:
        xl0 = tmp_x0 * c + tmp_y0 * s;
        yl0 = tmp_x0 * s - tmp_y0 * c;
        break;
    case 3:
        xl0 = -tmp_x0 * c + tmp_y0 * s;
        yl0 = tmp_x0 * s + tmp_y0 * c;
        break;
    case 4:
        xl0 = tmp_x0 * c - tmp_y0 * s;
        yl0 = -tmp_x0 * s - tmp_y0 * c;
        break;
    case 6:
        xl0 = tmp_x0 * c - tmp_y0 * s;
        yl0 = tmp_x0 * s + tmp_y0 * c;
        break;
    case 5:
        xl0 = -tmp_x0 * c + tmp_y0 * s;
        yl0 = -tmp_x0 * s - tmp_y0 * c;
        break;
    case 8:
        xl0 = tmp_x0 * c + tmp_y0 * s;
        yl0 = -tmp_x0 * s + tmp_y0 * c;
        break;
    case 7:
        xl0 = -tmp_x0 * c - tmp_y0 * s;
        yl0 = tmp_x0 * s - tmp_y0 * c;
        break;
    default:
        xl0 = tmp_x0;
        yl0 = tmp_y0;
        break;
    }
}

// 根据多束 vN / vE 判定飞行方向
// 标志位约定：1西南(SW) 2东北(NE) 3西北(NW) 4东南(SE)
inline int infer_flight_dir_flag(const std::vector<float> &vN,
                           const std::vector<float> &vE,
                           float eps = 0.3f) // 速度近零阈值，按需调整
{
  int n = (int)std::min(vN.size(), vE.size());
  int cntN_pos = 0, cntN_neg = 0, cntE_pos = 0, cntE_neg = 0;
  double sumN = 0.0, sumE = 0.0;

  for (int i = 0; i < n; ++i)
  {
    sumN += vN[i];
    sumE += vE[i];
    if (vN[i] > eps)
      ++cntN_pos;
    else if (vN[i] < -eps)
      ++cntN_neg;
    if (vE[i] > eps)
      ++cntE_pos;
    else if (vE[i] < -eps)
      ++cntE_neg;
  }

  // 多数投票优先，打平时用均值兜底
  float sN = 0.f, sE = 0.f;
  if (cntN_pos > cntN_neg)
    sN = +1.f;
  else if (cntN_neg > cntN_pos)
    sN = -1.f;
  else
    sN = (sumN >= 0.0 ? +1.f : -1.f);

  if (cntE_pos > cntE_neg)
    sE = +1.f;
  else if (cntE_neg > cntE_pos)
    sE = -1.f;
  else
    sE = (sumE >= 0.0 ? +1.f : -1.f);

  // 映射到标志值
  if (sN < 0 && sE < 0)
    return 1; // 西南 SW
  if (sN > 0 && sE > 0)
    return 2; // 东北 NE
  if (sN > 0 && sE < 0)
    return 3; // 西北 NW
  if (sN < 0 && sE > 0)
    return 4; // 东南 SE

  // 理论上不会到这
  return 2; // 默认给个常见的 NE
}