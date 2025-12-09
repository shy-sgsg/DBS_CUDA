
#include "DbsStitcher.hpp"
#include <algorithm>  // lower_bound
#include <numeric>    // accumulate
#include <cmath>

static inline double mean_of(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  return std::accumulate(v.begin(), v.end(), 0.0) / double(v.size());
}

bool DbsStitcher::mapPosToBeam(const std::vector<double>& t_utc,
                               const PosData& POS,
                               MapPosResult& out)
{
  // 退化条件：t_utc 为空或全 0
  const bool empty_or_zero = t_utc.empty() ||
    std::all_of(t_utc.begin(), t_utc.end(), [](double x){ return x==0.0; });

  if (empty_or_zero) {
    out.vN = static_cast<float>(mean_of(POS.vy));
    out.vE = static_cast<float>(mean_of(POS.vx));
    out.vU = static_cast<float>(mean_of(POS.vz));
    out.xyz0[0] = POS.x_m.empty()? 0.0f : static_cast<float>(POS.x_m.front());
    out.xyz0[1] = POS.y_m.empty()? 0.0f : static_cast<float>(POS.y_m.front());
    out.xyz0[2] = static_cast<float>(mean_of(POS.alt_m));
    out.vmean    = static_cast<float>(mean_of(POS.speed));
    return true;
  }

  // 否则：逐个脉冲时间查找最近的 POS 索引
  const size_t Npos = POS.gpstime.size();
  if (Npos == 0) return false;

  std::vector<size_t> idx; idx.reserve(t_utc.size());

  for (double t : t_utc) {
    // 由于 gpstime 通常单调递增，用 lower_bound 在 O(logN) 内找到右邻
    auto it = std::lower_bound(POS.gpstime.begin(), POS.gpstime.end(), t);
    size_t j_right = (it == POS.gpstime.end()) ? (Npos - 1) : size_t(it - POS.gpstime.begin());
    size_t j_left  = (j_right == 0) ? 0 : (j_right - 1);

    // 选更近的那个（最近邻）
    double dr = std::abs(POS.gpstime[j_right] - t);
    double dl = std::abs(POS.gpstime[j_left]  - t);
    size_t j  = (dl <= dr) ? j_left : j_right;
    idx.push_back(j);
  }

  // 用这些索引求均值
  auto mean_by_idx = [&](const std::vector<double>& arr)->double{
    if (idx.empty() || arr.empty()) return 0.0;
    double s = 0.0;
    for (size_t j : idx) s += arr[j];
    return s / double(idx.size());
  };

  out.vN = static_cast<float>(mean_by_idx(POS.vy));
  out.vE = static_cast<float>(mean_by_idx(POS.vx));
  out.vU = static_cast<float>(mean_by_idx(POS.vz));
  out.vmean = static_cast<float>(mean_by_idx(POS.speed));

  // 位置：x,y 取第一个脉冲对应的 POS；z 取 idx 集合高度均值
  // const size_t j0 = idx.front();
  const size_t j0 = idx[t_utc.size() / 2]; // 改为中间
  out.xyz0[0] = static_cast<float>(POS.x_m[j0]);
  out.xyz0[1] = static_cast<float>(POS.y_m[j0]);
  out.xyz0[2] = static_cast<float>(mean_by_idx(POS.alt_m));

  return true;
}
