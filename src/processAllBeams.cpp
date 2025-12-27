#include "DbsStitcher.hpp"
#include "DbsIO.hpp"
#include "RangeCompressCuFFT.hpp"
#include <fstream>
#include <iostream>
#include <chrono>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include "unwrap_fd.hpp"

// 光速（避免依赖外部 c0()）
static inline float c0f_local() { return 299792458.0f; }

inline std::vector<std::complex<double>>
buildHfSpec(int N, double fs, double Kr)
{
  std::vector<std::complex<double>> Hf(N);
  if (N <= 0 || fs <= 0.0 || Kr == 0.0)
    return Hf;

  const double df = fs / double(N);
  // 直接用“未移位的 DFT 频率”公式：
  //   n = 0..N-1
  //   f(n) = (n <= N/2-1) ? n*df : (n-N)*df
  for (int n = 0; n < N; ++n)
  {
    double fn = (n <= (N / 2 - 1)) ? n * df : (n - N) * df;
    double phase = M_PI * (fn * fn) / Kr; // -pi*f^2/Kr
    Hf[n] = std::complex<double>(std::cos(phase), std::sin(phase));
  }
  return Hf;
}

inline std::vector<std::complex<double>>
buildHfSpecFromRefFunc(const Params &P){
  int N = P.range_samp_total;
  FILE* fp = std::fopen(P.reffunc_path.c_str(), "rb");
  if (!fp) { std::perror("fopen"); return {}; }
  std::vector<std::complex<double>> Hf(N);
  std::vector<double> H_amp(N), H_phase(N);
  // 读取参考函数数据
  for (int n = 0; n < N; ++n)
  {
    float amp = 0.0f;
    if(std::fread(&amp, sizeof(float), 1, fp) != 1) {
      std::fclose(fp);
      return {};
    }
    H_amp[n] = static_cast<double>(amp);
  }
  for (int n = 0; n < N; ++n)
  {
    double re = 0.0, im = 0.0;
    float phase = 0.0f;
    if(std::fread(&phase, sizeof(float), 1, fp) != 1) {
      std::fclose(fp);
      return {};
    }
    re = H_amp[n] * std::cos(1.0*phase);
    im = H_amp[n] * std::sin(1.0*phase);
    Hf[n] = std::complex<double>(re, im);
  }
  std::fclose(fp);
  return Hf;
}

inline std::vector<std::complex<double>>
buildHfSpecFromParams(const Params &P)
{
  const int N = P.range_samp_total;
  const double fs = P.fs_hz;
  const double Kr = (P.tau_s > 0.0) ? (P.B_hz / P.tau_s) : 0.0;
  if (P.hasRefFunc)
    return buildHfSpecFromRefFunc(P);
  else
    return buildHfSpec(N, fs, Kr);
}

bool DbsStitcher::processAllBeams(Params &P,
                                  const PosData &POS,
                                  RDData &RD,
                                  MetaPack &meta)
{
  const int B = P.beams_per_period;
  const int N = P.beam_skip;
  const int W = P.pulses_per_beam;
  const int M = P.range_samp_used;

  const int B_count = (int)std::ceil(B / std::max(1, N)); // 实际处理的波位数

  // ---- 预分配 ----
  RD.nEff = (RD.nEff > 0) ? RD.nEff : estimateEffectiveAzBins(P, POS);
  RD.amp.resize(B_count);     // 每个元素: nEff x M
  RD.fd_axis.resize(B_count); // 每个元素: 1 x nEff（std::vector<float>）
  RD.rg_axis.resize(B_count); // 每个元素: 1 x M   （std::vector<float>）
  meta.beams.resize(B_count);

  // ---- 斜距轴（每波位相同，直接预生成一个 row 复用）----
  P.fs_hz = P.fs_hz * (static_cast<double>(M) / static_cast<double>(P.range_samp_total)); // 更新采样率
  const float Rbin = c0f_local() / (2.0f * static_cast<float>(P.fs_hz));
  std::vector<float> rg_axis_row;
  rg_axis_row.resize(M);
  for (int m = 0; m < M; ++m)
  {
    rg_axis_row[m] = static_cast<float>(P.Rmin_m) + m * Rbin;
  }

  // ===== 主循环：逐波位处理 =====
  for (int b = 0; b < B_count; ++b)
  {
    // -------------------- (a) 读单波位 & 距离脉压 --------------------
    BeamRaw<double> raw;
    int p = b * std::max(1, N) + 1; // 实际波位索引
    if (!readBeamRaw(P, p, raw))
    {
      std::fprintf(stderr, "[ERR] readBeamRaw failed on beam %d\n", b);
      return false;
    }

    std::vector<std::complex<double>> rc; // W*M，行主（每行一脉冲）
    if (!P.isPC)
    {
      std::vector<std::complex<double>> HfSpec = buildHfSpecFromParams(P);

      // --- 计时开始 ---
      auto start_time = std::chrono::high_resolution_clock::now();

      if (!rangeCompressCuFFT(P, raw, HfSpec, rc))
      {
        std::fprintf(stderr, "[ERR] rangeCompressCuFFT failed on beam %d\n", b);
        return false;
      }

      // --- 计时结束 ---
      auto end_time = std::chrono::high_resolution_clock::now();

      // 计算持续时间 (单位：毫秒)
      std::chrono::duration<double, std::milli> duration = end_time - start_time;
      
      // 打印结果
      // std::cout << "[TIMER] Beam " << b << " rangeCompressCuFFT took: "
      //           << duration.count() << " ms" << std::endl;
    }
    else
    {
      const int Lraw = raw.Lraw;
      const int Lraw2M = Lraw / M;
      rc.resize((size_t)W * M);
      for (int k = 0; k < W; ++k)
      {
        for (int m = 0; m < M; ++m)
        {
          rc[(size_t)k * M + m] = raw.s[(size_t)k * Lraw + m * Lraw2M];
        }
      }
    }
    // 调整 UTC 时间
    double mean_utc = 0.0;
    for(size_t i=0;i<raw.t_utc.size();++i){
      mean_utc += raw.t_utc[i];
    }
    mean_utc /= static_cast<double>(raw.t_utc.size());
    for(size_t i=0;i<raw.t_utc.size();++i){
      if(mean_utc - raw.t_utc[i] > 0.8)
        raw.t_utc[i] += 1;
    }

    // -------------------- (b) POS 就近映射到脉冲时间 --------------------
    MapPosResult mpos;
    if (!mapPosToBeam(raw.t_utc, POS, mpos))
    {
      std::fprintf(stderr, "[WARN] mapPosToBeam failed; fallback to global means\n");
      std::vector<double> empty; // 触发退化路径
      mapPosToBeam(empty, POS, mpos);
    }

    // meta 初始化
    MetaPerBeam m;
    m.vN = mpos.vN;
    m.vE = mpos.vE;
    m.vU = mpos.vU;
    m.x = mpos.xyz0[0];
    m.y = mpos.xyz0[1];
    m.z = mpos.xyz0[2];
    //DBG("Beam " << b << " mapped POS: vN=" << m.vN << " vE=" << m.vE
    //    << " vU=" << m.vU << " x=" << m.x << " y=" << m.y << " z=" << m.z);
    // 波束角平均
    if (!raw.fw_angle_deg.empty())
    {
      float sum = std::accumulate(raw.fw_angle_deg.begin(), raw.fw_angle_deg.end(), 0.0f);
      m.angle_deg = sum / static_cast<float>(raw.fw_angle_deg.size());
    }
    else
    {
      m.angle_deg = 0.0f;
    }

    // -------------------- (c) f_d 估计与模糊修正 --------------------
    const double speed = std::sqrt(m.vN * m.vN + m.vE * m.vE + m.vU * m.vU);
    const double lambda = c0f_local() / static_cast<double>(P.fc_hz);
    std::vector<double> fd_ctr = estimateFdCenter(rc,
                                P.pulses_per_beam,
                                P.range_samp_used,
                                P.PRF,
                                speed,
                                raw.fw_angle_deg,
                                lambda, // lambda
                                1 /*method*/);
    m.fd_ctr = fd_ctr[1];
    meta.beams[b].fd_ctr = m.fd_ctr;
    DBG("Beam " << b << " estimated fd_ctr=" << m.fd_ctr << " Hz");
    // -------------------- (d) 方位 FFT + 频率重心移零 --------------------
    // rc: W x M（行主）→ 先装入 Image2D，再 FFT&recentre
    Image2D<std::complex<float>> rcImg(W, M);
    {
      if (static_cast<int>(rc.size()) != W * M)
      {
        std::fprintf(stderr, "[ERR] rc size mismatch on beam %d\n", b);
        return false;
      }
      for (int r = 0; r < W; ++r)
      {
        for (int c = 0; c < M; ++c)
        {
          rcImg.at(r, c) = static_cast<std::complex<float>>(rc[static_cast<size_t>(r) * M + c]);
        }
      }
    }

    Image2D<std::complex<float>> azSpec; // W x M, 复数谱（已 recentre 使 fd~0 在中心）
    if (!azFftShiftAndRecenter(rcImg, static_cast<float>(P.PRF), m.fd_ctr, azSpec))
    {
      std::fprintf(stderr, "[ERR] azFftShiftAndRecenter failed on beam %d\n", b);
      return false;
    }

    // -------------------- (f) 裁剪有效方位通道 --------------------
    Image2D<float> amp_eff;         // nEff x M 幅度
    std::vector<float> fd_axis_eff; // 1 x nEff
    if (!sliceEffectiveAzBins(azSpec, 0.0f, RD.nEff, static_cast<float>(P.PRF),
                              amp_eff, fd_axis_eff))
    {
      std::fprintf(stderr, "[ERR] sliceEffectiveAzBins failed on beam %d\n", b);
      return false;
    }

    // ---- 写回 RD / meta ----
    RD.amp[b] = amp_eff;         // nEff x M
    RD.fd_axis[b] = fd_axis_eff; // 1 x nEff
    RD.rg_axis[b] = rg_axis_row; // 1 x M
    meta.beams[b] = m;
  }

  return true;
}

// 安全取整（替代 std::round，兼容 C++11）
static inline int iround_clamped(float x, int lo, int hi)
{
  int v = (x >= 0.0f) ? int(x + 0.5f) : int(x - 0.5f);
  if (v < lo)
    v = lo;
  if (v > hi)
    v = hi;
  return v;
}

bool DbsStitcher::sliceEffectiveAzBins(const Image2D<std::complex<float>> &azSpecC,
                                       float fd_ctr, int nEff, float PRF,
                                       Image2D<float> &amp_eff,
                                       std::vector<float> &fd_axis_eff)
{
  // 基本校验
  if (azSpecC.empty())
    return false;
  const int W = azSpecC.rows;
  const int M = azSpecC.cols;
  if (W <= 0 || M <= 0 || nEff <= 0)
    return false;

  // 频轴：linspace(-PRF/2, PRF/2, W)（包含两端）
  const float df = (W > 1) ? (PRF / float(W - 1)) : 0.0f;

  // 离 0Hz 最近的索引（0-based）
  int k0 = 0;
  if (W > 1)
  {
    // 对应 MATLAB 的 round((0 + PRF/2)/df)
    const float pos = (0.5f * PRF) / df; // 理论上 = (W-1)/2
    k0 = iround_clamped(pos, 0, W - 1);
  }

  // 以中心为零裁剪 nEff 个方位频通道（0-based）
  int kL = std::max(0, iround_clamped(k0 - nEff / 2.0f, 0, W - 1));
  int kR = std::min(W - 1, kL + nEff - 1);

  // 保证凑齐 nEff（若贴边则向另一侧补）
  int got = kR - kL + 1;
  if (got < nEff)
  {
    if (kL == 0)
    {
      kR = std::min(W - 1, kL + nEff - 1);
    }
    else if (kR == W - 1)
    {
      kL = std::max(0, kR - (nEff - 1));
    }
    got = kR - kL + 1;
  }
  if (got < nEff)
    nEff = got; // W<nEff 时退化为全部

  // 分配输出：amp_eff(nEff x M)
  amp_eff = Image2D<float>(nEff, M);

  // 取幅度：abs(azSpecC(kL:kR, :))
  for (int r = 0; r < nEff; ++r)
  {
    const std::complex<float> *src = azSpecC.row(kL + r);
    float *dst = amp_eff.row(r);
    for (int c = 0; c < M; ++c)
    {
      // std::abs(std::complex<float>) 返回幅度（>=C++11 可用）
      dst[c] = std::abs(src[c]);
    }
  }

  // 对应的方位频率轴（加回 fd_ctr 偏移）
  fd_axis_eff.resize(nEff);
  for (int r = 0; r < nEff; ++r)
  {
    const int k = kL + r;                    // 0-based
    const float fd_k = -0.5f * PRF + k * df; // linspace(-PRF/2, PRF/2, W)
    fd_axis_eff[r] = fd_k + fd_ctr;
  }

  return true;
}


bool DbsStitcher::updateFdCtrEstimates(Params &P,
                                        RDData &RD,
                                        MetaPack &meta,
                                        const std::string &xmlPath)
{
  const int B_count = (int)RD.amp.size();
  if (B_count <= 0 || (int)meta.beams.size() != B_count)
    return false;

  float lambda = c0f_local() / static_cast<float>(P.fc_hz);
  float v = std::hypot(meta.beams[12].vN, meta.beams[12].vE);

  int beam_center_offset = P.time_skip_pulses / P.pulses_per_beam;

  float angle1 = std::asin(- meta.beams[12 - beam_center_offset].fd_ctr * lambda / (2.0f * v)) * 180.0f / M_PI + 1;
  float angle2 = std::asin(- meta.beams[13 - beam_center_offset].fd_ctr * lambda / (2.0f * v)) * 180.0f / M_PI - 1;

  float angle_deg = (angle1 + angle2) / 2.0f;
  saveParamsToXml(xmlPath, angle_deg);
  DBG("Estimated look angle: " << angle_deg << " deg");
#ifdef DEBUG
  std::vector<float> fd_vals;
#endif
  for(int b=0; b < B_count; ++b){
    angle1 = angle_deg + meta.beams[b].angle_deg;
#ifdef DEBUG
    fd_vals.push_back(meta.beams[b].fd_ctr);
#endif
    float fd_new = unwrap_prf_to_model(meta.beams[b].fd_ctr, P.PRF, angle1, v, P.fc_hz);
    meta.beams[b].fd_ctr = fd_new;
    DBG("Beam " << b << " updated fd_ctr=" << fd_new << " Hz");
    // 更新 RD.fd_axis
    for(int r=0; r < (int)RD.fd_axis[b].size(); ++r){
      RD.fd_axis[b][r] += fd_new;
    }
  }
#ifdef DEBUG
  DBG("Old fd_ctr values:");
  FILE* fp = std::fopen("fd_old.txt", "w");
  if(fp){
    for(size_t i=0; i < fd_vals.size(); ++i){
      std::fprintf(fp, "%f\n", fd_vals[i]);
    }
    std::fclose(fp);
  }
#endif
  

  return true;
}