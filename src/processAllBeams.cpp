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
#include "AzFftCuFFT.hpp"
#include <cuda_runtime.h>
#include "PerfLogger.hpp"

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

bool DbsStitcher::processAllBeams(Params &P, const PosData &POS, RDData &RD, MetaPack &meta)
{
  const std::string algoGroup = "ProcessAllBeams";
  PerfLogger::Timer t_stage, t_total;
  // t_total.start();

  t_stage.start();
  const int B = P.beams_per_period;
  const int N = P.beam_skip;
  const int W = P.pulses_per_beam;
  const int M = P.range_samp_used;
  const int B_count = (int)std::ceil(B / std::max(1, N));

  // ---- 预分配 ----
  RD.nEff = (RD.nEff > 0) ? RD.nEff : estimateEffectiveAzBins(P, POS);
  RD.amp.resize(B_count);
  RD.fd_axis.resize(B_count);
  RD.rg_axis.resize(B_count);
  meta.beams.resize(B_count);

  // ---- 斜距轴生成 ----
  P.fs_hz = P.fs_hz * (static_cast<double>(M) / static_cast<double>(P.range_samp_total));
  const float Rbin = c0f_local() / (2.0f * static_cast<float>(P.fs_hz));
  std::vector<float> rg_axis_row(M);
  for (int m = 0; m < M; ++m) rg_axis_row[m] = static_cast<float>(P.Rmin_m) + m * Rbin;

  PerfLogger::add("PrecomputeSetup", t_stage.stop_ms(), algoGroup);

  // ===== 主循环：逐波位处理 =====
  for (int b = 0; b < B_count; ++b)
  {
    // CPU_01. 读取原始数据 
    t_stage.start();
    BeamRaw<double> raw;
    int p = b * std::max(1, N) + 1;
    if (!readBeamRaw(P, p, raw)) return false;
    PerfLogger::add("ReadRaw_IO", t_stage.stop_ms(), algoGroup);

    // CPU_02. 距离脉压
    std::vector<std::complex<double>> rc;
    if (!P.isPC)
    {
      std::vector<std::complex<double>> HfSpec = buildHfSpecFromParams(P);
      t_stage.start();
      if (!rangeCompressFFT(P, raw, HfSpec, rc)) return false;
      PerfLogger::add("RangeCompress", t_stage.stop_ms(), algoGroup);
    }

    // CPU_03. UTC 时间调整 
    t_stage.start();
    double mean_utc = 0.0;
    for(auto t : raw.t_utc) mean_utc += t;
    if(!raw.t_utc.empty()) mean_utc /= (double)raw.t_utc.size();
    for(size_t i=0; i<raw.t_utc.size(); ++i) {
        if(mean_utc - raw.t_utc[i] > 0.8) raw.t_utc[i] += 1;
    }
    PerfLogger::add("UTC_Adjustment", t_stage.stop_ms(), algoGroup);

    // CPU_04. POS 映射 
    t_stage.start();
    MapPosResult mpos;
    if (!mapPosToBeam(raw.t_utc, POS, mpos)) { /* fallback */ }
    PerfLogger::add("PosMapping", t_stage.stop_ms(), algoGroup);

    // CPU_05. Meta 初始化
    t_stage.start();
    MetaPerBeam m;
    m.vN = mpos.vN; m.vE = mpos.vE; m.vU = mpos.vU;
    m.x = mpos.xyz0[0]; m.y = mpos.xyz0[1]; m.z = mpos.xyz0[2];
    if (!raw.fw_angle_deg.empty()) {
      float sum = std::accumulate(raw.fw_angle_deg.begin(), raw.fw_angle_deg.end(), 0.0f);
      m.angle_deg = sum / static_cast<float>(raw.fw_angle_deg.size());
    } else {
      m.angle_deg = 0.0f;
    }
    PerfLogger::add("MetaInit", t_stage.stop_ms(), algoGroup);

    // CPU_06. 多普勒频率中心估计 & 方位向 FFT 
    t_stage.start();
    const double speed = std::sqrt(m.vN * m.vN + m.vE * m.vE + m.vU * m.vU);
    const double lambda = c0f_local() / static_cast<double>(P.fc_hz);
    std::vector<double> fd_ctr_vec = estimateFdCenter(rc, W, M, P.PRF, speed, raw.fw_angle_deg, lambda, 1);
    m.fd_ctr = (float)fd_ctr_vec[1];
    meta.beams[b].fd_ctr = m.fd_ctr;

    Image2D<std::complex<float>> rcImg(W, M);
    for (int r = 0; r < W; ++r) {
      for (int c = 0; c < M; ++c) {
        rcImg.at(r, c) = static_cast<std::complex<float>>(rc[static_cast<size_t>(r) * M + c]);
      }
    }

    Image2D<std::complex<float>> azSpec;
    if (!azFftShiftAndRecenter(rcImg, static_cast<float>(P.PRF), m.fd_ctr, azSpec)) return false;
    PerfLogger::add("Az FFT", t_stage.stop_ms(), algoGroup);

    // CPU_07. 有效通道切片 
    t_stage.start();
    Image2D<float> amp_eff;
    std::vector<float> fd_axis_eff;
    if (!sliceEffectiveAzBins(azSpec, 0.0f, RD.nEff, static_cast<float>(P.PRF), amp_eff, fd_axis_eff)) return false;
    PerfLogger::add("AzimuthSlice_Amp", t_stage.stop_ms(), algoGroup);

    // 保存结果
    RD.amp[b] = amp_eff;
    RD.fd_axis[b] = fd_axis_eff;
    RD.rg_axis[b] = rg_axis_row;
    meta.beams[b] = m;
  }

  // PerfLogger::add("CPU_ALL_BEAMS_DONE", t_total.stop_ms(), algoGroup);
  return true;
}

bool DbsStitcher::processAllBeamsGPU1(Params &P,
                                     const PosData &POS,
                                     RDData &RD,
                                     MetaPack &meta)
{
  const std::string algoGroup = "ProcessAllBeams";
  PerfLogger::Timer t_stage, t_total;
  // t_total.start();

  t_stage.start();
  const int B = P.beams_per_period;
  const int N = P.beam_skip;
  const int W = P.pulses_per_beam;
  const int M = P.range_samp_used;

  const int B_count = (int)std::ceil(B / std::max(1, N));

  // Work on a local copy of Params so we don't permanently mutate caller's
  // values while still applying the same sampling-rate adjustment used by the
  // CPU path.
  Params P2 = P;
  P2.fs_hz = P.fs_hz * (static_cast<double>(M) / static_cast<double>(P.range_samp_total));

  // ---- 预分配 ----
  RD.nEff = (RD.nEff > 0) ? RD.nEff : estimateEffectiveAzBins(P2, POS);
  RD.amp.resize(B_count);
  RD.fd_axis.resize(B_count);
  RD.rg_axis.resize(B_count);
  meta.beams.resize(B_count);

  // ---- 斜距轴（每波位相同，直接预生成一个 row 复用）----
  const float Rbin = c0f_local() / (2.0f * static_cast<float>(P2.fs_hz));
  std::vector<float> rg_axis_row(M);
  for (int m = 0; m < M; ++m)
    rg_axis_row[m] = static_cast<float>(P.Rmin_m) + m * Rbin;

  // Precompute HfSpec once when performing GPU range compression to avoid
  // rebuilding the same spec for every beam. This uses the adjusted sampling
  // rate in P2.
  std::vector<std::complex<double>> HfSpecGlobal;
  if (!P2.isPC)
    HfSpecGlobal = buildHfSpecFromParams(P2);
  
  PerfLogger::add("PrecomputeSetup", t_stage.stop_ms(), algoGroup);  

  // ===== 主循环：逐波位处理 =====
  for (int b = 0; b < B_count; ++b)
  {
    PerfLogger::Timer t_beam; t_beam.start();

    // (a) 读单波位 & 距离脉压（GPU：rangeCompressCuFFT）
    t_stage.start();
    BeamRaw<double> raw;
    int p = b * std::max(1, N) + 1;
    if (!readBeamRaw(P, p, raw))
    {
      std::fprintf(stderr, "[ERR] readBeamRaw failed on beam %d\n", b);
      return false;
    }
    PerfLogger::add("ReadRaw_IO", t_stage.stop_ms(), algoGroup);

    std::vector<std::complex<double>> rc; // W*M，行主（每行一脉冲）
    if (!P2.isPC)
    {
      t_stage.start();
      if (!rangeCompressCuFFT(P2, raw, HfSpecGlobal, rc))
      {
        std::fprintf(stderr, "[ERR] rangeCompressCuFFT failed on beam %d\n", b);
        return false;
      }
      PerfLogger::add("RangeCompress", t_stage.stop_ms(), algoGroup);
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

    // 调整 UTC 时间（与 CPU 路径一致）
    t_stage.start();
    double mean_utc = 0.0;
    for (size_t i = 0; i < raw.t_utc.size(); ++i)
      mean_utc += raw.t_utc[i];
    mean_utc /= static_cast<double>(raw.t_utc.size());
    for (size_t i = 0; i < raw.t_utc.size(); ++i)
      if (mean_utc - raw.t_utc[i] > 0.8)
        raw.t_utc[i] += 1;
    PerfLogger::add("UTC_Adjustment", t_stage.stop_ms(), algoGroup);

    // (b) POS 映射到脉冲时间
    t_stage.start();
    MapPosResult mpos;
    if (!mapPosToBeam(raw.t_utc, POS, mpos))
    {
      std::fprintf(stderr, "[WARN] mapPosToBeam failed; fallback to global means\n");
      std::vector<double> empty;
      mapPosToBeam(empty, POS, mpos);
    }
    PerfLogger::add("PosMapping", t_stage.stop_ms(), algoGroup);

    // meta 初始化
    t_stage.start();
    MetaPerBeam m;
    m.vN = mpos.vN;
    m.vE = mpos.vE;
    m.vU = mpos.vU;
    m.x = mpos.xyz0[0];
    m.y = mpos.xyz0[1];
    m.z = mpos.xyz0[2];
    if (!raw.fw_angle_deg.empty())
    {
      float sum = std::accumulate(raw.fw_angle_deg.begin(), raw.fw_angle_deg.end(), 0.0f);
      m.angle_deg = sum / static_cast<float>(raw.fw_angle_deg.size());
    }
    else
    {
      m.angle_deg = 0.0f;
    }
    PerfLogger::add("MetaInit", t_stage.stop_ms(), algoGroup);

    // (c) f_d 估计与模糊修正（保持原算法）
    t_stage.start();
    const double speed = std::sqrt(m.vN * m.vN + m.vE * m.vE + m.vU * m.vU);
    const double lambda = c0f_local() / static_cast<double>(P2.fc_hz);
    std::vector<double> fd_ctr = estimateFdCenter(rc,
                                P2.pulses_per_beam,
                                P2.range_samp_used,
                                P2.PRF,
                                speed,
                                raw.fw_angle_deg,
                                lambda,
                                1 /*method*/);
    m.fd_ctr = fd_ctr[1];
    meta.beams[b].fd_ctr = m.fd_ctr;

    // (d) 方位 FFT + 频率重心移零
    Image2D<std::complex<float>> rcImg(W, M);
    if (static_cast<int>(rc.size()) != W * M)
    {
      std::fprintf(stderr, "[ERR] rc size mismatch on beam %d\n", b);
      return false;
    }
    for (int r = 0; r < W; ++r)
      for (int c = 0; c < M; ++c)
        rcImg.at(r, c) = static_cast<std::complex<float>>(rc[static_cast<size_t>(r) * M + c]);

    Image2D<std::complex<float>> azSpec;
    if (!azFftShiftAndRecenterCuFFT(rcImg, static_cast<float>(P2.PRF), m.fd_ctr, azSpec))
    {
      std::fprintf(stderr, "[ERR] azFftShiftAndRecenterCuFFT failed on beam %d\n", b);
      return false;
    }
    PerfLogger::add("Az FFT", t_stage.stop_ms(), algoGroup);

    // (f) 裁剪有效方位通道
    t_stage.start();
    Image2D<float> amp_eff;
    std::vector<float> fd_axis_eff;
    if (!sliceEffectiveAzBins(azSpec, 0.0f, RD.nEff, static_cast<float>(P2.PRF),
                              amp_eff, fd_axis_eff))
    {
      std::fprintf(stderr, "[ERR] sliceEffectiveAzBins failed on beam %d\n", b);
      return false;
    }
    PerfLogger::add("AzimuthSlice_Amp", t_stage.stop_ms(), algoGroup);

    // 写回 RD / meta
    RD.amp[b] = amp_eff;
    RD.fd_axis[b] = fd_axis_eff;
    RD.rg_axis[b] = rg_axis_row;
    meta.beams[b] = m;
  }

  // PerfLogger::add("GPU1_ALL_BEAMS_DONE", t_total.stop_ms(), algoGroup);
  return true;
}

bool DbsStitcher::processAllBeamsGPU2(Params &P,
                                     const PosData &POS,
                                     RDData &RD,
                                     MetaPack &meta)
{
  const std::string algoGroup = "ProcessAllBeams";
  PerfLogger::Timer t_stage, t_total;
  // t_total.start();

  t_stage.start();
  const int B = P.beams_per_period;
  const int N = P.beam_skip;
  const int W = P.pulses_per_beam;
  const int M = P.range_samp_used;

  const int B_count = (int)std::ceil(B / std::max(1, N));

  // Work on a local copy of Params so we don't permanently mutate caller's
  // values while still applying the same sampling-rate adjustment used by the
  // CPU path.
  Params P2 = P;
  P2.fs_hz = P.fs_hz * (static_cast<double>(M) / static_cast<double>(P.range_samp_total));

  // ---- 预分配 ----
  RD.nEff = (RD.nEff > 0) ? RD.nEff : estimateEffectiveAzBins(P2, POS);
  RD.amp.resize(B_count);
  RD.fd_axis.resize(B_count);
  RD.rg_axis.resize(B_count);
  meta.beams.resize(B_count);

  // ---- 斜距轴（每波位相同，直接预生成一个 row 复用）----
  const float Rbin = c0f_local() / (2.0f * static_cast<float>(P2.fs_hz));
  std::vector<float> rg_axis_row(M);
  for (int m = 0; m < M; ++m)
    rg_axis_row[m] = static_cast<float>(P.Rmin_m) + m * Rbin;

  // Precompute HfSpec once when performing GPU range compression to avoid
  // rebuilding the same spec for every beam. This uses the adjusted sampling
  // rate in P2.
  std::vector<std::complex<double>> HfSpecGlobal;
  if (!P2.isPC)
    HfSpecGlobal = buildHfSpecFromParams(P2);

  // If GPU path: preallocate persistent device buffers and create a stream
  void *d_HfSpec = nullptr;
  void *d_inDouble = nullptr; // cuDoubleComplex* on device
  void *d_rcDouble = nullptr; // cuDoubleComplex* on device (W*M)
  void *d_rcFloat = nullptr;  // cufftComplex* on device (W*M)
  void *d_azFloat = nullptr;  // cufftComplex* on device (W*M) after az FFT
  cudaStream_t stream = nullptr;
  size_t size_hfspec_bytes = 0;
  size_t size_in_bytes = 0;
  size_t size_rc_bytes = 0;

  if (!P2.isPC) {
    // sizes depend on raw.Lraw which may vary per beam; allocate using worst-case
    // using P2.range_samp_total as an upper bound for Lraw
    int Lraw_max = P2.range_samp_total;
    int Lraw = Lraw_max;
    int Wloc = P2.pulses_per_beam;
    int Mloc = P2.range_samp_used;
    size_hfspec_bytes = (size_t)Lraw * sizeof(std::complex<double>);
    size_in_bytes = (size_t)Wloc * Lraw * sizeof(std::complex<double>);
    size_rc_bytes = (size_t)Wloc * Mloc * sizeof(std::complex<double>);

    if (HfSpecGlobal.size() == (size_t)Lraw) {
      if (cudaMalloc(&d_HfSpec, size_hfspec_bytes) != cudaSuccess) {
        std::fprintf(stderr, "[ERR] cudaMalloc d_HfSpec failed\n");
        return false;
      }
    }

    if (cudaMalloc(&d_inDouble, size_in_bytes) != cudaSuccess) {
      std::fprintf(stderr, "[ERR] cudaMalloc d_inDouble failed\n");
      return false;
    }
    if (cudaMalloc(&d_rcDouble, size_rc_bytes) != cudaSuccess) {
      std::fprintf(stderr, "[ERR] cudaMalloc d_rcDouble failed\n");
      return false;
    }
    if (cudaMalloc(&d_rcFloat, size_rc_bytes / sizeof(std::complex<double>) * sizeof(std::complex<float>)) != cudaSuccess) {
      std::fprintf(stderr, "[ERR] cudaMalloc d_rcFloat failed\n");
      return false;
    }
    if (cudaMalloc(&d_azFloat, size_rc_bytes / sizeof(std::complex<double>) * sizeof(std::complex<float>)) != cudaSuccess) {
      std::fprintf(stderr, "[ERR] cudaMalloc d_azFloat failed\n");
      return false;
    }

    if (cudaStreamCreate(&stream) != cudaSuccess) {
      std::fprintf(stderr, "[ERR] cudaStreamCreate failed\n");
      return false;
    }

    // copy HfSpecGlobal to device if available
    if (d_HfSpec && !HfSpecGlobal.empty()) {
      // HfSpecGlobal is std::complex<double>; device function expects cuDoubleComplex layout compatible
      if (cudaMemcpyAsync(d_HfSpec, HfSpecGlobal.data(), size_hfspec_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        std::fprintf(stderr, "[ERR] cudaMemcpyAsync HfSpec failed\n");
        return false;
      }
      cudaStreamSynchronize(stream);
    }
  }
  PerfLogger::add("PrecomputeSetup", t_stage.stop_ms(), algoGroup);

  // ===== 主循环：逐波位处理 =====
  for (int b = 0; b < B_count; ++b)
  {
    // 与 GPU1 保持一致的波位索引
    t_stage.start();
    BeamRaw<double> raw;
    int p = b * std::max(1, N) + 1; // 保证与 GPU1 一致
    if (!readBeamRaw(P, p, raw))
    {
      std::fprintf(stderr, "[ERR] readBeamRaw failed on beam %d\n", b);
      return false;
    }
    PerfLogger::add("ReadRaw_IO", t_stage.stop_ms(), algoGroup);

    t_stage.start();
    std::vector<std::complex<double>> rc; // W*M，行主（每行一脉冲）
    // prepare per-beam sizes
    const int Lraw = raw.Lraw;
    const int Wloc = raw.W;
    const int Mloc = P2.range_samp_used;

    // 调整 UTC 时间（与 CPU 路径一致）
    double mean_utc = 0.0;
    for (size_t i = 0; i < raw.t_utc.size(); ++i)
      mean_utc += raw.t_utc[i];
    if (!raw.t_utc.empty())
      mean_utc /= static_cast<double>(raw.t_utc.size());
    for (size_t i = 0; i < raw.t_utc.size(); ++i)
      if (mean_utc - raw.t_utc[i] > 0.8)
        raw.t_utc[i] += 1;
    PerfLogger::add("UTC_Adjustment", t_stage.stop_ms(), algoGroup);

    // (b) POS 映射到脉冲时间 -- moved earlier so fd estimation can use meta
    t_stage.start();
    MapPosResult mpos;
    if (!mapPosToBeam(raw.t_utc, POS, mpos))
    {
      std::fprintf(stderr, "[WARN] mapPosToBeam failed; fallback to global means\n");
      std::vector<double> empty;
      mapPosToBeam(empty, POS, mpos);
    }
    PerfLogger::add("PosMapping", t_stage.stop_ms(), algoGroup);

    // meta 初始化 (we need m for speed/lambda before fd estimation)
    t_stage.start();
    MetaPerBeam m;
    m.vN = mpos.vN;
    m.vE = mpos.vE;
    m.vU = mpos.vU;
    m.x = mpos.xyz0[0];
    m.y = mpos.xyz0[1];
    m.z = mpos.xyz0[2];
    if (!raw.fw_angle_deg.empty())
    {
      float sum = std::accumulate(raw.fw_angle_deg.begin(), raw.fw_angle_deg.end(), 0.0f);
      m.angle_deg = sum / static_cast<float>(raw.fw_angle_deg.size());
    }
    else
    {
      m.angle_deg = 0.0f;
    }
    PerfLogger::add("MetaInit", t_stage.stop_ms(), algoGroup);

    // --- fd_ctr估计必须在rc有数据后 ---
    const double speed = std::sqrt(m.vN * m.vN + m.vE * m.vE + m.vU * m.vU);
    const double lambda = c0f_local() / static_cast<double>(P2.fc_hz);
    if (!P2.isPC) {
      // ...existing code...
      t_stage.start();
      // GPU pipeline: async copy input to device, run device buffers pipeline,
      // then copy minimal result back (az spec) for host-side slicing.
      const int Lraw = raw.Lraw;
      const int Wloc = raw.W;
      const int Mloc = P2.range_samp_used;

      std::vector<std::complex<double>> h_inData((size_t)Wloc * Lraw);
      for (int i = 0; i < Wloc * Lraw; ++i)
        h_inData[i] = raw.s[i];

      size_t this_in_bytes = (size_t)Wloc * Lraw * sizeof(std::complex<double>);
      size_t this_rc_bytes = (size_t)Wloc * Mloc * sizeof(std::complex<double>);

      if (cudaMemcpyAsync(d_inDouble, h_inData.data(), this_in_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        std::fprintf(stderr, "[ERR] cudaMemcpyAsync input failed on beam %d\n", b);
        return false;
      }
      void *d_HfPtr = d_HfSpec;
      if (!rangeCompressCuFFT_device_buffers(P2, d_inDouble, d_HfPtr, d_rcDouble, Wloc, Lraw, Mloc, stream)) {
        std::fprintf(stderr, "[ERR] rangeCompressCuFFT_device_buffers failed on beam %d\n", b);
        return false;
      }
      rc.resize((size_t)Wloc * Mloc);
      size_t this_rc_bytes_host = (size_t)Wloc * Mloc * sizeof(std::complex<double>);
      if (cudaMemcpyAsync(rc.data(), d_rcDouble, this_rc_bytes_host, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        std::fprintf(stderr, "[ERR] cudaMemcpyAsync rc copy failed on beam %d\n", b);
        return false;
      }
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        std::fprintf(stderr, "[ERR] cudaStreamSynchronize failed on beam %d\n", b);
        return false;
      }
      // --- 关键修正：此时rc已有效 ---
      std::vector<double> fd_res = estimateFdCenter(rc, Wloc, Mloc, P2.PRF, speed, raw.fw_angle_deg, lambda, 1);
      m.fd_ctr = (fd_res.size() > 1) ? (float)fd_res[1] : 0.0f;
      meta.beams[b].fd_ctr = m.fd_ctr;
      PerfLogger::add("RangeCompress_And_FdEst", t_stage.stop_ms(), algoGroup);
    } else {
      // CPU path: rc已有效，直接估计
      std::vector<double> fd_res = estimateFdCenter(rc, Wloc, Mloc, P2.PRF, speed, raw.fw_angle_deg, lambda, 1);
      m.fd_ctr = (fd_res.size() > 1) ? (float)fd_res[1] : 0.0f;
      meta.beams[b].fd_ctr = m.fd_ctr;
      PerfLogger::add("FdCtrEstimate", t_stage.stop_ms(), algoGroup);
    }

    if (!P2.isPC)
    {
      t_stage.start();
      // GPU pipeline: async copy input to device, run device buffers pipeline,
      // then copy minimal result back (az spec) for host-side slicing.
      const int Lraw = raw.Lraw;
      const int Wloc = raw.W;
      const int Mloc = P2.range_samp_used;

      // prepare host input contiguous buffer (std::complex<double>) of size Wloc*Lraw
      std::vector<std::complex<double>> h_inData((size_t)Wloc * Lraw);
      for (int i = 0; i < Wloc * Lraw; ++i)
        h_inData[i] = raw.s[i];

      size_t this_in_bytes = (size_t)Wloc * Lraw * sizeof(std::complex<double>);
      size_t this_rc_bytes = (size_t)Wloc * Mloc * sizeof(std::complex<double>);

      // async copy host -> device input
      if (cudaMemcpyAsync(d_inDouble, h_inData.data(), this_in_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        std::fprintf(stderr, "[ERR] cudaMemcpyAsync input failed on beam %d\n", b);
        return false;
      }
    
      // choose device HfSpec pointer
      void *d_HfPtr = d_HfSpec;

      // run range compression on device (results in d_rcDouble)
      if (!rangeCompressCuFFT_device_buffers(P2, d_inDouble, d_HfPtr, d_rcDouble, Wloc, Lraw, Mloc, stream)) {
        std::fprintf(stderr, "[ERR] rangeCompressCuFFT_device_buffers failed on beam %d\n", b);
        return false;
      }
      // copy back the range-compressed data to host for fd estimation
      rc.resize((size_t)Wloc * Mloc);
      size_t this_rc_bytes_host = (size_t)Wloc * Mloc * sizeof(std::complex<double>);
      if (cudaMemcpyAsync(rc.data(), d_rcDouble, this_rc_bytes_host, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        std::fprintf(stderr, "[ERR] cudaMemcpyAsync rc copy failed on beam %d\n", b);
        return false;
      }

      // wait for range compress to finish so host can compute fd estimate
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        std::fprintf(stderr, "[ERR] cudaStreamSynchronize failed on beam %d\n", b);
        return false;
      }

      PerfLogger::add("RangeCompress", t_stage.stop_ms(), algoGroup);
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

    // (d) 方位 FFT + 频率重心移零
    t_stage.start();
    Image2D<std::complex<float>> azSpec;
    if (!P2.isPC) {
      // GPU path: run az FFT on device using preallocated buffers, then copy az-spectrum back
      size_t nelem = (size_t)Wloc * Mloc;
      if (static_cast<int>(rc.size()) != Wloc * Mloc) {
        std::fprintf(stderr, "[ERR] rc size mismatch on beam %d (gpu)\n", b);
        return false;
      }

      if (!convertDoubleToFloatDevice(d_rcDouble, d_rcFloat, nelem, stream)) {
        std::fprintf(stderr, "[ERR] convertDoubleToFloatDevice failed on beam %d\n", b);
        return false;
      }

      if (!azFftShiftAndRecenterCuFFT_device(d_rcFloat, Wloc, Mloc, static_cast<float>(P2.PRF), m.fd_ctr, d_azFloat, stream)) {
        std::fprintf(stderr, "[ERR] azFftShiftAndRecenterCuFFT_device failed on beam %d\n", b);
        return false;
      }

      std::vector<std::complex<float>> h_az(nelem);
      size_t this_az_bytes = nelem * sizeof(std::complex<float>);
      if (cudaMemcpyAsync(h_az.data(), d_azFloat, this_az_bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        std::fprintf(stderr, "[ERR] cudaMemcpyAsync az copy failed on beam %d\n", b);
        return false;
      }
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        std::fprintf(stderr, "[ERR] cudaStreamSynchronize failed on beam %d\n", b);
        return false;
      }

      azSpec = Image2D<std::complex<float>>(Wloc, Mloc);
      for (int r = 0; r < Wloc; ++r)
        for (int c = 0; c < Mloc; ++c)
          azSpec.at(r, c) = static_cast<std::complex<float>>(h_az[(size_t)r * Mloc + c]);
    }
    else {
      // CPU path: build rcImg and run host-side az FFT
      Image2D<std::complex<float>> rcImg(Wloc, Mloc);
      if (static_cast<int>(rc.size()) != Wloc * Mloc)
      {
        std::fprintf(stderr, "[ERR] rc size mismatch on beam %d (cpu)\n", b);
        return false;
      }
      for (int r = 0; r < Wloc; ++r)
        for (int c = 0; c < Mloc; ++c)
          rcImg.at(r, c) = static_cast<std::complex<float>>(rc[static_cast<size_t>(r) * Mloc + c]);

      if (!azFftShiftAndRecenterCuFFT(rcImg, static_cast<float>(P2.PRF), m.fd_ctr, azSpec))
      {
        std::fprintf(stderr, "[ERR] azFftShiftAndRecenterCuFFT failed on beam %d\n", b);
        return false;
      }
    }
    PerfLogger::add("Az FFT", t_stage.stop_ms(), algoGroup);

    // (f) 裁剪有效方位通道
    t_stage.start();
    Image2D<float> amp_eff;
    std::vector<float> fd_axis_eff;
    if (!sliceEffectiveAzBins(azSpec, 0.0f, RD.nEff, static_cast<float>(P2.PRF),
                              amp_eff, fd_axis_eff))
    {
      std::fprintf(stderr, "[ERR] sliceEffectiveAzBins failed on beam %d\n", b);
      return false;
    }
    PerfLogger::add("AzimuthSlice_Amp", t_stage.stop_ms(), algoGroup);

    // 写回 RD / meta
    RD.amp[b] = amp_eff;
    RD.fd_axis[b] = fd_axis_eff;
    RD.rg_axis[b] = rg_axis_row;
    meta.beams[b] = m;
  }

  // PerfLogger::add("GPU2_ALL_BEAMS_DONE", t_total.stop_ms(), algoGroup);
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