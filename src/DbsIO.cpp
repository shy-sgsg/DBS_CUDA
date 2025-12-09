#include "DbsIO.hpp"
#include <cstdio>
#include <cstring>
#include <fftw3.h>
#include <stdexcept>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline int16_t load_int16_le(const uint8_t *p)
{
  return static_cast<int16_t>(p[0] | (p[1] << 8));
}
static inline uint32_t load_u32_le(const uint8_t *p)
{
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// 读取一个波位的原始 IQ + 帧头字段（与 MATLAB 完全对应）
bool readBeamRaw(const Params &P, int beamIdx, BeamRaw<double> &out)
{
  const int hdr = P.frame_header_len;
  const int Lraw = P.range_samp_total;
  const int W = P.pulses_per_beam;

  const size_t bytesIQperPulse = (size_t)Lraw * 2 /*I/Q*/ * sizeof(float);
  const size_t stride = (size_t)hdr + bytesIQperPulse;
  const size_t off_bytes = (size_t)(beamIdx - 1) * W * stride + (size_t)P.time_skip_pulses * stride;

  FILE *fp = std::fopen(P.dbs_data_path.c_str(), "rb");
  if (!fp)
  {
    std::perror("fopen");
    return false;
  }
  if (std::fseek(fp, (long)off_bytes, SEEK_SET) != 0)
  {
    std::perror("fseek");
    std::fclose(fp);
    return false;
  }

  out.W = W;
  out.Lraw = Lraw;
  out.s.resize((size_t)W * Lraw);
  out.fw_angle_deg.assign(W, 0.0f);
  out.t_utc.assign(W, 0.0);

  std::vector<uint8_t> header(hdr);
  std::vector<float> iq_interleaved((size_t)Lraw * 2);

  for (int k = 0; k < W; ++k)
  {
    // 1) 读帧头
    size_t nr = std::fread(header.data(), 1, hdr, fp);
    if (nr != (size_t)hdr)
    {
      std::fclose(fp);
      return false;
    }

    // 波束方位角：info(219:220)-[218:219] -> int16 -> /100（度）
    if (hdr >= 220)
    {
      int16_t ang = load_int16_le(&header[218]);
      out.fw_angle_deg[k] = ang / 100.0f;
    }
    else
    {
      out.fw_angle_deg[k] = 0.0f;
    }

    // 2) 读 IQ（float32 交错）
    nr = std::fread(iq_interleaved.data(), sizeof(float), (size_t)Lraw * 2, fp);
    if (nr != (size_t)Lraw * 2)
    {
      std::fclose(fp);
      return false;
    }

    // 交错转复数
    std::complex<double> *row = &out.s[(size_t)k * Lraw];
    const float *src = iq_interleaved.data();
    for (int n = 0; n < Lraw; ++n)
    {
      row[n] = std::complex<double>(1.0*src[2 * n + 0], 1.0*src[2 * n + 1]);
    }

    // 3) UTC 时间解码（与你的 MATLAB 一致）
    // info(37:39) 经 BCD 修正: x - 6*floor(x/16)
    auto bcd_fix = [](uint8_t u) -> int
    { return (int)u - 6 * ((int)u / 16); };
    int hh = 0, mm = 0, ss = 0;
    if (hdr >= 40)
    {
      hh = bcd_fix(header[36]); // 注意：MATLAB 的下标从1起，这里从0起，因此(37)->[36]
      mm = bcd_fix(header[37]); // (38)->[37]
      ss = bcd_fix(header[38]); // (39)->[38]
    }
    double subsecs = 0.0;
    if (hdr >= 44)
    {
      uint32_t ds = load_u32_le(&header[39]); // (40:43)->[39..42]
      subsecs = ds / 100e6;                   // /100e6
    }
    // + 周偏移 + 19 秒
    double t_utc = hh * 3600.0 + mm * 60.0 + ss + subsecs + P.gps_week_offset * 24.0 * 3600.0 + P.secBias;
    out.t_utc[k] = t_utc;
  }

  std::fclose(fp);
  return true;
}

// 距离向脉压（FFTW 批量：对每个脉冲做频域卷积），输出 W x M 的复数矩阵（CV_32FC2）
/*bool rangeCompressFFT(const Params& P,
                      const BeamRaw& in,
                      const std::vector<std::complex<float>>& HfSpec,
                      std::vector<std::complex<float>>& rc_out)
{
  const int W    = in.W;
  const int Lraw = in.Lraw;
  const int M    = P.range_samp_used;
  if ((int)HfSpec.size() != Lraw) return false;

  // 准备 FFTW 缓冲：W×Lraw（行主）
  fftwf_complex* buf = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (size_t)W * Lraw);
  if (!buf) return false;

  // 拷贝输入到 buf
  for (int k = 0; k < W; ++k) {
    const std::complex<float>* src = &in.s[(size_t)k * Lraw];
    fftwf_complex* dst = &buf[(size_t)k * Lraw];
    std::memcpy(dst, src, sizeof(std::complex<float>) * (size_t)Lraw);
  }

  // plan_many：沿 Lraw 做 FFT（每行一个 FFT），howmany=W
  int rank=1, n[1]={Lraw}, howmany=W;
  int istride=1, ostride=1, idist=Lraw, odist=Lraw;
  int* inembed=n; int* onembed=n;

  fftwf_plan fwd = fftwf_plan_many_dft(rank, n, howmany, buf, inembed, istride, idist,
                                       buf, onembed, ostride, odist,
                                       FFTW_FORWARD, FFTW_MEASURE);
  if (!fwd) { fftwf_free(buf); return false; }
  fftwf_plan inv = fftwf_plan_many_dft(rank, n, howmany, buf, inembed, istride, idist,
                                       buf, onembed, ostride, odist,
                                       FFTW_BACKWARD, FFTW_MEASURE);
  if (!inv) { fftwf_destroy_plan(fwd); fftwf_free(buf); return false; }

  // 前向 FFT
  fftwf_execute(fwd);

  // 频域乘以 conj(HfSpec)
  for (int k = 0; k < W; ++k) {
    fftwf_complex* row = &buf[(size_t)k * Lraw];
    for (int f = 0; f < Lraw; ++f) {
      std::complex<float>& X = *reinterpret_cast<std::complex<float>*>(&row[f]);
      X *= std::conj(HfSpec[(size_t)f]);
    }
  }

  // 逆向 FFT
  fftwf_execute(inv);

  // 归一化 + 裁剪到 M，输出到 rc_out（W×M 行主）
  rc_out.resize((size_t)W * M);
  const float scale = 1.0f / float(Lraw);
  for (int k = 0; k < W; ++k) {
    const fftwf_complex* row = &buf[(size_t)k * Lraw];
    for (int m = 0; m < M; ++m) {
      rc_out[(size_t)k * M + m] =
        (*reinterpret_cast<const std::complex<float>*>(&row[m])) * scale;
    }
  }

  fftwf_destroy_plan(fwd);
  fftwf_destroy_plan(inv);
  fftwf_free(buf);
  return true;
}*/

// 距离向脉压（FFTW 批量：对每个脉冲做频域卷积），输出 W x M 的复数矩阵（CV_32FC2）
bool rangeCompressFFT(const Params &P,
                      const BeamRaw<double> &in,
                      const std::vector<std::complex<double>> &HfSpec, // len = Lraw
                      std::vector<std::complex<double>> &rc_out)       // 新增：线程数（<=0 自动）
{
  const int W = in.W;
  const int Lraw = in.Lraw;
  const int M = P.range_samp_used;

  int nthreads = 0;
  if ((int)HfSpec.size() != Lraw)
    return false;

  // ---------- (1) FFTW 线程设置（可选，取决于是否链接了 fftw3_threads） ----------
#ifdef FFTW3_THREADS
  static bool fftw_thr_inited = false;
  if (!fftw_thr_inited)
  {
    fftw_init_threads();
    fftw_thr_inited = true;
  }
  if (nthreads <= 0)
  {
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#else
    nthreads = (int)std::max(1u, std::thread::hardware_concurrency());
#endif
  }
  fftw_plan_with_nthreads(nthreads);
#else
  (void)nthreads; // 未链接线程库时，避免未使用参数告警
#endif

  // ---------- (2) 准备 FFTW 缓冲：W×Lraw（行主） ----------
  fftw_complex *buf =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (size_t)W * Lraw);
  if (!buf)
    return false;

  // ---------- (3) 构建 plan_many：沿 Lraw 做 FFT（每行一个 FFT），howmany=W ----------
  int rank = 1, n[1] = {Lraw}, howmany = W;
  int istride = 1, ostride = 1, idist = Lraw, odist = Lraw;
  int *inembed = n;
  int *onembed = n;

  fftw_plan fwd = fftw_plan_many_dft(rank, n, howmany,
                                       buf, inembed, istride, idist,
                                       buf, onembed, ostride, odist,
                                       FFTW_FORWARD, FFTW_MEASURE);
  if (!fwd)
  {
    fftw_free(buf);
    return false;
  }

  fftw_plan inv = fftw_plan_many_dft(rank, n, howmany,
                                       buf, inembed, istride, idist,
                                       buf, onembed, ostride, odist,
                                       FFTW_BACKWARD, FFTW_MEASURE);
  if (!inv)
  {
    fftw_destroy_plan(fwd);
    fftw_free(buf);
    return false;
  }

  // 拷贝输入到 buf（行并行）
  for (int k = 0; k < W; ++k)
  {
    const std::complex<double> *src = &in.s[(size_t)k * Lraw];
    fftw_complex *dst = &buf[(size_t)k * Lraw];
    std::memcpy(dst, src, sizeof(std::complex<double>) * (size_t)Lraw);
  }

  // ---------- (4) 前向 FFT（内部可多线程） ----------
  fftw_execute(fwd);
  // ---------- (5) 频域乘以 conj(HfSpec)（行并行） ----------
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int k = 0; k < W; ++k)
  {
    fftw_complex *row = &buf[(size_t)k * Lraw];
    for (int f = 0; f < Lraw; ++f)
    {
      std::complex<double> &X = *reinterpret_cast<std::complex<double> *>(&row[f]);
      X *= HfSpec[(size_t)f];
    }
  }

  // ---------- (6) 逆向 FFT（内部可多线程） ----------
  fftw_execute(inv);

  // ---------- (7) 归一化 + 裁剪到 M（行并行），输出到 rc_out（W×M 行主） ----------
  rc_out.resize((size_t)W * M);
  const double scale = 1.0 / double(Lraw);
  const int Lraw2M = Lraw / M; // 假设 Lraw 是 M 的整数倍

  for (int k = 0; k < W; ++k)
  {
    const fftw_complex *row = &buf[(size_t)k * Lraw];
    for (int m = 0; m < M; ++m)
    {
      rc_out[(size_t)k * M + m] =
          (*reinterpret_cast<const std::complex<double> *>(&row[m * Lraw2M])) * scale;
    }
  }

  // ---------- (8) 清理 ----------
  fftw_destroy_plan(fwd);
  fftw_destroy_plan(inv);
  fftw_free(buf);
  return true;
}


std::vector<double> estimateFdCenter(const std::vector<std::complex<double>> &data,
                                    int Na, int Nr,
                                    double PRF,
                                    double vmean,
                                    const std::vector<double> &fw_angle_deg,
                                    double lambda,
                                    int k)
{
  if (Na <= k || Nr <= 0 || (int)data.size() != Na * Nr)
    return {0.0f, 0.0f};

  // ---- 1) fdc_HSXG: 相关法估计 fa2 ----
  // R_sum = sum_{mm=k..Na-1} sum_{nn=0..Nr-1} data(mm,nn) * conj(data(mm-k,nn))
  std::complex<double> R_sum(0.0f, 0.0f);
  for (int mm = k; mm < Na; ++mm)
  {
    const std::complex<double> *row_mm = &data[(size_t)mm * Nr];
    const std::complex<double> *row_mmk = &data[(size_t)(mm - k) * Nr];
    for (int nn = 0; nn < Nr; ++nn)
    {
      R_sum += row_mm[nn] * std::conj(row_mmk[nn]);
    }
  }
  // fa2 = PRF/(2*pi) * arg(R_sum)
  const double fa2 = PRF * std::atan2(std::imag(R_sum), std::real(R_sum)) / (2.0 * M_PI);

  // ---- 2) 几何先验：fd0 = 2*vmean*sin(-fw_angle(64))/lambda ----
  double ang_deg = 0.0;
  if (!fw_angle_deg.empty())
  {
    if ((int)fw_angle_deg.size() >= 256)
      ang_deg = fw_angle_deg[200]; // 1-based的第64个
    else
      ang_deg = fw_angle_deg[fw_angle_deg.size() / 2]; // 取中位
  }
  const double fd0 = 2.0 * vmean * std::sin(-ang_deg * (M_PI / 180.0)) / lambda;

  // ---- 3) 合成规则（与你给的分支一致）----
  double fd_ctr;

  if (fd0 > PRF * 3 / 2 && fa2 < 0)
    fd_ctr = 2 * PRF + fa2;
  else if (fd0 > PRF * 3 / 2 && fa2 > 0)
    fd_ctr = 2 * PRF - fa2;
  else if (fd0 > PRF && fa2 < 0)
    fd_ctr = PRF + fa2;
  else if (fd0 > PRF && fa2 > 0)
    fd_ctr = fa2 + PRF;
  else if (fd0 > PRF / 2 && fa2 > 0)
    fd_ctr = PRF - fa2;
  else if (fd0 > PRF / 2 && fa2 < 0)
    fd_ctr = PRF + fa2;
  else if (fd0 > -PRF / 2)
    fd_ctr = fa2;
  else if (fd0 > -PRF && fa2 < 0)
    fd_ctr = -PRF - fa2;
  else if (fd0 > -PRF && fa2 > 0)
    fd_ctr = -PRF + fa2;
  else if (fd0 > -PRF * 3 / 2 && fa2 > 0)
    fd_ctr = -2 * PRF + fa2;
  else if (fd0 > -PRF * 3 / 2 && fa2 < 0)
    fd_ctr = -PRF + fa2;
  else if (fd0 < -PRF * 3 / 2 && fa2 > 0)
    fd_ctr = fa2 - 2 * PRF;
  else
    fd_ctr = -fa2 - 2 * PRF;

  return {fd_ctr, fa2};
}
