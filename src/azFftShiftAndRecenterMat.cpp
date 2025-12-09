#include "DbsStitcher.hpp"
#include <fftw3.h>
#include <algorithm>  // std::rotate, std::min/max
#include <cstdio>
#include <cmath>


// --- 工具：对每一列做环移（circshift）: 正 shift 表示向“下”移 ---
template<typename T>
static void circshift_rows_inplace(Image2D<T>& mat, int shift) {
  const int W = mat.rows, M = mat.cols;
  if (W<=0 || M<=0) return;
  shift %= W; if (shift < 0) shift += W;

  std::vector<T> col; col.resize(W);
  for (int m=0; m<M; ++m) {
    // 拷贝第 m 列到临时缓冲
    for (int k=0; k<W; ++k) col[k] = mat.at(k, m);
    // 旋转
    std::rotate(col.begin(), col.end() - shift, col.end());
    // 写回
    for (int k=0; k<W; ++k) mat.at(k, m) = col[k];
  }
}


// 小工具：安全取整（替代 std::round，兼容 C++11）
static inline int iround_pos_clamped(float x, int lo, int hi) {
  if (x < (float)lo) return lo;
  if (x > (float)hi) return hi;
  return (x >= 0.0f) ? int(x + 0.5f) : int(x - 0.5f);
}

bool DbsStitcher::azFftShiftAndRecenter(const Image2D<std::complex<float> >& rc,
                                        float PRF, float fd_ctr,
                                        Image2D<std::complex<float> >& azOut)
{
  if (rc.rows<=0 || rc.cols<=0) return false;
  const int W = rc.rows;
  const int M = rc.cols;

  // 输出拷贝（原地做 FFT）
  azOut = Image2D<std::complex<float> >(W, M);
  if (rc.size() != azOut.size()) return false;
  for (size_t i=0;i<rc.size();++i) azOut.buf[i] = rc.buf[i];

  // --- 1) 沿“脉冲维（行）”做 batched FFT（每列一条长度 W 的 FFT） ---
  fftwf_complex* data = reinterpret_cast<fftwf_complex*>(azOut.buf.data());

  // plan_many 参数：行主内存下，同一列相邻元素跨距=cols（M）
  int rank = 1;
  int n[1] = { W };
  int howmany = M;
  int inembed[1] = { W };
  int onembed[1] = { W };
  int istride = M;   // 行步长：同一列相邻元素相距 M 个复数
  int ostride = M;
  int idist = 1;     // 批间距：下一列起点相距 1 个复数元素（列索引 +1）
  int odist = 1;

  fftwf_plan plan = fftwf_plan_many_dft(rank, n, howmany,
                                        data, inembed, istride, idist,
                                        data, onembed, ostride, odist,
                                        FFTW_FORWARD, FFTW_MEASURE);
  if (!plan) {
    std::fprintf(stderr, "[ERR] fftwf_plan_many_dft failed (W=%d,M=%d)\n", W, M);
    return false;
  }
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);

  // --- 2) fftshift：沿行方向移位 W/2 ---
  circshift_rows_inplace(azOut, W/2);

  // --- 3) recentre：把估计的 fd_ctr 搬到“零频附近” ---
  // 与 MATLAB: fd_axis = linspace(-PRF/2, PRF/2, W) 一致（包含两端）
  int k0 = 0;
  if (W > 1) {
    const float df = PRF / float(W - 1);
    float pos = (fd_ctr + 0.5f*PRF) / df;           // 映射到 [0, W-1]
    k0 = iround_pos_clamped(pos, 0, W-1);
  }
  const int center0 = int(std::ceil(W/2.0)) - 1;    // 0-based 中心
  const int shift = center0 - k0;
  circshift_rows_inplace(azOut, shift);

  return true;
}

bool DbsStitcher::azFftShiftAndRecenter(const std::vector<std::complex<float> >& dataRC,
                                        int W, int M,
                                        float PRF, float fd_ctr,
                                        Image2D<std::complex<float> >& azOut)
{
  if ((int)dataRC.size()!=W*M) return false;
  Image2D<std::complex<float> > rc(W, M);
  for (int r=0;r<W;++r)
    for (int c=0;c<M;++c)
      rc.at(r,c) = dataRC[static_cast<size_t>(r)*M + c];
  return azFftShiftAndRecenter(rc, PRF, fd_ctr, azOut);
}

