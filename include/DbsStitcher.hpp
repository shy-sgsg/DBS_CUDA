#pragma once
#include "DbsTypes.hpp"
#include "DbsIO.hpp"
#include "geo/GaussProj.hpp" // <== 新增
#include <tinyxml2.h>
#include <memory>
#include <array>

//#define DEBUG // 调试开关
#include <iostream>

#ifdef DEBUG
  #define DBG(msg)  std::cout << "[DEBUG] " << __FILE__ << ":" << __LINE__ << " " << msg << std::endl
#else
  #define DBG(msg)  ((void)0)
#endif

#define ERR(msg)  std::cerr << "[ERROR] " << __FILE__ << ":" << __LINE__ << " " << msg << std::endl

#ifdef _OPENMP
#include <omp.h>         // OpenMP 支持
#endif

class DbsStitcher
{
public:
  explicit DbsStitcher();
  ~DbsStitcher();

  bool loadParamsFromXml(const std::string &xmlPath, Params &P);
  bool saveParamsToXml(const std::string &xmlPath, const double &P);

  // 不再初始化 PROJ，改为设置中央经线
  bool setLonRef(double lon_ref_deg)
  {
    lon_ref_deg_ = lon_ref_deg;
    return true;
  }

  bool loadPosAndEstimateVelocity(const Params &P, PosData &POS);
  bool processAllBeams(Params &P, const PosData &POS, RDData &RD, MetaPack &meta);

  // GPU-accelerated end-to-end beam processing. Preserves the original
  // API but reorganizes the workflow to leverage existing GPU helpers
  // (e.g. `rangeCompressCuFFT`) and reduce redundant host work.
  // This function keeps the original `processAllBeams` intact so callers
  // can choose CPU or GPU processing explicitly.
  bool processAllBeamsGPU(Params &P, const PosData &POS, RDData &RD, MetaPack &meta);

  bool updateFdCtrEstimates(Params &P, RDData &RD, MetaPack &meta, const std::string &xmlPath);

  bool estimateMosaicExtent(const Params &P, const RDData &RD, const MetaPack &meta,
                            Bounds &b, Grid &grid);

  bool buildMosaic(const Params &P, const RDData &RD, const MetaPack &meta,
                   const Grid &grid, Mosaic &mosaic);

  // GPU-accelerated version of buildMosaic. Preserves the original API but
  // performs batched bilinear interpolation and other hot paths on the GPU.
  bool buildMosaicGPU(const Params &P, const RDData &RD, const MetaPack &meta,
                      const Grid &grid, Mosaic &mosaic);

  bool writeProducts(const Params &P, const Grid &grid, const Mosaic &mosaic, const Bounds &b);

  int estimateEffectiveAzBins(const Params &P, const PosData &POS);

  bool nextGMTIFileNamePNG(const std::string &dir,
                                     std::string &out_path,
                                     int &out_index) const;
  bool nextGMTIFileNameTxt(const std::string &dir,
                                     std::string &out_path,
                                     int &out_index) const;

private:
  static bool mkdir_p(const std::string &dir);
  // 使用 Gaussp3 / Gaussp3RV
  bool proj_ll_to_xy(double lat_deg, double lon_deg, double &x, double &y)
  {
    Gaussp3(lat_deg, lon_deg, lon_ref_deg_, x, y);
    return std::isfinite(x) && std::isfinite(y);
  }
  bool proj_xy_to_ll(double x, double y, double &lat_deg, double &lon_deg)
  {
    return Gaussp3RV(x, y, lon_ref_deg_, lat_deg, lon_deg);
  }

  // 按脉冲时间将 POS 右邻/最近样本映射并求均值
  bool mapPosToBeam(const std::vector<double> &t_utc,
                    const PosData &POS,
                    MapPosResult &out);

  // rcComplex: W×M, CV_32FC2（行=脉冲，列=距离门）
  // azOut: W×M, CV_32FC2（已 FFT + fftshift + recentre）
  bool azFftShiftAndRecenter(const Image2D<std::complex<float>> &rcComplex,
                                float PRF, float fd_ctr,
                                Image2D<std::complex<float>> &azOut);

  // 便捷重载：从行主 W×M 的 std::vector<std::complex<float>> 进入
  bool azFftShiftAndRecenter(const std::vector<std::complex<float>> &dataRC,
                                int W, int M,
                                float PRF, float fd_ctr,
                                Image2D<std::complex<float>> &azOut);

  // azSpecC: W×M, CV_32FC2（已做 fftshift + recentre，使 f_d≈0 位于中心）
  // 输出：amp_eff: nEff×M, CV_32F（幅度）
  //      fd_axis_eff: 1×nEff, CV_32F（Hz，中心平移回 +fd_ctr）
  bool sliceEffectiveAzBins(const Image2D<std::complex<float>> &azSpecC,
                               float fd_ctr, int nEff, float PRF,
                               Image2D<float> &amp_eff, std::vector<float> &fd_axis_eff);

private:
  double lon_ref_deg_ = 117.0; // 中央经线（度）
};
