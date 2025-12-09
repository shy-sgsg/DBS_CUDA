#pragma once
#include <vector>
#include <string>
#include <complex>
#include <cstdint>
#include <array>

// 轻量二维容器
template <typename T>
struct Image2D
{
  int rows = 0, cols = 0;
  std::vector<T> buf;
  Image2D() = default;
  Image2D(int r, int c) : rows(r), cols(c), buf((size_t)r * c) {}
  inline T &at(int r, int c) { return buf[(size_t)r * cols + c]; }
  inline const T &at(int r, int c) const { return buf[(size_t)r * cols + c]; }
  inline T *row(int r) { return &buf[(size_t)r * cols]; }
  inline const T *row(int r) const { return &buf[(size_t)r * cols]; }
  inline size_t size() const { return buf.size(); }
  inline bool empty() const { return buf.empty(); }
};

// -------------------------
// 基本参数和数据结构定义
// -------------------------

struct Params
{
  // paths
  std::string dbs_data_path;
  std::string result_dir;
  std::string pos_path;
  std::string reffunc_path;

  // acquisition
  int frame_header_len = 0;
  int range_samp_total = 0;
  int range_samp_used = 0;
  int pulses_per_beam = 0;
  double fc_hz = 17e9;
  double B_hz = 0.0;
  double fs_hz = 0.0;
  double tau_s = 0.0;
  double PRF = 0.0;
  double Rmin_m = 0.0;

  // scanning
  int beams_per_period = 0;
  double beamwidth_deg = 0.0;
  double scan_max_az_deg = 0.0;
  int period_first = 0;
  int period_count = 1;

  // stitch & geometry
  int beam_skip = 1;
  int range_skip = 1;
  double lon_ref_deg = 0.0;
  double out_res_m = 1.0;
  double mean_ground_h = 0.0;
  int interp_mode = 1;     // 1=nearest, 2=bilinear
  int flight_dir_flag = 3; // 1西南 2东北 3西北 4东南

  // dataset specifics
  int time_skip_pulses = 0;
  double gps_week_offset = 0.0;
  double secBias = 0.0;
  int isPC = 1;
  int hasRefFunc = 0;
  int squint_side = 0; // 0 for left, 1 for right
};

struct PosData
{
  std::vector<double> gpstime;
  std::vector<double> lat_rad, lon_rad, alt_m;
  std::vector<double> vx, vy, vz, speed;
  std::vector<double> x_m, y_m;
};

struct RDData
{
  // 每波位一个矩阵：行=nEff(方位频通道)，列=M(距离门)
  std::vector<Image2D<float>> amp;         // nEff x M（幅度）
  std::vector<std::vector<float>> fd_axis; // 每波位 1×nEff
  std::vector<std::vector<float>> rg_axis; // 每波位 1×M
  int nEff = 0;
};

struct MetaPerBeam
{
  float vN = 0, vE = 0, vU = 0;
  float x = 0, y = 0, z = 0;
  float fd_ctr = 0;
  float angle_deg = 0;
};
struct MetaPack
{
  std::vector<MetaPerBeam> beams;
};

struct Grid
{
  std::vector<float> x; // size=nx
  std::vector<float> y; // size=ny
  float dx = 1, dy = 1;
};

struct Mosaic
{
  Image2D<float> amp;           // ny x nx
  Image2D<uint16_t> which_beam; // ny x nx (存 1-based 波位)
  Image2D<float> r_save;        // ny x nx
  Image2D<float> fd_save;       // ny x nx
  Image2D<float> x_rel_save;    // ny x nx
  Image2D<float> y_rel_save;    // ny x nx
};

struct Bounds
{
  float minX = 0, maxX = 0, minY = 0, maxY = 0;
};

// 波束原始数据（行主 W×Lraw）
// W: 波束数，Lraw: 每个波束的原始采样
// s: 复数采样数据（IQ），fw_angle_deg: 波束方位角（度），t_utc: UTC时间（秒）
template <typename T>
struct BeamRaw
{
  int W = 0;
  int Lraw = 0;
  std::vector<std::complex<T>> s; // 行主 W×Lraw
  std::vector<T> fw_angle_deg;    // size=W
  std::vector<double> t_utc;          // size=W
};

struct MapPosResult
{
  float vN = 0, vE = 0, vU = 0;
  float xyz0[3] = {0, 0, 0};
  float vmean = 0;
};
