#include "DbsIO.hpp"
#include "RangeCompressCuFFT.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <chrono>
#include <cmath>

using Clock = std::chrono::high_resolution_clock;

static std::mt19937_64 rng(123456789);
static std::uniform_real_distribution<double> urd( -1.0, 1.0 );

static std::complex<double> rnd_cplx()
{
    return {urd(rng), urd(rng)};
}

// Build Params and random BeamRaw
static Params makeParams(int Lraw, int M, int W)
{
    Params P;
    P.range_samp_total = Lraw;
    P.range_samp_used = M;
    P.pulses_per_beam = W;
    P.frame_header_len = 0;
    P.isPC = false;
    P.fs_hz = 1.0;
    P.tau_s = 0.0;
    P.B_hz = 0.0;
    P.hasRefFunc = false;
    P.reffunc_path = "";
    P.dbs_data_path = "";
    P.time_skip_pulses = 0;
    P.gps_week_offset = 0;
    P.secBias = 0.0;
    P.PRF = 1.0;
    P.fc_hz = 1.0;
    return P;
}

static BeamRaw<double> makeBeamRaw(int W, int Lraw)
{
    BeamRaw<double> raw;
    raw.W = W;
    raw.Lraw = Lraw;
    raw.s.resize((size_t)W * Lraw);
    raw.fw_angle_deg.assign(W, 0.0);
    raw.t_utc.assign(W, 0.0);
    for (int k = 0; k < W; ++k) {
        for (int n = 0; n < Lraw; ++n) {
            raw.s[(size_t)k * Lraw + n] = rnd_cplx();
        }
    }
    return raw;
}

static std::vector<std::complex<double>> makeHfSpec(int Lraw)
{
    std::vector<std::complex<double>> H(Lraw);
    for (int i = 0; i < Lraw; ++i) H[i] = std::complex<double>(1.0, 0.0);
    return H;
}

static double norm2(const std::vector<std::complex<double>>& v)
{
    long double s = 0.0L;
    for (auto &x : v) s += std::norm(x);
    return std::sqrt((double)s);
}

int main()
{
    std::cout << "Test: rangeCompressCuFFT vs rangeCompressFFT" << std::endl;

    // test cases: tuples of (W, Lraw, M)
    std::vector<std::tuple<int,int,int>> cases;
    // small -> medium
    // cases.push_back({4, 256, 64});
    // cases.push_back({8, 512, 128});
    // cases.push_back({16, 1024, 256});
    // cases.push_back({32, 2048, 512});
    // larger cases to show scaling
    // cases.push_back({64, 4096, 1024});
    // cases.push_back({128, 4096, 1024});
    cases.push_back({64, 8192, 2048});
    cases.push_back({128, 8192, 2048});
    cases.push_back({256, 8192, 2048});
    cases.push_back({512, 8192, 2048});
    cases.push_back({1024, 8192, 2048});
    cases.push_back({2048, 8192, 2048});
    cases.push_back({4096, 1024, 256});
    cases.push_back({8192, 1024, 256});

    // open CSV output file in project root
    const std::string csvPath = std::string("./build/range_compress_results.csv");
    FILE *csv = std::fopen(csvPath.c_str(), "w");
    if (!csv) {
        std::fprintf(stderr, "Failed to open CSV output '%s'\n", csvPath.c_str());
        return 2;
    }
    std::fprintf(csv, "W,Lraw,M,output_size,cpu_ms,gpu_ms,speedup,max_abs,rmse,rel_rmse\n");

    for (auto &tc : cases) {
        int W, Lraw, M;
        std::tie(W, Lraw, M) = tc;
        std::cout << "\nCase W=" << W << " Lraw=" << Lraw << " M=" << M << "\n";

        Params P = makeParams(Lraw, M, W);
        BeamRaw<double> raw = makeBeamRaw(W, Lraw);
        std::vector<std::complex<double>> H = makeHfSpec(Lraw);

        std::vector<std::complex<double>> rc_cpu, rc_gpu;

        // Warmup
        rangeCompressFFT(P, raw, H, rc_cpu);
        rangeCompressCuFFT(P, raw, H, rc_gpu);

        // CPU timing (repeat N)
        int repeats = 10;
        auto t0 = Clock::now();
        for (int r = 0; r < repeats; ++r) {
            rangeCompressFFT(P, raw, H, rc_cpu);
        }
        auto t1 = Clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeats;

        // GPU timing (repeat N)
        // To reduce impact of initial allocation, we call once before timing
        rangeCompressCuFFT(P, raw, H, rc_gpu);
        t0 = Clock::now();
        for (int r = 0; r < repeats; ++r) {
            rangeCompressCuFFT(P, raw, H, rc_gpu);
        }
        t1 = Clock::now();
        double gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeats;

        // compare sizes
        if (rc_cpu.size() != rc_gpu.size()) {
            std::cout << "Size mismatch: cpu=" << rc_cpu.size() << " gpu=" << rc_gpu.size() << "\n";
            continue;
        }

        // compute error metrics
        double max_abs = 0.0;
        long double err2 = 0.0L;
        long double ref2 = 0.0L;
        for (size_t i = 0; i < rc_cpu.size(); ++i) {
            std::complex<double> d = rc_cpu[i] - rc_gpu[i];
            long double e = std::norm(d);
            err2 += e;
            ref2 += std::norm(rc_cpu[i]);
            double absd = std::abs(d);
            if (absd > max_abs) max_abs = absd;
        }
        double rmse = std::sqrt((double)(err2 / rc_cpu.size()));
        double rel_rmse = (ref2 > 0.0L) ? rmse / std::sqrt((double)ref2) : rmse;

        double speedup = (gpu_ms > 0.0) ? (cpu_ms / gpu_ms) : 0.0;
        std::cout << "CPU time (ms): " << cpu_ms << ", GPU time (ms): " << gpu_ms << ", speedup: " << speedup << "\n";
        std::cout << "Output size: " << rc_cpu.size() << ", max_abs_diff: " << max_abs << ", RMSE: " << rmse << ", rel_RMSE: " << rel_rmse << "\n";

        // write CSV line
        std::fprintf(csv, "%d,%d,%d,%zu,%.6f,%.6f,%.6f,%.12e,%.12e,%.12e\n",
                     W, Lraw, M, rc_cpu.size(), cpu_ms, gpu_ms, speedup,
                     max_abs, rmse, rel_rmse);
    }
    std::fclose(csv);
    std::cout << "\nTest complete. Results written to ./build/range_compress_results.csv" << std::endl;
    return 0;
}
