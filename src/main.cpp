#include "DbsStitcher.hpp"
#include "DbsTypes.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <chrono>   // 必须包含
#include <iomanip>  // 用于格式化输出

// 定义计时器类，方便调用
class HighResTimer {
public:
    void start() { m_start = std::chrono::high_resolution_clock::now(); }
    double stop() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = end - m_start;
        return ms.count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

static int main_dbs_stitch(const std::string& xmlPath) {
    HighResTimer total_timer, step_timer;
    total_timer.start();

    // 1) 解析参数
    step_timer.start();
    DbsStitcher app;
    Params P;
    if (!app.loadParamsFromXml(xmlPath, P)) {
        std::cerr << "[ERR] parse xml failed: " << xmlPath << "\n";
        return 2;
    }
    app.setLonRef(P.lon_ref_deg);
    std::cout << "[PERF] 1. Load XML: " << step_timer.stop() << " ms" << std::endl;

    // 2) 读取POS
    step_timer.start();
    PosData POS;
    if (!app.loadPosAndEstimateVelocity(P, POS)) {
        std::cerr << "[ERR] load/estimate POS failed\n";
        return 3;
    }
    std::cout << "[PERF] 2. Load/Est POS: " << step_timer.stop() << " ms" << std::endl;

    // 3) 预估通道数
    step_timer.start();
    int nEff = app.estimateEffectiveAzBins(P, POS);
    std::cout << "[PERF] 3. Est AzBins: " << step_timer.stop() << " ms" << std::endl;

#ifdef _OPENMP
    int num_of_threads = std::max(1, omp_get_max_threads() - 1);
    omp_set_num_threads(num_of_threads);
    DBG("OpenMP enabled, max threads = " << num_of_threads);
#endif

    // 4) 处理波位成像 (核心耗时步骤)
    step_timer.start();
    RDData RD;
    RD.nEff = nEff;
    MetaPack meta;
    if (!app.processAllBeams(P, POS, RD, meta)) {
        std::cerr << "[ERR] processAllBeams failed\n";
        return 4;
    }
    std::cout << "[PERF] 4. Process All Beams: " << step_timer.stop() << " ms" << std::endl;

    step_timer.start();
    if(!app.updateFdCtrEstimates(P, RD, meta, xmlPath)) {
        std::cerr << "[ERR] updateFdCtrEstimates failed\n";
        return 8;
    }
    std::cout << "[PERF] 5. Update FdCtr: " << step_timer.stop() << " ms" << std::endl;

    // 5) 范围计算
    step_timer.start();
    Bounds bounds;
    Grid grid;
    if (!app.estimateMosaicExtent(P, RD, meta, bounds, grid)) {
        std::cerr << "[ERR] estimateMosaicExtent failed\n";
        return 5;
    }
    std::cout << "[PERF] 6. Est Extent: " << step_timer.stop() << " ms" << std::endl;

    // 6) GPU 拼接 (重点监控)
    step_timer.start();
    Mosaic mosaic;
    if (!app.buildMosaic(P, RD, meta, grid, mosaic)) {
        std::cerr << "[ERR] buildMosaic failed\n";
        return 6;
    }
    std::cout << "[PERF] 7. buildMosaicGPU: " << step_timer.stop() << " ms" << std::endl;

    // 7) 写出产品
    step_timer.start();
    if (!app.writeProducts(P, grid, mosaic, bounds)) {
        std::cerr << "[ERR] writeProducts failed\n";
        return 7;
    }
    std::cout << "[PERF] 8. Write Products: " << step_timer.stop() << " ms" << std::endl;

    std::cout << "------------------------------------------" << std::endl;
    std::cout << "[TOTAL] Full Pipeline: " << total_timer.stop() << " ms" << std::endl;
    std::cout << "DBS拼接完成。输出目录: " << P.result_dir << "\n";
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "用法: dbs_core <DBS_parameter.xml>\n";
        return 1;
    }
    return main_dbs_stitch(argv[1]);
}
