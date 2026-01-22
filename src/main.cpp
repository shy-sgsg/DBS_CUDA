#include "DbsStitcher.hpp"
#include "DbsTypes.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <chrono>   // 必须包含
#include <iomanip>  // 用于格式化输出
#include "PerfLogger.hpp"

static int main_dbs_stitch(const std::string& xmlPath) {
    // 统一使用 PerfLogger 内部定义的 Timer
    PerfLogger::Timer total_timer, step_timer;
    total_timer.start();

    // 定义分组名称常量，避免手打出错
    const std::string flowGroup = "MainFlow";

    // 1) 解析参数
    step_timer.start();
    DbsStitcher app;
    Params P;
    if (!app.loadParamsFromXml(xmlPath, P)) {
        std::cerr << "[ERR] parse xml failed: " << xmlPath << "\n";
        return 2;
    }
    app.setLonRef(P.lon_ref_deg);
    // 记录到 MainFlow 分组
    PerfLogger::add("01. Load XML", step_timer.stop_ms(), flowGroup);

    // 2) 读取POS
    step_timer.start();
    PosData POS;
    if (!app.loadPosAndEstimateVelocity(P, POS)) {
        std::cerr << "[ERR] load/estimate POS failed\n";
        return 3;
    }
    PerfLogger::add("02. Load/Est POS", step_timer.stop_ms(), flowGroup);

    // 3) 预估通道数
    step_timer.start();
    int nEff = app.estimateEffectiveAzBins(P, POS);
    PerfLogger::add("03. Est AzBins", step_timer.stop_ms(), flowGroup);

#ifdef _OPENMP
    int num_of_threads = std::max(1, omp_get_max_threads() - 1);
    omp_set_num_threads(num_of_threads);
#endif

    // 4) 处理波位成像
    step_timer.start();
    RDData RD;
    RD.nEff = nEff;
    MetaPack meta;
    if (!app.processAllBeams(P, POS, RD, meta)) {
        std::cerr << "[ERR] processAllBeams failed\n";
        return 4;
    }
    PerfLogger::add("04. Process All Beams", step_timer.stop_ms(), flowGroup);

    step_timer.start();
    if(!app.updateFdCtrEstimates(P, RD, meta, xmlPath)) {
        std::cerr << "[ERR] updateFdCtrEstimates failed\n";
        return 8;
    }
    PerfLogger::add("05. Update FdCtr", step_timer.stop_ms(), flowGroup);

    // 5) 范围计算
    step_timer.start();
    Bounds bounds;
    Grid grid;
    if (!app.estimateMosaicExtent(P, RD, meta, bounds, grid)) {
        std::cerr << "[ERR] estimateMosaicExtent failed\n";
        return 5;
    }
    PerfLogger::add("06. Est Extent", step_timer.stop_ms(), flowGroup);

    // 6) GPU 拼接
    step_timer.start();
    Mosaic mosaic;
    if (!app.buildMosaicGPU(P, RD, meta, grid, mosaic, true)) {
        std::cerr << "[ERR] buildMosaic failed\n";
        return 6;
    }
    PerfLogger::add("07. buildMosaicGPU", step_timer.stop_ms(), flowGroup);

    // 7) 写出产品
    step_timer.start();
    if (!app.writeProducts(P, grid, mosaic, bounds)) {
        std::cerr << "[ERR] writeProducts failed\n";
        return 7;
    }
    PerfLogger::add("08. Write Products", step_timer.stop_ms(), flowGroup);

    // 记录总耗时
    std::cout << total_timer.stop_ms() << std::endl;

    // 导出主流程图表 (加载、成像、GPU拼接)
    PerfLogger::dump("perf_main_flow.txt", "MainFlow");

    // 导出算法内部耗时分布 (读取、FFT、插值、切片)
    PerfLogger::dump("perf_process_beams_detail.txt", "ProcessAllBeams");

    // 分别画图
    // std::system("python3 ../utils/plot.py perf_main_flow.txt &");
    // std::system("python3 ../utils/plot.py perf_process_beams_detail.txt &");

    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "用法: dbs_core <DBS_parameter.xml>\n";
        return 1;
    }
    return main_dbs_stitch(argv[1]);
}
