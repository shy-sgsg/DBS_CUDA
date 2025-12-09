// src/main.cpp
#include "DbsStitcher.hpp"
#include "DbsTypes.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include <string>


static int main_dbs_stitch(const std::string& xmlPath) {
    // 1) 配置与常量
    DbsStitcher app;
    Params P;
    if (!app.loadParamsFromXml(xmlPath, P)) {
        std::cerr << "[ERR] parse xml failed: " << xmlPath << "\n";
        return 2;
    }
    app.setLonRef(P.lon_ref_deg);

    // 2) 读取POS并估计速度
    PosData POS;
    if (!app.loadPosAndEstimateVelocity(P, POS)) {
        std::cerr << "[ERR] load/estimate POS failed\n";
        return 3;
    }

    // 3) 预估有效方位通道数（避免无效频带）
    int nEff = app.estimateEffectiveAzBins(P, POS);

    // 4) 逐波位读取与成像基本数据（RD切片）
    //    生成/注入脉压参考（频域）
    // 约定：在 DbsStitcher 内部使用该参考（如果你实现了 setHfSpec，可这样传）
    // app.setHfSpec(HfSpec);

#ifdef _OPENMP
    int num_of_threads = std::max(1, omp_get_max_threads() - 1);
    omp_set_num_threads(num_of_threads);
    DBG("OpenMP enabled, max threads = " << num_of_threads);

#endif


    RDData RD;
    RD.nEff = nEff;
    MetaPack meta;
    if (!app.processAllBeams(P, POS, RD, meta)) {
        std::cerr << "[ERR] processAllBeams failed\n";
        return 4;
    }

    if(!app.updateFdCtrEstimates(P, RD, meta, xmlPath)) {
        std::cerr << "[ERR] updateFdCtrEstimates failed\n";
        return 8;
    }

    // 5) 粗投影求拼接范围（降采点投影）
    Bounds bounds;
    Grid grid;
    if (!app.estimateMosaicExtent(P, RD, meta, bounds, grid)) {
        std::cerr << "[ERR] estimateMosaicExtent failed\n";
        return 5;
    }

    // 6) 双线性/最近点插值拼接（严格按原逻辑）
    Mosaic mosaic;
    if (!app.buildMosaic(P, RD, meta, grid, mosaic)) {
        std::cerr << "[ERR] buildMosaic failed\n";
        return 6;
    }

    // 7) 产品写出（raw/png/经纬角点）
    if (!app.writeProducts(P, grid, mosaic, bounds)) {
        std::cerr << "[ERR] writeProducts failed\n";
        return 7;
    }

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
