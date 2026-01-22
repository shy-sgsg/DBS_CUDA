#pragma once
#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <fstream>
#include <unordered_map>

class PerfLogger {
public:
    // 分组数据结构
    struct GroupData {
        std::unordered_map<std::string, double> data;
        std::vector<std::string> order;
    };

    // 默认分组名称
    static constexpr const char* DEFAULT_GROUP = "default";

    /**
     * @brief 添加耗时记录
     * @param name 条目名称
     * @param ms 耗时(毫秒)
     * @param group 分组名称（如果不指定，则进入 default 分组）
     */
    static void add(const std::string &name, double ms, const std::string &group = DEFAULT_GROUP) {
        std::lock_guard<std::mutex> lk(get().m);
        
        auto &g = get().groups[group]; // 获取或创建分组
        
        if (g.data.find(name) == g.data.end()) {
            g.order.push_back(name);
        }
        g.data[name] += ms;
    }

    /**
     * @brief 将指定分组的日志导出到文件
     * @param path 文件路径
     * @param group 要导出的分组名称
     */
    static void dump(const std::string &path, const std::string &group = DEFAULT_GROUP) {
        std::lock_guard<std::mutex> lk(get().m);
        
        if (get().groups.find(group) == get().groups.end()) return;
        
        std::ofstream f(path);
        if (!f.is_open()) return;

        const auto &g = get().groups[group];
        int idx = 1;
        for (const auto &name : g.order) {
            f << "[PERF] " << idx << ". " << name << ": " << g.data.at(name) << " ms\n";
            ++idx;
        }
        f.close();
    }

    // 辅助计时器
    struct Timer {
        std::chrono::time_point<std::chrono::high_resolution_clock> t0;
        void start() { t0 = std::chrono::high_resolution_clock::now(); }
        double stop_ms() const {
            auto t1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> d = t1 - t0;
            return d.count();
        }
    };

private:
    PerfLogger() {}
    static PerfLogger& get() { static PerfLogger inst; return inst; }

    // 主存储：Group Name -> Group Data
    std::unordered_map<std::string, GroupData> groups;
    std::mutex m;
};