import re
import matplotlib.pyplot as plt

import sys

def read_log(path):
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        print("Failed to read log:", e)
        return ""

if len(sys.argv) >= 2:
    log_data = read_log(sys.argv[1])
else:
    log_data = ""

def plot_perf(data, orientation='h', log_scale=False):
    """
    orientation: 'h' 为横向条形图, 'v' 为纵向直方图/条形图
    log_scale: 是否启用对数刻度
    """
    names = []
    times = []
    
    # 提取 [PERF] 数据
    perf_matches = re.findall(r"\[PERF\].*?\.\s+(.*?):\s+([\d.]+)\s+ms", data)
    for name, time in perf_matches:
        names.append(name)
        times.append(float(time))

    plt.style.use('seaborn-v0_8-muted') # 使用更现代的配色
    fig, ax = plt.subplots(figsize=(12, 7))

    if orientation.lower() == 'h':
        # 横向模式：适合名称较长的情况
        if log_scale:
            ax.set_xscale('log') 
        bars = ax.barh(names, times, color='#5dade2')
        ax.set_xlabel('Time (ms)')
        ax.invert_yaxis() # 从上往下排列步骤 1, 2, 3...
        # 标注数值
        for bar in bars:
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                    f'{bar.get_width():.2f} ms', va='center')
    else:
        # 纵向模式：传统直方图外观
        if log_scale:
            ax.set_yscale('log') 
        bars = ax.bar(names, times, color='#ec7063')
        ax.set_ylabel('Time (ms)')
        plt.xticks(rotation=45, ha='right') # 旋转标签防止重叠
        # 标注数值
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                    f'{bar.get_height():.1f}', ha='center')

    ax.set_title('Performance Analysis')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # default: horizontal bar with log scale to emphasize hotspots
    plot_perf(log_data, orientation='h', log_scale=False)