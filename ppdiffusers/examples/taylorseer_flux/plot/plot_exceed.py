import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.ticker as mticker
bold_font_path ="/root/paddlejob/workspace/env_run/gxl/paddle_speed/ppdiffusers/examples/taylorseer_flux/plot/times.ttf"
font_manager.fontManager.addfont(bold_font_path)
bold_font = font_manager.FontProperties(fname=bold_font_path,size=26)
print(bold_font.get_name())
# 设置字体（可选）
plt.rcParams['font.family'] = bold_font.get_name()  # 例如 'Times New Roman'
plt.rcParams["font.weight"] = "bold"   
plt.rcParams["font.size"] = 18
plt.rcParams["axes.titlesize"] =18
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16
# 示例数据，你需要替换成自己的统计结果
epsilons = [0.03, 0.08, 0.13, 0.18]
steps_below = [27, 35, 37, 39]  # 小于等于 epsilon 的步数（如果你也想画出来）  
steps_above = [23, 15, 13, 11]  # 超过 epsilon 的步数

x = np.arange(len(epsilons))  # 横轴位置

# 设置柱状图宽度
bar_width = 0.25

plt.figure(figsize=(6,5))

# 画两组柱状图：超过 vs 小于
bars_above = plt.bar(x - bar_width/2, steps_above, width=bar_width, label="steps exceeding ε", color="#6BAED6")
bars_below = plt.bar(x + bar_width/2, steps_below, width=bar_width, label="steps below ε", color="#FDAE6B")
# bars_above = plt.bar(x - bar_width/2, steps_above, width=bar_width, label="steps exceeding ε", color="#FBD9D5")
# bars_below = plt.bar(x + bar_width/2, steps_below, width=bar_width, label="steps below ε", color="#DAE8FC")

# 给每个柱子显示数值
for bar in bars_above:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.15, str(height),
             ha='center', va='bottom', fontsize=12)

for bar in bars_below:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.15, str(height),
             ha='center', va='bottom', fontsize=12)

# 美化
plt.xticks(x, [f"{e:.2f}" for e in epsilons])   # 横轴显示 ε 值
plt.xlabel("Confidence Threshold ε")
plt.ylabel("Number of Steps")
# plt.title("Steps above and below ε")
plt.legend()
# 设置图例在图上方
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.13),  # 1.05~1.2 之间调整高度
    ncol=2,
    frameon=False
)
# plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
plt.grid(True,linestyle="--",linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig("./plot/steps_exceed.png",dpi=600)
