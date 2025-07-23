import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.ticker as mticker
fontpath ="/root/paddlejob/workspace/env_run/gxl/paddle_speed/ppdiffusers/examples/taylorseer_flux/plot/times.ttf"
prop = font_manager.FontProperties(fname=fontpath)
plt.rcParams["font.family"] = "Times New Roman"
print(plt.rcParams['font.family'])
# 读取 Excel 文件（你需要将文件名替换成自己的）
df = pd.read_excel("/root/paddlejob/workspace/env_run/gxl/paddle_speed/ppdiffusers/examples/taylorseer_flux/plot/plot_speed.xlsx")  # 例如 "results.xlsx"

import matplotlib.pyplot as plt
import pandas as pd
import os

# 设置字体（可选）
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20
plt.rcParams["axes.titlesize"] =18
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 14
# 柔和的浅色调（可扩展）
pastel_colors = [
    "#aec7e8",  # 浅蓝
    "#c5b0d5",  # 浅紫
    "#98df8a",  # 浅绿
    "#ffdb99",  # 浅黄
    "#f7b6d2",  # 粉色
    "#c7c7c7",  # 灰色
]

# 创建方法到颜色的映射，ours 特别设置为深红
color_map = {}
methods = df["method"].unique()

for i, method in enumerate(methods):
    if method == "Ours":
        color_map[method] = "#d62728"  # 高亮深红
    else:
        color_map[method] = pastel_colors[i % len(pastel_colors)]  # 自动循环使用浅色

# 开始绘图
plt.figure(figsize=(8, 6))

for method, group in df.groupby("method"):
    color = color_map[method]
    linestyle = "-" if method == "ours" else "--"
    linestyle =  "--"
    label = method 

    plt.plot(
        group["speedup ratio"],
        group["ssim"],
        label=label,
        color=color,
        linestyle=linestyle,
        marker='o',
        linewidth=2
    )

plt.xlabel("Speedup Ratio")
plt.ylabel("SSIM↑")
# plt.title("SSIM vs Speedup Ratio")
plt.legend(loc='lower right')
plt.grid(True,linestyle="--",linewidth=0.5, alpha=0.7)
plt.tight_layout()

# 保存图像
os.makedirs("./plot", exist_ok=True)
plt.savefig("./plot/plot_speed_ssim.png", dpi=300,bbox_inches='tight')
plt.show()


# 开始绘图
plt.figure(figsize=(8, 6))

for method, group in df.groupby("method"):
    color = color_map[method]
    linestyle = "-" if method == "ours" else "--"
    linestyle =  "--"
    label = method 

    plt.plot(
        group["speedup ratio"],
        group["psnr"],
        label=label,
        color=color,
        linestyle=linestyle,
        marker='o',
        linewidth=2
    )

plt.xlabel("Speedup Ratio")
plt.ylabel("PSNR↑")
# plt.title("SSIM vs Speedup Ratio")
plt.legend(loc='lower right')
plt.grid(True,linestyle="--",linewidth=0.5, alpha=0.7)
plt.tight_layout()

# 保存图像
os.makedirs("./plot", exist_ok=True)
plt.savefig("./plot/plot_speed_psnr.png", dpi=300,bbox_inches='tight')
plt.show()


# --------- 1⃣ 创建上下子图并准备数据，保持不变 ----------
fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1, sharex=True, figsize=(8, 6),
    gridspec_kw={'height_ratios': [3, 1]}
)

pos_vals = df.loc[df["image reward"] > 0, "image reward"]
neg_vals = df.loc[df["image reward"] < 0, "image reward"]

# --------- 2⃣ 画折线（循环同前），保持不变 ----------
for method, group in df.groupby("method"):
    color = color_map[method]
    linestyle = "-" if method == "ours" else "--"
    marker = 'o'
    lw = 2

    ax_top.plot(group["speedup ratio"], group["image reward"],
                label=method, color=color,
                linestyle=linestyle, marker=marker, linewidth=lw)

    ax_bottom.plot(group["speedup ratio"], group["image reward"],
                   label=method, color=color,
                   linestyle=linestyle, marker=marker, linewidth=lw)

# --------- 3⃣ 只给上轴设置网格 ----------
ax_top.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
ax_bottom.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

fmt = mticker.FormatStrFormatter('%.3f')     # 3 位小数
ax_top.yaxis.set_major_formatter(fmt)
ax_bottom.yaxis.set_major_formatter(fmt)
# --------- 4⃣ 轴范围、断轴、锯齿，同前 ----------
ax_top.set_ylim(pos_vals.min() * 0.995, pos_vals.max() * 1.005)
ax_bottom.set_ylim(neg_vals.min() * 1.005, neg_vals.max() * 0.995)

ax_top.spines['bottom'].set_visible(False)
ax_bottom.spines['top'].set_visible(False)
# ax_top.tick_params(labeltop=True)
ax_bottom.tick_params()
ax_bottom.xaxis.tick_bottom()

d = .015
kwargs = dict(color='k', clip_on=False, transform=ax_top.transAxes)
ax_top.plot((-d, +d), (-d, +d), **kwargs)
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

kwargs.update(transform=ax_bottom.transAxes)
ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# --------- 5⃣ 只保留一个 y 轴标签 ----------
fig.text(0.01, 0.5, "Image Reward↑", va='center', rotation='vertical', fontsize=20)
ax_bottom.set_ylabel("")           # 下轴置空
ax_bottom.set_xlabel("Speedup Ratio")

# --------- 6⃣ 图例与布局 ----------
# ax_top.legend(ncol=len(df["method"].unique()),
#               loc="upper center", bbox_to_anchor=(0.5, 1.15))
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
plt.savefig("./plot/plot_speed_imagereward.png", dpi=300,bbox_inches='tight')