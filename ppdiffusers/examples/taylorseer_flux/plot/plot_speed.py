import matplotlib.pyplot as plt
import pandas as pd

# 构造表格数据
data = {
    "method": [
        "PAB", "PAB", "PAB",
        "teacache", "teacache", "teacache",
        "taylorseer", "taylorseer", "taylorseer",
        "ours", "ours", "ours"
    ],
    "speedup_ratio": [
        1.37, 1.57, 1.83,
        1.44, 1.73, 2.29,
        1.71, 2.04, 2.83,
        2.06, 2.45, 3.23
    ],
    "ssim": [
        0.914, 0.849, 0.485,
        0.8114, 0.805, 0.716,
        0.85, 0.763, 0.648,
        0.937, 0.897, 0.813
    ]
}

df = pd.DataFrame(data)

# 设置图像大小
plt.figure(figsize=(8, 6))

# 绘图
for method, group in df.groupby("method"):
    color = "red" if method == "ours" else "gray"
    linestyle = "-" if method == "ours" else "--"
    label = method if method == "ours" else None  # 避免重复标签

    plt.plot(
        group["speedup_ratio"],
        group["ssim"],
        label=label,
        color=color,
        linestyle=linestyle,
        marker='o',
        linewidth=2
    )

# 添加图例、标签、网格
plt.xlabel("Speedup Ratio")
plt.ylabel("SSIM")
plt.title("SSIM vs Speedup Ratio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("plot_speed.png")
