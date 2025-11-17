from transformers import AutoProcessor
import json
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

model_path = "../Downloads/Models/Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)


with open("../datas/VisuRiddles/syndata.json", "r", encoding="utf-8") as f:
    lines = f.readlines()

lens = []
for _ in lines:
    js = json.loads(_)
    text = js["explanation"]
    lens.append(len(processor.tokenizer(text)["input_ids"]))

data = np.array(lens)

# ====== 创建画布 ======
plt.figure(figsize=(8, 5))

# ====== 直方图（数量 count） ======
counts, bins, patches = plt.hist(data, bins=20, alpha=0.5)

# ====== KDE 平滑曲线 ======
kde = gaussian_kde(data)
xs = np.linspace(data.min(), data.max(), 300)

# 将 KDE 密度转换成 "数量"
bin_width = bins[1] - bins[0]
ys = kde(xs) * len(data) * bin_width

plt.plot(xs, ys)

# ====== 设置对数纵轴（重点） ======
plt.yscale("log")

# ====== 标签与标题 ======
plt.xlabel("Value")
plt.ylabel("Count (log-scale)")
plt.title("Smoothed Histogram (KDE) - Log Scale")

# ====== 保存图像 ======
plt.savefig("smooth_hist_log.png", dpi=300, bbox_inches="tight")
print("图像已保存到 smooth_hist_log.png")
