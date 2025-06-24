import os
import json
import re

# ===== 配置路径 =====
prompt_json_path = "VBench_full_info.json"  # 包含 prompt_en 的 JSON 文件路径
video_dir = "/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_hunyuan/taylorseer"             # 视频所在目录
video_suffix = ".mp4"

# 合法化文件名（移除特殊字符）
def sanitize_filename(name):
    name = re.sub(r'[<>:"/\\|?*]', '', name)  # 去除非法字符
    name = name.strip().replace('\n', '').replace('\r', '')
    return name

# ===== 加载 prompt_en 列表 =====
with open(prompt_json_path, "r", encoding="utf-8") as f:
    prompts = json.load(f)

# ===== 重命名视频文件 =====
for idx, item in enumerate(prompts):
    prompt_text = item["prompt_en"]
    safe_prompt = sanitize_filename(prompt_text)

    old_name = f"{idx}{video_suffix}"
    new_name = f"{safe_prompt}-{idx}{video_suffix}"

    old_path = os.path.join(video_dir, old_name)
    new_path = os.path.join(video_dir, new_name)

    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"✅ {old_name} → {new_name}")
    else:
        print(f"⚠️ 找不到文件: {old_name}")
