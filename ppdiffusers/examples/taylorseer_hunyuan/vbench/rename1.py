import os
import json
import re

# ===== 配置路径 =====
prompt_json_path = "VBench_full_info.json"  # 包含 prompt_en 的 JSON 文件路径
video_dir = "/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_hunyuan/taylorseer"             # 视频所在目录
video_suffix = ".mp4"

# 合法化文件名（与第一次相同）
def sanitize_filename(name):
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = name.strip().replace('\n', '').replace('\r', '')
    return name

# ===== 加载 prompt_en 列表 =====
with open(prompt_json_path, "r", encoding="utf-8") as f:
    prompts = json.load(f)

# ===== 重新命名所有文件为统一后缀 =====
for idx, item in enumerate(prompts):
    prompt_text = item["prompt_en"]
    safe_prompt = sanitize_filename(prompt_text)

    # 查找当前旧文件名（之前的格式：<prompt>-<idx>.mp4）
    old_name = f"{safe_prompt}-{idx}{video_suffix}"
    new_name = f"{safe_prompt}-0{video_suffix}"

    old_path = os.path.join(video_dir, old_name)
    new_path = os.path.join(video_dir, new_name)

    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"✅ Renamed: {old_name} → {new_name}")
    else:
        print(f"⚠️ File not found: {old_name}")
