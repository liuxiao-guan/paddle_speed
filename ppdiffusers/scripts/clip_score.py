import numpy as np 
import argparse 
import torch 
from coco_evaluator import evaluate_model, compute_clip_score

import pandas as pd
import os
import numpy as np
from PIL import Image
import re
def natural_sort_key(filename):
    """提取文件名中的数字用于自然排序"""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', filename)]

def load_images_to_array(folder_path, target_shape=None):
    """
    参数:
        folder_path: 图片文件夹路径
        target_shape: 可选参数，指定输出数组形状(h,w,3)
    返回:
        np.ndarray: 形状为[image_num, h, w, 3]的数组
    """
    # 获取并排序图片文件
    img_files = [f for f in os.listdir(folder_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    img_files.sort(key=natural_sort_key)
    
    # 预分配数组空间
    first_img = np.array(Image.open(os.path.join(folder_path, img_files[0])))
    if target_shape:
        h, w = target_shape[:2]
    else:
        h, w = first_img.shape[:2]
    
    img_array = np.empty((len(img_files), h, w, 3), dtype=np.uint8)
    
    # 按顺序加载图片
    for i, filename in enumerate(img_files):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path) 
        if target_shape:
            img = img.resize((w, h))
        img_array[i] = np.array(img)
    
    return img_array

@torch.no_grad()
def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_score", action="store_true")
    parser.add_argument("--path", type=str)
    parser.add_argument("--total_eval_number", type=int, default=1000)
    args = parser.parse_args()

    all_captions = []
    df = pd.read_csv(os.path.join("/root/paddlejob/workspace/env_run/test_data/coco1k","coco1k.tsv"), sep="\t")

    # 假设列名为 "prompt"，提取成 list
    all_captions = df['caption_en'].tolist()
    all_images = load_images_to_array(args.path)
    # all_images = np.concatenate(all_images, axis=0)
    print("all_images len ", len(all_images))

    # all_captions = [caption for sublist in all_captions for caption in sublist]
    data_dict = {"all_images": all_images, "all_captions": all_captions}
    if args.clip_score:
        clip_score = compute_clip_score(
            images=data_dict["all_images"],
            captions=data_dict["all_captions"],
            clip_model="ViT-G/14",
            device="cuda",
            how_many=args.total_eval_number
        )
        stats = {
                        "clip_score": float(clip_score)
                    }
    print(stats)

if __name__ == "__main__":
    evaluate() 
