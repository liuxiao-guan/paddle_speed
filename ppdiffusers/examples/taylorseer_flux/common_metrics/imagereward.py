import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import ImageReward as RM

class CocoDataset(Dataset):
    def __init__(self, image_folder, prompts_file, transforms, tokenizer):
        self.image_folder = image_folder
        self.transforms = transforms
        self.tokenizer = tokenizer
        
        # Load prompts from file
        with open(prompts_file, 'r') as f:
            self.prompts = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, f"{idx}.png")
        
        img = self.transforms(Image.open(image_path))
        caption = self.prompts[idx]
        tokenized_caption = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35, return_tensors="pt")
        
        return img, tokenized_caption

def collate_eval(batch):

    images = torch.stack([sample[0] for sample in batch])
    captions = {k: torch.cat([sample[1][k] for sample in batch]) for k in batch[0][1].keys()}
    return images, captions

def evaluate_coco(prompt_path, img_path, model, batch_size, preprocess_val, tokenizer, device, score_save_path):
    dataset = CocoDataset(img_path, prompt_path, preprocess_val, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_eval)

    score_list = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            images, texts = batch
            images = images.to(device, non_blocking=True)
            texts = {k: v.to(device, non_blocking=True) for k, v in texts.items()}

            with torch.cuda.amp.autocast():
                
                image_embeds = model.blip.visual_encoder(images)
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

                text_output = model.blip.text_encoder(
                    texts['input_ids'],
                    attention_mask=texts['attention_mask'],
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True
                )
                
                txt_features = text_output.last_hidden_state[:, 0, :].float()
                rewards = model.mlp(txt_features)
                rewards = (rewards - model.mean) / model.std

                score_list.extend(rewards.detach().cpu().numpy().flatten())

    with open(score_save_path, "w") as score_file:
        for score in score_list:
            score_file.write(f"{score}\n")

    # avg_score = np.mean(score_list) if score_list else 0
    total_score = sum(score_list)  # 计算所有分数的总和
    num_scores = len(score_list)    # 获取分数的数量
    avg_score = total_score / num_scores if num_scores > 0 else 0  # 计算平均分数，避免除以零
    with open(score_save_path, "a") as score_file:
        score_file.write(f"{avg_score}\n")
    # print(f"Average score: {avg_score}")
    print(f"Average score: {avg_score:.4f}")
    return score_list, avg_score

def load_model(model_path, device):
    model = RM.load(model_path)
    model = model.to(device)
    model.eval()
    return model
model_path = '/root/.cache/ImageReward/ImageReward.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model(model_path, device)
tokenizer = model.blip.tokenizer
preprocess = model.preprocess

def evaluate(mode: str, prompt_path: str, image_path: str, model_path: str, batch_size: int = 20, score_save_path: str = None) -> None:
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = load_model(model_path, device)
    # tokenizer = model.blip.tokenizer
    # preprocess = model.preprocess

    if mode == 'coco':
        score_list, avg_score = evaluate_coco(prompt_path, image_path, model, batch_size, preprocess, tokenizer, device, score_save_path)
        return score_list, avg_score
    else:
        raise NotImplementedError

if __name__ == '__main__':
    # prompt_path = '/root/autodl-fs/CODE/SD_CFG/EXP-seed/search_by_semantic/data_prompts/DrawBench.txt'
    prompt_path = '/root/paddlejob/workspace/env_run/gxl/paddle_speed/ppdiffusers/examples/taylorseer_flux/prompts/DrawBench.txt'

    import os
    import csv

    # image_dir_base ="/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16"
    # image_dirs_list = ["DrawBench_blockdance","DrawBench_firstblock_predicterror_taylor0.03", \
    # "DrawBench_firstblock_predicterror_taylor0.13","DrawBench_pab","DrawBench_taylorseer_N2","DrawBench_taylorseer_N5"]
    image_dir_base ="/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16"
    image_dirs_list = ["DrawBench_firstblock_predicterror_taylor0.08","DrawBench_timeemb_predicterror"]
    i =0
    for image_dir in image_dirs_list:
        # i = i +1
        # if i==2:
        #     break
        image_dir = os.path.join(image_dir_base, image_dir)
    
        mode = 'coco'
        output_dir = '/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/eval/DBresults'

        os.makedirs(output_dir, exist_ok=True)
        class_name = os.path.basename(image_dir)
        #class_name = image_dir.split('/')[-5] + image_dir.split('/')[-4] + '-' + image_dir.split('/')[-2]
        # class_name = image_dir.split('/')[-4] + '-' + image_dir.split('/')[-2]

        score_save_path = os.path.join(output_dir, f'{class_name}.txt')

        score_list, avg_score = evaluate(mode, prompt_path, image_dir, model_path="/root/.cache/ImageReward/ImageReward.pt", batch_size=100, score_save_path=score_save_path)
