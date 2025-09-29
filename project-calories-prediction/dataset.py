from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import timm
from transformers import AutoTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import numpy as np

class MealDataset(Dataset):
    def __init__(self, df, config, transforms):
        self.df = df
        self.transforms = transforms
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        ingredients = self.df.iloc[idx]["ingredients"]
        calories = self.df.iloc[idx]["total_calories"]
        mass = self.df.iloc[idx]["total_mass"]

        img_path = os.path.join(self.config.ROOT_PATH, self.config.IMAGES_PATH, 
                                self.df.iloc[idx]["dish_id"], 'rgb.png')
        
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image=np.array(image))["image"]

        return {"calories": calories, "mass": mass,
                "image": image, "text": ingredients}


def collate_fn(batch, tokenizer):
    texts = [item["text"] for item in batch]
    images = torch.stack([item["image"] for item in batch])
    calories = torch.FloatTensor([item["calories"] for item in batch])
    mass = torch.FloatTensor([item["mass"] for item in batch])

    tokenized_input = tokenizer(texts,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=True)
    return {
        "mass": mass,
        "calories": calories,
        "image": images,
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"]
    }

def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
    if ds_type == "train":
        transforms  = A.Compose([
                A.SmallestMaxSize(max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.CenterCrop(height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Affine(scale=(0.8, 1.2),
                        rotate=(-15, 15),
                        translate_percent=(-0.1, 0.1),
                        shear=(-10, 10),
                        fill=0,
                        p=0.2),
                A.CoarseDropout(num_holes_range=(2, 8),
                                hole_height_range=(int(0.07 * cfg.input_size[1]),
                                                int(0.15 * cfg.input_size[1])),
                                hole_width_range=(int(0.1 * cfg.input_size[2]),
                                                int(0.15 * cfg.input_size[2])),
                                fill=0,
                                p=0.3),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
                A.SaltAndPepper(amount=(0.01, 0.06), p=0.2),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                ToTensorV2(p=1.0) #######################
            ])
    else:
        transforms  = A.Compose([
                A.SmallestMaxSize(max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.CenterCrop(height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                ToTensorV2(p=1.0) #######################
            ])

    return transforms
