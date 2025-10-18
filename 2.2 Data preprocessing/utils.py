import re
import os

import timm
import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from torchmetrics import MeanAbsoluteError
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchinfo import summary
from functools import partial

from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import MealDataset, collate_fn, get_transforms
from config import Config
#from drive.MyDrive.ydsc.dataset import MealDataset, collate_fn, get_transforms
#from drive.MyDrive.ydsc.config import Config


class MealMultimodalModel(nn.Module):
    def __init__(self, config, print_model_info=False):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0 
        )
        #заморозка:
        for param in self.image_model.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False

        self.text_proj = nn.Sequential(
            nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM),
            nn.Dropout(0.3))

        self.image_proj = nn.Sequential(
            nn.Linear(self.image_model.num_features, config.HIDDEN_DIM),
            nn.Dropout(0.3))

        self.regressor = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM*2+1, config.HIDDEN_DIM // 2),
            nn.BatchNorm1d(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.HIDDEN_DIM // 2, config.HIDDEN_DIM // 4),
            nn.BatchNorm1d(config.HIDDEN_DIM // 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.HIDDEN_DIM // 4, 1) # 1- регрессия
        )

    def forward(self, input_ids, attention_mask, image, mass):
        text_features = self.text_model(input_ids, attention_mask) \
                                .last_hidden_state[:,  0, :]
        image_features = self.image_model(image)
        scaled_mass = mass / 2_000.0 # из EDA знаем, что масса блюда была 1102

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)
        
        fused_emb = torch.cat([scaled_mass.unsqueeze(1),
                               text_emb, image_emb], dim=1)
        
        output = self.regressor(fused_emb)
        return output

def train(config, train_dish_df, test_dish_df):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Инициализация модели
    model = MealMultimodalModel(config, print_model_info=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    

    # Оптимизатор с разными LR
    optimizer = AdamW([
        {'params': model.text_model.parameters(), 'lr': config.TEXT_LR, 
         'weight_decay': config.WEIGHT_DECAY},
        {'params': model.image_model.parameters(), 'lr': config.IMAGE_LR, 
         'weight_decay': config.WEIGHT_DECAY},
        {'params': model.text_proj.parameters(), 'lr': config.REGRESSOR_LR, 
         'weight_decay': config.WEIGHT_DECAY},
        {'params': model.image_proj.parameters(), 'lr': config.REGRESSOR_LR, 
         'weight_decay': config.WEIGHT_DECAY},
        {'params': model.regressor.parameters(), 'lr': config.REGRESSOR_LR, 
         'weight_decay': config.WEIGHT_DECAY}
    ])

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.SCHEDULER_T_MAX,
        eta_min=config.SCHEDULER_ETA_MIN)
    
    criterion = nn.L1Loss()
    
    # Загрузка данных
    transforms = get_transforms(config)
    val_transforms = get_transforms(config, ds_type="val")
    train_dataset = MealDataset(train_dish_df, config, transforms)
    val_dataset = MealDataset(test_dish_df, config, val_transforms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )
    
    # инициализируем метрику
    mae_metric = MeanAbsoluteError().to(device)
    best_mae = float('inf')
    train_mae_all = []
    val_mae_all = []

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0
        mae_metric.reset()
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True) as pbar:
            for batch_idx, batch in enumerate(pbar, start=1):
                # Подготовка данных
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                    'image': batch['image'].to(device),
                    'mass': batch['mass'].to(device)
                }
                targets = batch['calories'].to(device).float()
                targets = targets.squeeze(-1)
                
                # Forward
                optimizer.zero_grad()
                outputs  = model(**inputs)
                outputs = outputs.squeeze(-1)
                loss = criterion(outputs, targets)
                
                # Backward
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                batch_loss = loss.item()
                total_loss += batch_loss
                mae_metric.update(outputs, targets)
                pbar.set_postfix({'batch': batch_idx, 'loss': f"{batch_loss:.4f}"})

        # Валидация
        train_mae = mae_metric.compute().item()
        mae_metric.reset()
        
        val_mae = validate(model, val_loader)
        mae_metric.reset()

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Train MAE: {train_mae:.4f} | "
            f"Val MAE: {val_mae:.4f}"
        )
        train_mae_all.append(train_mae)
        val_mae_all.append(val_mae)

        if val_mae < best_mae:  # Минимизируем MAE
            best_mae = val_mae
            torch.save(model.state_dict(), config.SAVE_PATH)
            if val_mae <= config.TARGET_MAE:
                print(f'Критерий успеха достигнут на эпохе {epoch+1}')
                break

    # Преобразуем потери в обычные числа
    train_mae_float = [m.item() if isinstance(m, torch.Tensor) else m for m in train_mae_all]
    val_mae_float = [m.item() if isinstance(m, torch.Tensor) else m for m in val_mae_all]

    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(train_mae_float)+1), train_mae_float, label='Train MAE')
    plt.plot(range(1, len(val_mae_float)+1), val_mae_float, label='Validation MAE')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.title('График MAE на тренировке и валидации')
    plt.legend()
    plt.grid(True)
    plt.show()

def validate(model, val_loader):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mae_metric = MeanAbsoluteError().to(device)

    model.eval()
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device)
            }
            targets = batch['calories'].to(device).float()
            targets = targets.squeeze(-1) ###########
                        
            outputs = model(**inputs)
            outputs = outputs.squeeze(-1) ###########
            
            mae_metric.update(outputs, targets)
    
    return mae_metric.compute().cpu().numpy() 


def inference(config, test_dish_df):

    all_calories = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #загрузим модель из указанного в конфиге файла:
    model = MealMultimodalModel(config)
    model.load_state_dict(torch.load(config.SAVE_PATH, map_location=device))
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    
    # Загрузка данных
    val_transforms = get_transforms(config, ds_type="val")
    val_dataset = MealDataset(test_dish_df, config, val_transforms)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )

    mae_metric = MeanAbsoluteError().to(device)
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device)
            }
            targets = batch['calories'].to(device).float()
            targets = targets.squeeze(-1)
                        
            outputs = model(**inputs)
            outputs = outputs.squeeze(-1)
            all_calories.append (outputs)
    
    all_calories = torch.cat(all_calories).cpu().numpy()
    return all_calories
