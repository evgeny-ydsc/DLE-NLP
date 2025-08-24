# импортируем библиотеки, которые пригодятся для задачи
import torch
import torch.nn as nn
import re
import random
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# класс датасета
class MaskedBertDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=7):
        self.samples = []
        for line in texts:
            token_ids = tokenizer.encode(line, add_special_tokens=False, 
                                         max_length=512, truncation=True)
            if len(token_ids) < seq_len:
                continue
            for i in range(1, len(token_ids)):
                # набираем подпоследовательности длины seq_len-1,
                # чтобы предсказывать последний токен
                context = token_ids[max(0, i - seq_len): i]
                if len(context) < seq_len:
                    continue
                target = token_ids[i]
                self.samples.append((context, target))
        
    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

# функция для "чистки" текстов
def clean_string(text: str):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_loaders (dataset_path: str, max_texts_count: int = 10000):

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = list(f)

    # длины последовательностей в датасете
    # seq_len = 7 => 6 токенов до <TARGET> + токен <TARGET>
    seq_len = 7

    # удаляем слишком короткие тексты
    texts = [line for line in dataset if len(line.split()) >= seq_len]

    # "чистим" тексты
    cleaned_texts = list(map(clean_string, texts))

    # разбиение на тренировочную и валидационную выборки
    val_size = 0.1
    test_size = 0.1

    train_texts, valtest_texts = train_test_split(cleaned_texts[:max_texts_count],
                                    test_size=(val_size+test_size), random_state=42)

    val_texts, test_texts = train_test_split(valtest_texts,
                                    test_size=test_size/(test_size+val_size)
                                    , random_state=42)

    # Загружаем BERT токенизатор
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # тренировочный и валидационный датасеты
    train_dataset = MaskedBertDataset(train_texts, tokenizer, seq_len=seq_len)
    val_dataset = MaskedBertDataset(val_texts, tokenizer, seq_len=seq_len)
    test_dataset = MaskedBertDataset(test_texts, tokenizer, seq_len=seq_len)

    # даталоадеры
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    return train_loader, val_loader, test_loader, tokenizer, val_texts, test_texts