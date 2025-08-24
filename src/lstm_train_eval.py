from src.lstm_model import LSTMGenerator
import evaluate
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm

class ModelTrainer ():
    def __init__(self, model:LSTMGenerator, tokenizer):
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=0.002)
        self.criterion = nn.CrossEntropyLoss()
        self.tokenizer = tokenizer

    def evaluate(self, loader):
        self.model.eval()

        correct, total = 0, 0
        sum_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch, y_batch
                x_output = self.model(x_batch)
                loss = self.criterion(x_output, y_batch)
                preds = torch.argmax(x_output, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                sum_loss += loss.item()
        return sum_loss / len(loader), correct / total, 

    def evaluate_on_full_texts(self, texts):
        rouge = evaluate.load("rouge")
        rouge1 = []
        rouge2 = []
        self.model.eval()
        for line in texts:
            l = len(line)
            i = int(l*0.75)
            start = line [:i]
            finish = line [i:]
            predicted_finish = self.model.generate_output_text(start, l-i)
            results = rouge.compute(predictions=[predicted_finish], references=[finish])
            rouge1.append(results["rouge1"])
            rouge2.append(results["rouge2"])
        return np.mean (rouge1), np.mean (rouge2)


    def train(self, train_loader, val_loader, val_texts, n_epochs=3):
        for epoch in range(n_epochs):
            self.model.train()
            train_loss = 0
            for x_batch, y_batch in tqdm(train_loader):
                x_batch, y_batch = x_batch, y_batch
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(x_batch), y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            rouge1, rouge2 = self.evaluate_on_full_texts (val_texts)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} "
                  f"| Val Loss: {val_loss:.3f} | Val Accuracy: {val_acc:.2%}"
                  f"| rouge1: {rouge1:.3f} | rouge2: {rouge2:.3f}")