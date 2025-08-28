import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        hidden_out = out[:, -1, :]
        linear_out = self.fc(hidden_out)
        return linear_out

    def generate_output_text (self, tokenizer, input_text:str, max_length:int=20):
        out_text = input_text
        for i in range(max_length):
            token_ids = tokenizer.encode(out_text, add_special_tokens=False, 
                                         max_length=512, truncation=True)
            input_tensor = torch.tensor([token_ids], dtype=torch.long)
            linear_out = self.forward (input_tensor)
            probs = F.softmax(linear_out, dim=-1)
            predicted_id = torch.argmax(probs, dim=-1)
            predicted_token = tokenizer.decode(predicted_id.tolist())
            if predicted_id.item() == tokenizer.eos_token_id:
                break
            out_text += predicted_token
        return out_text[len(input_text):]
