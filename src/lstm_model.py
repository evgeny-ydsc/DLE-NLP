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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        hidden_out = out[:, -1, :]
        linear_out = self.fc(hidden_out)
        return linear_out

    def generate_output_text(self, tokenizer, input_text: str, max_length: int = 20):
        self.eval()  # Режим инференса
        input_ids = tokenizer.encode(input_text, add_special_tokens=False, truncation=True, max_length=512)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        # Получаем эмбеддинги и прогоняем через LSTM, чтобы получить начальное состояние hidden и cell
        with torch.no_grad():
            emb = self.embedding(input_tensor)
            out, (hidden, cell) = self.lstm(emb)

        generated_ids = input_ids.copy()

        for _ in range(max_length):
            # Подать на вход только последний токен
            last_token_tensor = torch.tensor([[generated_ids[-1]]], dtype=torch.long).to(self.device)

            with torch.no_grad():
                emb = self.embedding(last_token_tensor)
                out, (hidden, cell) = self.lstm(emb, (hidden, cell))
                logits = self.fc(out[:, -1, :])
                probs = F.softmax(logits, dim=-1)
                predicted_id = torch.argmax(probs, dim=-1).item()

            if predicted_id == tokenizer.eos_token_id:
                break

            generated_ids.append(predicted_id)

        # Декодируем только сгенерированную часть, без исходного input_text
        generated_tokens = generated_ids[len(input_ids):]
        out_text = tokenizer.decode(generated_tokens, clean_up_tokenization_spaces=True)
        return out_text