from torch.utils.data import Dataset
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch import nn

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]

        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

 
class LanguageAwareTextClassificationDataset(Dataset):
    def __init__(self, texts, labels, lang_ids, tokenizer, max_length):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.lang_ids = lang_ids.reset_index(drop=True)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        lang_id = self.lang_ids.iloc[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "lang_ids": torch.tensor(lang_id, dtype=torch.long),
        }
    

class LanguageAwareClassifier(nn.Module):
    def __init__(self, model_name, num_labels, num_languages):
        super().__init__()

        self.encoder = AutoModelForSequenceClassification.from_pretrained(model_name, use_safetensors=True)

        hidden_size = self.encoder.config.hidden_size

        self.lang_embedding = nn.Embedding(num_languages, 32)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 32, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask, lang_ids):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lang_ids=lang_ids
        )

        text_repr = outputs.last_hidden_state[:, 0]  # CLS token
        lang_repr = self.lang_embedding(lang_ids)

        combined = torch.cat([text_repr, lang_repr], dim=1)

        return self.classifier(combined)
   

