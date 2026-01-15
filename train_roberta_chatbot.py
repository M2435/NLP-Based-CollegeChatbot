# Fine-tune RoBERTa for NLP intent classification (fine-tuning script)
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
import joblib

# --------------------------
# Load intents data
# --------------------------
data = pd.read_csv("intents.csv")

# Extract patterns and tags
X = data["pattern"].tolist()
y = data["tag"].tolist()

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# --------------------------
# Tokenizer & Dataset
# --------------------------
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


class ChatDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# Create dataset and dataloader
dataset = ChatDataset(X, y_encoded, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# --------------------------
# Model
# --------------------------
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=len(encoder.classes_)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# --------------------------
# Training loop
# --------------------------
epochs = 8
model.train()

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    running_loss = 0
    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f"Batch {i+1}/{len(dataloader)} - Loss: {running_loss/10:.4f}")
            running_loss = 0

# --------------------------
# Evaluation on training data
# --------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100
print(f"\n✅ Training Accuracy: {accuracy:.2f}%")

# --------------------------
# Save model and label encoder
# --------------------------
model.save_pretrained("roberta_chatbot_model")
tokenizer.save_pretrained("roberta_chatbot_model")
joblib.dump(encoder, "label_encoder.pkl")

print("\n✅ RoBERTa Chatbot model trained, evaluated, and saved!")
