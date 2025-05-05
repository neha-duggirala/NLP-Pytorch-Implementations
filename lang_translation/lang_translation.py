import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
from datasets import load_dataset
import torch.optim as optim

# Load Hugging Face Tokenizers
tokenizer_es = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
tokenizer_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")

# Load Dataset
dataset = load_dataset("multi30k", split={"train": "train", "validation": "validation", "test": "test"})

# Preprocessing Function
def preprocess_function(examples):
    inputs = [ex for ex in examples["translation"]["es"]]
    targets = [ex for ex in examples["translation"]["en"]]
    model_inputs = tokenizer_es(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer_en(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize Dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Data Collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer_es, model="Helsinki-NLP/opus-mt-es-en")

# Create DataLoaders
BATCH_SIZE = 128
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=BATCH_SIZE, collate_fn=data_collator)
valid_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=BATCH_SIZE, collate_fn=data_collator)
test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=BATCH_SIZE, collate_fn=data_collator)

# Initialize Model
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define Optimizer and Loss
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Training Loop
def train(model, dataloader, optimizer):
    model.train()
    epoch_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

# Evaluation Loop
def evaluate(model, dataloader):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

N_EPOCHS = 10

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_dataloader, optimizer)
    valid_loss = evaluate(model, valid_dataloader)

    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')