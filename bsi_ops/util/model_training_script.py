import torch
import numpy as np
import pandas as pd
from datasets import load_dataset

dataset = load_dataset('imdb')

from transformers import BertForSequenceClassification, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(sample):
    return tokenizer(sample['text'], padding='max_length', truncation=True, max_length=512)
tokenized_dataset = dataset.map(tokenize, batched=True)

tokenized_dataset = tokenized_dataset.remove_columns(['text'])
tokenized_dataset.set_format('torch')
print(f"Tokenized dataset: {tokenized_dataset}")

from torch.utils.data import DataLoader, RandomSampler

batch_size = 8
train_dataloader = DataLoader(
    tokenized_dataset['train'],
    batch_size=batch_size,
    sampler=RandomSampler(tokenized_dataset['train'])
)

test_dataloader = DataLoader(
    tokenized_dataset['test'],
    batch_size=batch_size,
    sampler=RandomSampler(tokenized_dataset['test'])
)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = 2,
    output_attentions=False,
    output_hidden_states=False
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Available device {device}")
model.to(device)
model_on_device = next(model.parameters()).device
print(f"Model on device{model_on_device}")

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps = 1e-8)

import time
import datetime

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train(model, train_dataloader, optimizer, device):
    t0 = time.time()
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(train_dataloader)

import os
save_model_path = "/home/017510883/RA/bert_imdb_weights_store"
os.makedirs(save_model_path, exist_ok=True)

epochs = 90

for epoch in range(epochs+1):
    if epoch%5 == 0:
        save_model_name = "bert_imdb"+str(epoch)+".pth"
        complete_path = os.path.join(save_model_path, save_model_name)
        torch.save(model, complete_path)
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
    print('Training...')
    train_loss = train(model, train_dataloader, optimizer, device)
    print(f"Loss at {epoch} is {train_loss}")
"""
  if epoch==(epochs//2):
    save_model_name = "bert_imdb"+str(epoch)+".pth"
    complete_path = os.path.join(save_model_path, save_model_name)
    torch.save(model, complete_path)

  if epoch==epochs-1:
    save_model_name = "bert_imdb"+str(epoch)+".pth"
    complete_path = os.path.join(save_model_path, save_model_name)
    torch.save(model, complete_path)
"""