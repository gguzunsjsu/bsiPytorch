import torch
from datasets import load_dataset
# from evaluate import load as load_metric
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
import time
import datetime
from torch.utils.data import DataLoader, RandomSampler 

dataset = load_dataset('dbpedia_14')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(sample):
  return tokenizer(sample['content'], truncation=True, padding='max_length', max_length=512)
tokenized_datasets = dataset.map(tokenize, batched=True)

# Removing unwanted columns and setting data format to tensor
tokenized_datasets = tokenized_datasets.remove_columns(['content', 'title'])
tokenized_datasets.set_format('torch')

# Preparing data using dataloader
batch_size = 8
train_dataloader = DataLoader(
    dataset=tokenized_datasets['train'],
    sampler=RandomSampler(tokenized_datasets['train']),
    batch_size=batch_size
)

test_dataloader = DataLoader(
    dataset = tokenized_datasets['test'],
    sampler = RandomSampler(tokenized_datasets['test']),
    batch_size = batch_size
)

# Model Initialization
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = 14,
    output_attentions = False,
    output_hidden_states = False
)

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Available device {device}")
model.to(device)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-1, eps = 1e-8)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

#  training loop
def train(model, 
          train_dataloader, 
          loss_function, 
          optimizer,
          device: torch.device):
  total_loss = 0
  t0 = time.time()
  model.train()
  for step, batch in enumerate(train_dataloader):
    if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)
    optimizer.zero_grad()
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

    loss = outputs.loss
    total_loss += loss.item()
    loss.backward()
    optimizer.step()
  return total_loss / len(train_dataloader)

# evaluation loop

def eval(model,
         dataloader,
         device: torch.device):
  model.eval()
  preds = []
  labels = []
  total_loss = 0
  with torch.no_grad():
    for batch in dataloader:
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['label'].to(device)

      outputs = model(input_ids, attention_mask=attention_mask, labels = labels)
      loss = outputs.loss
      total_loss += loss.item()
      
      logits = outputs.logits
      temp_preds = logits.argmax(dim=1)

      preds.extend(temp_preds.cpu().numpy())
      labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(labels, preds)
    return avg_loss, accuracy

epochs = 20
for epoch in range(epochs):

  if epoch==0:
    save_model_name = "bert_dbpedia_e"+str(epoch)+".pth"
    torch.save(model, save_model_name)

  print(f"Epoch {epoch+1}")
  train_loss = train(model,
                     train_dataloader,
                     loss_function,
                     optimizer,
                     device)
  print(f"Traing loss {train_loss:.4f}")

  test_loss, accuracy = eval(model, test_dataloader, device)
  print(f"Test loss {test_loss:.4f} and accuracy {accuracy:.4f}")

  if epoch==(epochs*0.5):
    save_model_name = "bert_dbpedia_e"+str(epoch)+".pth"
    torch.save(model, save_model_name)

  if epoch==epochs-1:
    save_model_name = "bert_dbpedia_e"+str(epoch)+".pth"
    torch.save(model, save_model_name)