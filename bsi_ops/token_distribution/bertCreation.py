import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

torch.manual_seed(42)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(42)

#Bert Configuration class
class BertConfig:
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        layer_norm_eps=1e-12,
        num_labels=2,  
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size  
        self.num_hidden_layers = num_hidden_layers  
        self.num_attention_heads = num_attention_heads  
        self.hidden_act = hidden_act  
        self.intermediate_size = intermediate_size  
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings  
        self.type_vocab_size = type_vocab_size  
        self.layer_norm_eps = layer_norm_eps
        self.num_labels = num_labels  

#Bert embeddings generation class
class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size) #embeddings should be centered around zero and first four bits to zero and lsb to hold weight --> can this be done?
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        device = input_ids.device

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # [batch_size, seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings  # [batch_size, seq_length, hidden_size]

#Bert attention
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("Hidden size must be divisible by the number of attention heads.")

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Query, Key, Value matrices
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout for attention probabilities
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # x shape: [batch_size, seq_length, all_head_size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # [batch_size, seq_length, num_heads, head_size]
        return x.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_length, head_size]

    def forward(self, hidden_states, attention_mask):
        # Linear projections
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # Transpose for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)  # [batch_size, num_heads, seq_length, head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [batch_size, num_heads, seq_length, seq_length]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply attention mask
        attention_scores = attention_scores + attention_mask

        # Normalize to probabilities
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # Compute context layer
        context_layer = torch.matmul(attention_probs, value_layer)  # [batch_size, num_heads, seq_length, head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_length, num_heads, head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # [batch_size, seq_length, hidden_size]

        return context_layer

#self attention

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        # Linear layer after attention
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Layer normalization and dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # Apply linear layer
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # Add residual connection and layer normalization
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


#attention score -> end attention output
class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask):
        # Self-attention
        self_outputs = self.self(hidden_states, attention_mask)
        # Apply output layer
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        # Feedforward layer
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # Activation function
        if config.hidden_act == "gelu":
            self.intermediate_act_fn = F.gelu
        elif config.hidden_act == "relu":
            self.intermediate_act_fn = F.relu
        else:
            raise ValueError("Unsupported activation function: {}".format(config.hidden_act))

    def forward(self, hidden_states):
        # Apply feedforward and activation
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        # Linear layer
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # Layer normalization and dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # Apply linear layer
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # Add residual connection and layer normalization
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        # Attention layer
        self.attention = BertAttention(config)
        # Intermediate layer
        self.intermediate = BertIntermediate(config)
        # Output layer
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        # Attention output
        attention_output = self.attention(hidden_states, attention_mask)
        # Intermediate output
        intermediate_output = self.intermediate(attention_output)
        # Layer output
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        # Stack of transformer layers
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        # Iterate over transformer layers
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states  # [batch_size, seq_length, hidden_size]

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        # Linear layer to project [CLS] token's hidden state
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Activation function (tanh)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # Take the hidden state corresponding to [CLS] token (first token)
        first_token_tensor = hidden_states[:, 0]
        # Apply linear layer and activation
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output  # [batch_size, hidden_size]

class BertModel(nn.Module):
    def __init__(self, config):
        super(BertModel, self).__init__()
        # Embeddings
        self.embeddings = BertEmbeddings(config)
        # Encoder (transformer stack)
        self.encoder = BertEncoder(config)
        # Pooler
        self.pooler = BertPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        # Convert attention mask to float
        attention_mask = attention_mask.to(dtype=torch.float32)
        # Expand dimensions for broadcasting
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_length]
        # Apply mask to attention scores (transform mask values)
        attention_mask = (1.0 - attention_mask) * -10000.0

        # Get embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)
        # Pass through encoder
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        # Get pooled output
        pooled_output = self.pooler(encoder_outputs)
        return encoder_outputs, pooled_output  # encoder_outputs: [batch_size, seq_length, hidden_size], pooled_output: [batch_size, hidden_size]

class BertForSequenceClassification(nn.Module):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__()
        # BERT model
        self.bert = BertModel(config)
        # Classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config  # Save config for use in forward

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # Get outputs from BERT
        encoder_outputs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        # Get logits
        logits = self.classifier(pooled_output)

        if labels is not None:
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits

# Load the IMDB dataset
dataset = load_dataset('imdb')

# Split into training and test sets
train_dataset = dataset['train']
test_dataset = dataset['test']


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

max_seq_length = 256 #following sliding window technique of bert

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt',  
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),  
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_data_loader(dataset, tokenizer, max_len, batch_size):
    ds = IMDBDataset(
        texts=dataset['text'],
        labels=dataset['label'],
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True
    )

batch_size = 16

train_data_loader = create_data_loader(train_dataset, tokenizer, max_seq_length, batch_size)
test_data_loader = create_data_loader(test_dataset, tokenizer, max_seq_length, batch_size)

config = BertConfig()

model = BertForSequenceClassification(config)

model = model.to(device)


import torch.optim as optim

optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Number of training steps
total_steps = len(train_data_loader) * 3

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)


def accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def train_epoch(
    model,
    data_loader,
    optimizer,
    device,
    scheduler
):
    model = model.train()
    losses = []
    acc = []

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss, logits = outputs
        losses.append(loss.item())
        acc.append(accuracy(logits.detach().cpu().numpy(), labels.cpu().numpy()))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return np.mean(losses), np.mean(acc)

def eval_model(model, data_loader, device):
    model = model.eval()
    losses = []
    acc = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss, logits = outputs
            losses.append(loss.item())
            acc.append(accuracy(logits.detach().cpu().numpy(), labels.cpu().numpy()))

    return np.mean(losses), np.mean(acc)

epochs = 30

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    train_loss, train_acc = train_epoch(
        model,
        train_data_loader,
        optimizer,
        device,
        scheduler
    )

    print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f}')

    val_loss, val_acc = eval_model(
        model,
        test_data_loader,
        device
    )

    print(f'Val   loss {val_loss:.4f} accuracy {val_acc:.4f}')

    # Adjust learning rate if using scheduler
    scheduler.step()

