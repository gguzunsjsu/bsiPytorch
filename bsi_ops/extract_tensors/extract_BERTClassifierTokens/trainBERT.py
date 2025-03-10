from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle, os

print("Import success")

def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

# Load pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# Read the train.txt file
with open('train.txt', 'r') as file:
    lines = file.readlines()

# Split sentences and labels
sentences = [line.split('.')[0].strip() for line in lines]
labels = [1 if 'Positive' in line else 0 for line in lines]

# Split into train and test sets
train_sentences, test_sentences, train_labels, test_labels = train_test_split(sentences, labels, test_size=0.2, random_state=42)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'label': self.labels[idx]}

train_dataset = SentimentDataset(train_sentences, train_labels)
test_dataset = SentimentDataset(test_sentences, test_labels)

print("Dataset created successfully")

# Tokenize and encode the dataset
train_encodings = tokenizer(train_dataset.texts, truncation=True, padding=True, return_tensors='pt', max_length=512)
test_encodings = tokenizer(test_dataset.texts, truncation=True, padding=True, return_tensors='pt', max_length=512)

# Convert labels to tensors
train_labels = torch.tensor(train_dataset.labels)
test_labels = torch.tensor(test_dataset.labels)

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(**train_encodings, labels=train_labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Inference
model.eval()
with torch.no_grad():
    outputs = model(**test_encodings)
    predictions = torch.argmax(outputs.logits, dim=1)

# Calculate accuracy
accuracy = accuracy_score(test_labels.numpy(), predictions.numpy())
print(f"Accuracy: {accuracy}")

# Trained the BERT model. Now would like to extract Q,K,V as before.
# Read the contents of the text file into a variable
with open("test.txt", 'r') as file:
    input_lines = file.readlines()

# Tokenize and classify each line in the input file
true_labels = []
predicted_labels = []
batch_size = 32

for i in range(0, len(input_lines), batch_size):
    batch_lines = input_lines[i:i + batch_size]

    # Extract text and sentiment from each line in the batch
    batch_texts = [line.strip().rsplit('.', 1) for line in batch_lines]
    batch_true_labels = [1 if 'Positive' in sentiment else 0 for text, sentiment in batch_texts]
    true_labels.extend(batch_true_labels)

    # Tokenize the batch
    batch_inputs = tokenizer([text for text, sentiment in batch_texts], return_tensors="pt", padding=True,
                             truncation=True)

    # Perform inference on the batch
    batch_outputs = model(**batch_inputs, output_hidden_states=True)
    batch_logits = batch_outputs.logits
    batch_last_hidden_state = batch_outputs.hidden_states[-1]

    # Extract the Q, K, and V vectors at each layer for the batch
    all_q_vectors = []
    all_k_vectors = []
    all_v_vectors = []

    for layer in model.bert.encoder.layer:  # Modify this line
        q_vectors = layer.attention.self.query(batch_last_hidden_state)
        k_vectors = layer.attention.self.key(batch_last_hidden_state)
        v_vectors = layer.attention.self.value(batch_last_hidden_state)

        all_q_vectors.append(q_vectors)
        all_k_vectors.append(k_vectors)
        all_v_vectors.append(v_vectors)

    # Extract predicted classes
    batch_predicted_classes = torch.argmax(batch_logits, dim=1).tolist()
    predicted_labels.extend(batch_predicted_classes)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

# Print the accuracy
print(f"Accuracy: {accuracy:.2f}")


# Print the shapes of the extracted Q, K, and V tensors
for i, (q, k, v) in enumerate(zip(all_q_vectors, all_k_vectors, all_v_vectors), 1):
    print(f"Layer {i} - Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")

# Store these tensors in a list
triplets = []

for i in range(len(model.bert.encoder.layer)):
    # Flatten the tensors to [32, 52 * 1024]
    Q_flat = all_q_vectors[i].reshape(32, -1)
    K_flat = all_k_vectors[i].reshape(32, -1)
    V_flat = all_v_vectors[i].reshape(32, -1)

    # Normalize the flattened tensors using min-max normalization
    Q_normalized = min_max_normalize(Q_flat)
    K_normalized = min_max_normalize(K_flat)
    V_normalized = min_max_normalize(V_flat)
    print(f"Layer {i} normalized")
    # Flatten to a vector of length 32 * 52 * 1024
    Q_flat_vector = Q_normalized.reshape(32 * 52 * 1024)
    K_flat_vector = K_normalized.reshape(32 * 52 * 1024)
    V_flat_vector = V_normalized.reshape(32 * 52 * 1024)

    # Store the flattened vectors in a tuple
    triplet = (Q_flat_vector, K_flat_vector, V_flat_vector)
    triplets.append(triplet)

# Specify the directory path
output_directory = './extract_BERTClassifierTokens/'

# Create the directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Specify the path to save the pickle file
output_file = os.path.join(output_directory, 'bert_large_triplets.pkl')

# Save the list of flattened triplets to a pickle file
with open(output_file, 'wb') as f:
    pickle.dump(triplets, f)

print(f"Flattened triplets saved to {output_file}")