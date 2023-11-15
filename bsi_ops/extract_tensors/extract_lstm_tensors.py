import torch
import torch.nn as nn
import numpy as np
import pickle
import gensim
import gensim.downloader as api
from sklearn.datasets import fetch_20newsgroups



# Process batch of data
def process_batch(batch_data, model, batch_size):
    sentence_embeddings = torch.from_numpy(
        batch_data).long()  # Convert the NumPy document embeddings to PyTorch tensors
    output = model(sentence_embeddings)  # forward pass
    return output


text = fetch_20newsgroups(subset="all")
print('data fetched')

# Load a pre-trained Word2Vec model
w2v_model = api.load("word2vec-google-news-300")

# Preprocess and tokenize data
documents = text.data
tokenized_documents = [gensim.utils.simple_preprocess(doc) for doc in documents]

# Create embeddings for each word in the documents
word_embeddings = []
for doc in tokenized_documents:
    doc_embeddings = [w2v_model[word] for word in doc if word in w2v_model]
    if doc_embeddings:
        doc_mean = np.mean(doc_embeddings, axis=0)  # Average the word embeddings in each document
        word_embeddings.append(doc_mean)

# Stack the word embeddings to create a matrix of document embeddings
sentence_embeddings = np.vstack(word_embeddings)
print('Sentence embeddings created')


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return lstm_out


# sentence_embeddings = torch.from_numpy(sentence_embeddings).long()
# sentence_embeddings = sentence_embeddings.long()

batch_size = 32
input_size = sentence_embeddings.shape[1]
hidden_size = 128
num_layers = 3

model = LSTMModel(input_size, hidden_size, num_layers)

print(model)

# Forward pass through the model (if no batch processing)
# output = model(sentence_embeddings)
# print('output')
# Batch processing
for i in range(0, len(sentence_embeddings), batch_size):
    batch_data = sentence_embeddings[i:i + batch_size]
    output = process_batch(batch_data, model, batch_size)

# Access the weight tensors at each layer
# model_weights = [param for param in model.parameters()]

print('model weights created')

# print(model_weights)

# # Convert weight tensors to NumPy arrays
# weight_arrays = [param.detach().numpy() for param in model.parameters()]
#
# for name, param in model.named_parameters():
#     print(f"Layer: {name}, Size: {param.size()}")

#Method to return weight pairs
def get_weight_pairs(model):
    weight_pairs = []

    input_weights, hidden_weights = None, None

    for name, param in model.named_parameters():
        if "weight_ih" in name:
            input_weights = param.data.reshape(-1)
        elif "weight_hh" in name:
            hidden_weights = param.data.reshape(-1)
        else:
            input_weights, hidden_weights = None, None

        if input_weights is not None and hidden_weights is not None:
            print(
                f"Layer {len(weight_pairs) + 1}: Input Weights - {input_weights.size()}, Hidden Weights - {hidden_weights.size()}")
            weight_pairs.append((input_weights, hidden_weights))
            input_weights, hidden_weights = None, None

    return weight_pairs

weight_pairs = get_weight_pairs(model)
# Specify the file path for saving the weight pairs
pickle_file_path = 'weight_pairs.pkl'

# Save the weight pairs to the pickle file
with open(pickle_file_path, 'wb') as file:
    pickle.dump(weight_pairs, file)

normalized_weight_pairs = []

for input_weights, hidden_weights in weight_pairs:
        # Normalize input weights to the range [0, 1]
        input_weights_normalized = (input_weights - input_weights.min()) / (input_weights.max() - input_weights.min())

        # Normalize hidden weights to the range [0, 1]
        hidden_weights_normalized = (hidden_weights - hidden_weights.min()) / (hidden_weights.max() - hidden_weights.min())

        normalized_weight_pairs.append((input_weights_normalized, hidden_weights_normalized))
# Specify the file path for saving the normalized weight pairs
pickle_file_path = 'normalized_weight_pairs.pkl'

# Save the normalized weight pairs to the pickle file
with open(pickle_file_path, 'wb') as file:
    pickle.dump(normalized_weight_pairs, file)
