import torch
import torch.nn as nn
import numpy as np
import gensim
import gensim.downloader as api
from sklearn.datasets import fetch_20newsgroups
import pickle


def process_batch(batch_data, model):
    sentence_embeddings = torch.from_numpy(batch_data).long()
    output, layer_output = model(sentence_embeddings)  # Forward pass and capture output from the desired layer
    return output, layer_output


text = fetch_20newsgroups(subset="all")
print('data fetched')

w2v_model = api.load("word2vec-google-news-300")

documents = text.data
num_documents = len(documents)
print("Number of documents:", num_documents)
tokenized_documents = [gensim.utils.simple_preprocess(doc) for doc in documents]

word_embeddings = []
for doc in tokenized_documents:
    doc_embeddings = [w2v_model[word] for word in doc if word in w2v_model]
    if doc_embeddings:
        doc_mean = np.mean(doc_embeddings, axis=0)  # Average the word embeddings in each document
        word_embeddings.append(doc_mean)

len(word_embeddings)

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
        return lstm_out, embedded


batch_size = 32
input_size = sentence_embeddings.shape[1]
hidden_size = 128
num_layers = 20

model = LSTMModel(input_size, hidden_size, num_layers)

print(model)

output_layer_index = 7  # Adjust this to the desired layer index
output_layers = []  # List to store output from the desired layer
#
# for i in range(0, len(sentence_embeddings), batch_size):
#     batch_data = sentence_embeddings[i:i + batch_size]
#     output, layer_output = process_batch(batch_data, model, batch_size)
#     if output_layer_index == i // batch_size:
#         output_layers.append(layer_output)
# Train on a single batch
batch_data = sentence_embeddings[:batch_size]
output, layer_output = process_batch(batch_data, model)
print(layer_output.shape)
model_weights = [param for param in model.parameters()] # weight for each layer

print('model weights created')

weight_arrays = [param.detach().numpy() for param in model.parameters()]

for name, param in model.named_parameters():
    print(f"Layer: {name}, Size: {param.size()}")


# Filter the parameters based on their names
weight_ih_tensors = [param.detach().numpy() for name, param in model.named_parameters() if 'lstm.weight_ih' in name]
weight_hh_tensors = [param.detach().numpy() for name, param in model.named_parameters() if 'lstm.weight_hh' in name]

# Convert the lists to numpy arrays
weight_ih_array = np.array(weight_ih_tensors)
weight_hh_array = np.array(weight_hh_tensors)

# Print the shapes of the arrays
print("Weight_ih Array Shape:", weight_ih_array.shape)
print("Weight_hh Array Shape:", weight_hh_array.shape)