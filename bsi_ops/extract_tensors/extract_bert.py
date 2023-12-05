from transformers import BertModel, BertTokenizer
import torch
import pickle

# Min-max normalization function
def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor



# Load pre-trained BERT model and tokenizer
model = BertModel.from_pretrained('bert-large-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
# Specify the path to your text file
file_path = './bert_tensors/input.txt'  # Update this path to the actual location of your text file

# Read the contents of the text file into a variable
with open(file_path, 'r') as file:
    input_text = file.read()

# Now, input_text contains the content of the text file
print(input_text)

#input_text = "This is an example sentence. " * 80

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# Forward pass through the model
outputs = model(**inputs)

# Extract the Q, K, and V vectors at each layer
all_q_vectors = []
all_k_vectors = []
all_v_vectors = []

for layer in model.encoder.layer:
    q_vectors = layer.attention.self.query(outputs.last_hidden_state)
    k_vectors = layer.attention.self.key(outputs.last_hidden_state)
    v_vectors = layer.attention.self.value(outputs.last_hidden_state)

    all_q_vectors.append(q_vectors)
    all_k_vectors.append(k_vectors)
    all_v_vectors.append(v_vectors)


# Print the shapes of the extracted Q, K, and V tensors
for i, (q, k, v) in enumerate(zip(all_q_vectors, all_k_vectors, all_v_vectors), 1):
    print(f"Layer {i} - Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")

# Store these tensors in a list
triplets = []

for i in range(len(model.encoder.layer)):
    # Flatten the tensors to [1, 482, 4096]
    # [1, 512, 1024] in large uncased
    Q_flat = all_q_vectors[i].view(1, 512, -1)
    K_flat = all_k_vectors[i].view(1, 512, -1)
    V_flat = all_v_vectors[i].view(1, 512, -1)

    # Normalize the flattened tensors using min-max normalization
    Q_normalized = min_max_normalize(Q_flat)
    K_normalized = min_max_normalize(K_flat)
    V_normalized = min_max_normalize(V_flat)

    # Store the flattened tensors in a tuple
    triplet = (Q_normalized, K_normalized, V_normalized)
    triplets.append(triplet)

# Specify the path to save the pickle file
output_file = './bert_tensors/bert_large_triplets.pkl'

# Save the list of flattened triplets to a pickle file
with open(output_file, 'wb') as f:
    pickle.dump(triplets, f)

print(f"Flattened triplets saved to {output_file}")

