from transformers import BertModel
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import pickle
import os


def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor-min_val)/(max_val-min_val)
    return normalized_tensor

def extract_bert_tensor_weights(output_dir):
    model = BertModel.from_pretrained('bert-base-uncased')

    triplets = []
    for layer_idx, layer in enumerate(model.encoder.layer):
        print(f"Processing layer: {layer_idx}")

        attention = model.encoder.layer[layer_idx].attention.self
        #extracting query,key, and value weights
        query_weights = attention.query.weight.detach().cpu().numpy()
        key_weights = attention.key.weight.detach().cpu().numpy()
        value_weights = attention.value.weight.detach().cpu().numpy()

        # print(f"query weights shape: {query_weights.shape}")

        batch_size = 53
        sequence_length = 256
        hidden_size = query_weights.shape[0]

        q_result = np.zeros((batch_size, sequence_length, hidden_size), dtype=np.float32)
        k_result = np.zeros((batch_size, sequence_length, hidden_size), dtype=np.float32)
        v_result = np.zeros((batch_size, sequence_length, hidden_size), dtype=np.float32)

        for i in range(batch_size):
            for j in range(sequence_length):
                q_result[i, j, :] = query_weights[:, j % hidden_size]
                k_result[i, j, :] = key_weights[:, j % hidden_size]
                v_result[i, j, :] = value_weights[:, j % hidden_size]
        # print(f"q_result shape: {q_result.shape}")

        q_normalized = min_max_normalize(q_result)
        k_normalized = min_max_normalize(k_result)
        v_normalized = min_max_normalize(v_result)

        q_flat_vector = q_normalized.reshape(-1)
        # print(f"Checking flat vector size of q: {type(q_flat_vector)}")
        k_flat_vector = k_normalized.reshape(-1)
        # print(f"Checking flat vector size of k: {type(k_flat_vector)}")
        v_flat_vector = v_normalized.reshape(-1)
        # print(f"Checking flat vector size of v: {type(v_flat_vector)}")

        triplets.append((q_flat_vector, k_flat_vector, v_flat_vector))
    
    output_file = os.path.join(output_dir,'qkv_weights-initial.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(triplets, f)
    print(f"Created triplets and created a pickle file")

if __name__=="__main__":
    output_dir = 'output_39882/bertVectors/'
    num_hidden_layers = 12
    extract_bert_tensor_weights(output_dir=output_dir)