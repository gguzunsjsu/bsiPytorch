import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
#This file must convert the q,k,v vectors in each of the six layers into triplets
#The output must have 10 files for 10 epochs, each with the q,k,v triplets

def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

def extract(output_dir, num_hidden_layers, analysis_dir):
        logging.info(f"Extraction started for {os.path.join(output_dir)}")
        for epoch in range(10):
            #Start the triplet list
            triplets = []            
            for layer_idx in range(num_hidden_layers):
                if epoch > 0:
                    epoch_str = f"epoch_{epoch:.1f}_layer_{layer_idx + 1}_results.pkl"
                else:
                    epoch_str = f"epoch_{int(epoch)}_layer_{layer_idx + 1}_results.pkl"

                pickle_file_path = os.path.join(output_dir, epoch_str)
                # print(f"Printing pickle file path: {pickle_file_path}") included to check the file path
                logging.info(f"Analysis started for {pickle_file_path}")
                if os.path.exists(pickle_file_path):
                    with open(pickle_file_path, 'rb') as pickle_file:
                        results_dict = pickle.load(pickle_file)
                    logging.info(f"Analysis for {pickle_file_path}")
                    # Plot distribution of Q_result for each layer
                    #We have 53 vectors of shape 256*768
                    #The batach size is 53 and each row has 256 tokens, 768 is the hidden size of
                    #the BERT Model
                    logging.info(f"Shape of the vectors: {results_dict['q_result'].shape}" )
                    q_result = results_dict['q_result'].reshape(53, -1)
                    v_result = results_dict['w_result'].reshape(53, -1)
                    k_result = results_dict['k_result'].reshape(53, -1)
                    # Log dimensions of Q_result.flatten() and Q_result
                    # Normalize the flattened tensors using min-max normalization
                    Q_normalized = min_max_normalize(q_result)
                    K_normalized = min_max_normalize(k_result)
                    V_normalized = min_max_normalize(v_result)
                    logging.info(f"Layer {layer_idx} normalized")
                    # Flatten to a vector of length 32 * 52 * 1024
                    Q_flat_vector = Q_normalized.reshape(-1)
                    K_flat_vector = K_normalized.reshape(-1)
                    V_flat_vector = V_normalized.reshape(-1)
                    logging.info(f"Shape of the flatterned vectors: {Q_flat_vector.shape} and {V_flat_vector.shape}" )
                    # Store the flattened vectors in a tuple
                    triplet = (Q_flat_vector, K_flat_vector, V_flat_vector)
                    triplets.append(triplet)
            #Save the triplet for each epoch
            # Specify the path to save the pickle file
            # print(f"triplets: \n {triplets}")
            output_file = os.path.join(analysis_dir, f'bertVectors_{epoch}.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump(triplets, f)
            logging.info(f"Flattened triplets saved to {output_file}")
                    

if __name__ == "__main__":
        # Specify the output directory and number of hidden layers
        output_dir = "./output_39882/"  
        analysis_dir = os.path.join(output_dir, 'bertVectors/')
        os.makedirs(analysis_dir, exist_ok=True)
        num_hidden_layers = 6 

        # Set up logging
        log_file_path = os.path.join(analysis_dir, 'analysis_log.txt')
        logging.basicConfig(filename=log_file_path, level=logging.INFO)
        logging.info('Analysis started at %s.', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        # Analyze the results from the first two epochs
        extract(output_dir, num_hidden_layers,analysis_dir)