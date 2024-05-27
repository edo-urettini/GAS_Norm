import os
import json
import numpy as np

def compute_statistics(data):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    return mean, std

def process_json_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as file:
                data = json.load(file)
                
                test_losses = [result['test_loss'] for result in data['results']]
                test_mases = [result['test_MASE'] for result in data['results']]
                
                mean_loss, std_loss = compute_statistics(test_losses)
                mean_mase, std_mase = compute_statistics(test_mases)
                
                print(f"File: {filename}")
                print(f"Mean Test Loss: {mean_loss:.4f}, Std Test Loss: {std_loss:.4f}")
                print(f"Mean Test MASE: {mean_mase:.4f}, Std Test MASE: {std_mase:.4f}\n")

if __name__ == "__main__":
    folder_path = 'experiments_results\ECL'  # Replace with the path to your folder containing JSON files
    process_json_files(folder_path)
