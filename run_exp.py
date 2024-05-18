import subprocess
import os 



def run_experiment(command):
    full_command = ["python", "main_exp.py"] + command.split()
    subprocess.run(full_command)

if __name__ == "__main__":
    "arguments: --data_choice --use_gas_normalization  --use_batch_norm --use_revin --normalizer_choice --degrees_freedom --batch_size --max_encoder_length --max_prediction_length --num_trials"
    
    
    experiment_commands = [
        "--data_choice AR --normalizer_choice EncoderNormalizer() --num_trials 1",
        # Add more experiment commands as needed
    ]

    for command in experiment_commands:
        run_experiment(command)
