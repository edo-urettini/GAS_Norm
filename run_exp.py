import subprocess
import os 



def run_experiment(command):
    full_command = ["python", "main_exp.py"] + command.split()
    subprocess.run(full_command)

if __name__ == "__main__":
    "arguments: --data_choice --use_gas_normalization  --use_batch_norm --use_revin --normalizer_choice --degrees_freedom --batch_size --max_encoder_length --max_prediction_length --num_trials --gas_init_zero_one"
    ###WARNING: to boolean arguments, any string will be considered True
    
    experiment_commands = [    
       "--data_choice ECL --use_gas_normalization True  --num_trials 1 --degrees_freedom 100 --max_encoder_length 200 --gas_init_zero_one True --batch_size 512",
       

        # Add more experiment commands as needed
    ]

    for command in experiment_commands:
        run_experiment(command)
