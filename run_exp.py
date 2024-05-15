import subprocess
import os
import sys


def run_experiment(script_name, args):
    """Run a python script with given arguments."""
    command = [sys.executable, script_name] + args  # Use sys.executable for the correct Python interpreter
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([os.path.abspath(os.path.dirname(__file__)), env.get('PYTHONPATH', '')])
    print(f"Running {' '.join(command)} with PYTHONPATH={env['PYTHONPATH']}")
    subprocess.run(command, env=env)

def main():
    # Define the list of experiments with corresponding arguments
    experiments = [
        (".\models\RecurrentNetwork_SAN.py", []),
        # Add more experiments or scripts as needed
    ]

    # Run experiments
    for script, args in experiments:
        run_experiment(script, args)

if __name__ == "__main__":
    main()
