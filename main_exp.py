import pytorch_lightning as pl
from lightning.pytorch.tuner import Tuner
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_forecasting import Baseline, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder, TorchNormalizer, EncoderNormalizer
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import MAE, SMAPE, MultivariateNormalDistributionLoss, NormalDistributionLoss, RMSE
import numpy as np
import os
import json
import argparse

from models.GAS_LSTM import GAS_LSTM, GAS_MAE
from models.RecurrentNetwork_mod import RecurrentNetwork_mod
from utils.data_prep import prepare_dataset, data_generation



#HYPERPARAMETER OPTIMIZATION
def hyperoptimization(data, args, trials_args):
    use_gas_normalization = args.use_gas_normalization
    use_batch_norm = args.use_batch_norm
    use_revin = args.use_revin
    batch_size = args.batch_size

    #create directory for logs and checkpoints
    experiment_logs_dir = f"experiments_logs/{args.data_choice}/{trials_args}"
    if not os.path.exists(experiment_logs_dir):
        os.makedirs(experiment_logs_dir)
    
    # Parameters for optimization
    best_performance = float('inf')
    best_learning_rate = None
    best_norm_strength = None

    # checkpointing for training
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss", filename="hyper_optim", mode="min", save_top_k=1
    )

    def prepare_and_train(norm_strngth=None):
        training, validation, test, gas_params = prepare_dataset(
                data, use_gas_normalization=use_gas_normalization, gas_norm_strength=norm_strength
            )
        train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0, shuffle=False)
        val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0, shuffle=False)

        trial_dir = f"{experiment_logs_dir}/hyperopt_trial"
        if not os.path.exists(trial_dir):
            os.makedirs(trial_dir)
        
        trainer = pl.Trainer(default_root_dir=trial_dir)  

        if use_gas_normalization:
            net = GAS_LSTM.from_dataset(
                training,
                cell_type="LSTM",   
                learning_rate=1e-2,
                hidden_size=30,
                rnn_layers=2,
                loss=GAS_MAE(),
                optimizer="Adam",
                gas_params= gas_params,
                )
        else:
            net = RecurrentNetwork_mod.from_dataset(
                training,
                cell_type="LSTM",   
                learning_rate=1e-2,
                hidden_size=30,
                rnn_layers=2,
                loss=MAE(),
                optimizer="Adam",
                use_batch_norm=use_batch_norm,
                use_revin=use_revin
                )
            
        res = Tuner(trainer).lr_find(
            net,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            min_lr=1e-5,
            max_lr=1e0,
            early_stop_threshold=100,
        )
        learning_rate = res.suggestion()
        print(f"suggested learning rate: {res.suggestion()}")
        net.hparams.learning_rate = learning_rate

        # Full training cycle
        trainer = pl.Trainer(
            max_epochs=3,
            callbacks=[checkpoint_callback],
            enable_checkpointing=True,
            default_root_dir=trial_dir,
        )
        trainer.fit(net, train_dataloader, val_dataloader)

        # Evaluate model performance
        current_performance = trainer.test(net, val_dataloader)[0]['test_loss']
        return current_performance, learning_rate
    
    
    if use_gas_normalization:
        # Define a range for norm_strength
        norm_strength_values = [0.001, 0.01, 0.1, 0.5]
        for norm_strength_value in norm_strength_values:
            norm_strength = [norm_strength_value, norm_strength_value]
            current_performance, learning_rate = prepare_and_train(norm_strength)
            # Update best parameters if current model is better
            if current_performance < best_performance:
                best_performance = current_performance
                best_norm_strength = norm_strength
                best_learning_rate = learning_rate
    else:
        current_performance, learning_rate = prepare_and_train()
        best_learning_rate = learning_rate

    print('Best learning rate: ', best_learning_rate)
    print('Best norm_strength: ', best_norm_strength)

    return best_learning_rate, best_norm_strength
    




############TRAINING AND TESTING
def train_test(data, best_learning_rate, best_norm_strength, args, trials_args):
    # If use GAS normalization, we need to build the datalosader again with the best norm_strength
    use_gas_normalization = args.use_gas_normalization
    use_batch_norm = args.use_batch_norm
    use_revin = args.use_revin
    batch_size = args.batch_size
    num_trials = args.num_trials

    experiment_logs_dir = f"experiments_logs/{args.data_choice}/{trials_args}"
    if not os.path.exists(experiment_logs_dir):
        os.makedirs(experiment_logs_dir)


    if use_gas_normalization:
        training, validation, test, gas_params = prepare_dataset(
            data, use_gas_normalization=use_gas_normalization, args=args, gas_norm_strength=best_norm_strength, 
        )
    else:
        training, validation, test, _ = prepare_dataset(
            data, use_gas_normalization=use_gas_normalization, args=args
        )

    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0, shuffle=False)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0, shuffle=False)
    test_dataloader = test.to_dataloader(train=False, batch_size=batch_size, num_workers=0, shuffle=False)
        

    #We now repeat the training and test process n times to get a better estimate of the performance
    #At the end of each training, we save the model with the best performance on the validation set
    #The best model is then tested on the test set and the performance is saved for later analysis
    
    results = np.zeros(num_trials)

    for trial in range(num_trials):

        pl.seed_everything(trial, workers=True)

        trial_dir = f"{experiment_logs_dir}/trial_{trial}"
        if not os.path.exists(trial_dir):
            os.makedirs(trial_dir)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            filename=f"best_model_{trial}",  # include the trial number in the filename
            mode="min", 
            save_top_k=1
        )

        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, verbose=False, mode="min")

        trainer = pl.Trainer(
        max_epochs=10,
        enable_model_summary=True,
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_checkpointing=True,
        default_root_dir=trial_dir,
        )
        #Init model
        if use_gas_normalization:
            
            net = GAS_LSTM.from_dataset(
                training,
                cell_type="LSTM",
                learning_rate=best_learning_rate,
                log_interval=1,
                log_val_interval=1,
                hidden_size=30,
                rnn_layers=2,
                optimizer="Adam",
                loss=GAS_MAE(), 
                gas_params= gas_params,
            )

        else:
            
            net = RecurrentNetwork_mod.from_dataset(
                training,
                cell_type="LSTM",
                learning_rate=best_learning_rate,
                log_interval=1,
                log_val_interval=1,
                hidden_size=30,
                rnn_layers=2,
                optimizer="Adam",
                loss=MAE(), 
                use_batch_norm=use_batch_norm,
                use_revin=use_revin
                )


        trainer.fit(
            net,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        
        #Load best model
        net = net.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        print(f"Best model on validation set: {trainer.checkpoint_callback.best_model_path}")
      

        #Test model
        result = trainer.test(
            net,
            test_dataloader,
            verbose=True,
        )

        #Save results
        results[trial] = result[0]['test_loss'], result[0]['test_MASE']

    #Print results
    print('Results: ', results)

    return results, best_learning_rate, best_norm_strength


#########MAIN
# run the experiments and save the results

def main(args):    
    pl.seed_everything(42, workers=True)
    trials_args = f"gas_{args.use_gas_normalization}_batchnorm_{args.use_batch_norm}_revin_{args.use_revin}_normalizer_{args.normalizer_choice}_enc_{args.max_encoder_length}_dec_{args.max_prediction_length}_df_{args.degrees_freedom}"


    # Generate data
    data = data_generation(args)

    # Hyperparameter optimization
    best_learning_rate, best_norm_strength = hyperoptimization(data, args, trials_args)

    # Train and test
    results, best_learning_rate, best_norm_strength = train_test(data, best_learning_rate, best_norm_strength, args, trials_args)

    # Save results
    results_dir = f"experiments_results/{args.data_choice}/"
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"results_{trials_args}.json")

    results_data = {
        "best_learning_rate": best_learning_rate,
        "best_norm_strength": best_norm_strength,
        "results": results.tolist()  # convert numpy array to list for JSON serialization
    }

    with open(result_file, "w") as f:
        json.dump(results_data, f, indent=4)

    print(f"Results saved to {result_file}")

    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a series of experiments with different arguments.")
    parser.add_argument('--data_choice', type=str, required=True, help='Choice of dataset: AR, VIX, or ECL')
    parser.add_argument('--use_gas_normalization', type=bool, default=False, help='Whether to use GAS normalization')
    parser.add_argument('--use_batch_norm', type=bool, default=False, help='Whether to use batch normalization')
    parser.add_argument('--use_revin', type=bool, default=False, help='Whether to use RevIN')
    parser.add_argument('--normalizer_choice', type=str, default=None, help='Normalizer choice as a string to be evaluated')
    parser.add_argument('--degrees_freedom', type=int, default=100, help='Degrees of freedom for normalization')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--max_encoder_length', type=int, default=100, help='Maximum encoder length')
    parser.add_argument('--max_prediction_length', type=int, default=50, help='Maximum prediction length')
    parser.add_argument('--num_trials', type=int, default=5, help='Number of trials to run for training and testing')

    args = parser.parse_args()

    # Custom validation logic
    if not args.use_gas_normalization and not args.use_batch_norm and not args.use_revin:
        if args.normalizer_choice is None:
            parser.error("--normalizer_choice is required when use_gas_normalization, use_batch_norm, and use_revin are all False.")
    else:
        args.normalizer_choice = "TorchNormalizer(method='identity', center=False)"
            
    args.normalizer_choice = eval(args.normalizer_choice)  # Convert string to function call

    main(args)