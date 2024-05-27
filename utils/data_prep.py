import pandas as pd
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import TorchNormalizer
import matplotlib.pyplot as plt
from normalizers.GAS_norm import SD_Normalization_Student
import numpy as np
import yfinance as yf

def data_generation(args):

    data_name = args.data_choice

    if data_name == 'AR':
        #generate ar_data
        ar_data = generate_ar_data(seasonality=20.0, timesteps=5000, n_series=1, seed=42)
        ar_data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(ar_data.time_idx, "D")
        ar_data = ar_data.astype(dict(series=str))
        data = ar_data

    elif data_name == 'VIX':        
        #Download the VIX index from yahoo finance
        

        vix = yf.download("^VIX")

        #Compute the returns of the VIX index
        vix['Adj Close'] = vix['Adj Close'].diff()
        vix = vix.dropna().reset_index()


        #Construct Pandas dataframe with the VIX index, the Date, a time index starting from 0 and a series column to identify the series
        vix['time_idx'] = vix.index
        vix['series'] = 0
        #Change column name
        vix = vix.rename(columns={'Adj Close':'value', 'Date':'date'})
        #Drop columns
        vix = vix.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        data = vix


    elif data_name == 'ECL':
        #ECL data

        #Load the data
        ecl_small = pd.read_csv("./datasets/ecl_data_small.csv")

        #transform the date column to datetime
        ecl_small["date"] = pd.to_datetime(ecl_small["date"])
        data = ecl_small

    else:
        raise ValueError(f"Unknown data_choice: {data_name}")

    return data



def prepare_dataset(data, use_gas_normalization, args, norm_strength=None):
    max_prediction_length = args.max_prediction_length
    max_encoder_length = args.max_encoder_length
    normalizer_choice = args.normalizer_choice
    degrees_freedom = args.degrees_freedom

    # Define cutoffs for training, validation, and test sets (10% validation, 20% test)
    total_length = len(data)
    validation_cutoff = int(total_length * 0.7)
    test_cutoff = int(total_length * 0.8)

    if use_gas_normalization:
        # Perform GAS normalization
        mu_list, sigma2_list, y_norm, alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, nu = SD_Normalization_Student(
            data['value'], data['value'][:validation_cutoff], args=args, mode='predict',
            norm_strength=norm_strength, degrees_freedom=degrees_freedom, 
        )
        data['mu'] = mu_list
        data['sigma2'] = sigma2_list
        target_normalizer = TorchNormalizer(method='identity', center=False)

        
        #Add norm_strength to gas_params
        gas_params = alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, nu, norm_strength


    else:
        # Use other normalization method
        gas_params = []
        target_normalizer = normalizer_choice


    # Create datasets and dataloaders
    training = TimeSeriesDataSet(
        data[data.time_idx <= validation_cutoff],
        time_idx="time_idx",
        target="value",
        group_ids=["series"],
        time_varying_unknown_reals=["value"],
        time_varying_known_reals=['mu', 'sigma2'] if use_gas_normalization else [],
        scalers={'mu': None, 'sigma2': None} if use_gas_normalization else {},
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        target_normalizer=target_normalizer
    )
    
    validation = TimeSeriesDataSet.from_dataset(training, data[data.time_idx <= test_cutoff], min_prediction_idx=validation_cutoff + 1)
    test = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=test_cutoff + 1)

    
    return training, validation, test, gas_params