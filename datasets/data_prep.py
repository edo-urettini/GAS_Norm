import pandas as pd
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import TorchNormalizer
import matplotlib.pyplot as plt
from normalizers.GAS_norm import SD_Normalization_Student
import numpy as np

def data_generation(args):

    data_name = args.data_choice

    if data_name == 'AR':
        #generate ar_data
        ar_data = generate_ar_data(seasonality=20.0, timesteps=5000, n_series=1, seed=42)
        ar_data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(ar_data.time_idx, "D")
        ar_data = ar_data.astype(dict(series=str))
        data = ar_data

    if data_name == 'VIX':        
        #Download the VIX index from yahoo finance
        import yfinance as yf

        vix = yf.download("^VIX")

        #Compute the returns of the VIX index
        vix['Adj Close'] = vix['Adj Close'].diff()
        vix = vix.dropna()


        #Construct Pandas dataframe with the VIX index, the Date, a time index starting from 0 and a series column to identify the series

        vix = vix.reset_index()
        vix['time_idx'] = vix.index
        vix['series'] = 0
        #Change column name
        vix = vix.rename(columns={'Adj Close':'value', 'Date':'date'})
        #Drop columns
        vix = vix.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        data = vix


    if data_name == 'ECL':
        #ECL data

        #load the data
        ecl_data = pd.read_csv(".\datasets\ECL.txt", sep = ";")

        #Transform the first column into a datetime object
        ecl_data["date"] = pd.to_datetime(ecl_data.iloc[:,0])

        #Drop first column and Select the columns Date and MT_001
        ecl_data = ecl_data.drop(ecl_data.columns[0], axis = 1)[['date', 'MT_001']]

        #Change the commas to dots, transfrom to numeric and fill NaN with 0 in the column MT_001
        ecl_data['value'] = ecl_data['MT_001'].str.replace(',','.').astype(float).fillna(0)

        #Drop all the data before the first non-zero value of the column MT_001_new
        ecl_data = ecl_data.drop(ecl_data.index[0: ecl_data['value'].ne(0).idxmax()])[0:10000]

        ecl_data.reset_index(drop = True, inplace = True)
        ecl_data['time_idx'] = ecl_data.index
        ecl_data['series'] = 0
        ecl_data.drop(['MT_001'], axis = 1, inplace = True)
        data = ecl_data

    return data



def prepare_dataset(data, use_gas_normalization, args, norm_strength=None, degrees_freedom=None):
    max_prediction_length = args.max_prediction_length
    max_encoder_length = args.max_encoder_length
    normalizer_choice = args.normalizer_choice

    # Define cutoffs for training, validation, and test sets (10% validation, 20% test)
    total_length = len(data)
    validation_cutoff = int(total_length * 0.7)
    test_cutoff = int(total_length * 0.8)

    if use_gas_normalization:
        # Perform GAS normalization
        mu_list, sigma2_list, y_norm, alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, nu = SD_Normalization_Student(
            data['value'], data['value'][:validation_cutoff], mode='predict',
            norm_strength=norm_strength, degrees_freedom=degrees_freedom
        )
        data['mu'] = mu_list
        data['sigma2'] = sigma2_list
        target_normalizer = TorchNormalizer(method='identity', center=False)

        
        #Add norm_strength to gas_params
        gas_params = alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, nu, norm_strength

        # Plot the normalized data
        plt.plot(data['time_idx'][1:], y_norm[1:])
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Normalized data')
        plt.show()

        # Plot the original data with the predicted mean and 95% variability interval
        plt.plot(data['time_idx'], data['value'], label='Original data')
        plt.plot(data['time_idx'], mu_list, label='Predicted mean')
        plt.fill_between(data['time_idx'], mu_list-1.96*np.sqrt(sigma2_list), mu_list+1.96*np.sqrt(sigma2_list), alpha=0.5, label='95% variability interval', color='orange')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Original data with predicted mean and 95% variability interval')
        plt.legend()
        plt.show()

    else:
        # Use other normalization method
        gas_params = []
        target_normalizer = normalizer_choice

        # Plot original data
        plt.plot(data['time_idx'], data['value'], label='Original data')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Original data')
        plt.legend()
        plt.show()

    # Create datasets and dataloaders
    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= validation_cutoff],
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
    
    validation = TimeSeriesDataSet.from_dataset(training, data[lambda x: x.time_idx <= test_cutoff], min_prediction_idx=validation_cutoff + 1)
    test = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=test_cutoff + 1)

    
    return training, validation, test, gas_params