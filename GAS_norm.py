from scipy.optimize import minimize
from scipy.special import gamma
import numpy as np

#Score Driven Normalization Gaussian Process
#The regularization choices are Full (natural gradient descent) and Root (normalized gradient descent)

#The function takes the single time series to normalize, the training data to fit the static parameters and outputs
#the normalized time series and the lists of mean and variances
#The mode (predict or update) let us choose if we want to normalize the data with the previous prediction or with the current updated parameters
#Norm strength is a chosen parameter that let us choose the strength of the normalization for sigma and mu

def SD_Normalization_Gaussian(y, y_train, regularization, mode='predict', norm_strength=[1,1]):
    alpha_mu, alpha_sigma, mu_0, sigma2_0 = Optimized_params_Gaussian(y_train, regularization)
    alpha_sigma = alpha_sigma * norm_strength[1]
    alpha_mu = alpha_mu * norm_strength[0]

    T = len(y)
    mu_list, sigma2_list = np.zeros(T), np.ones(T)
    y_normalized = np.zeros(T)

    for t in range(0, T):
        if t == 0:
            #At the first step, we update starting from the inizialization parameters
            mu_list[t], sigma2_list[t] = Update_function_Gaussian(regularization, y[t], mu_0, sigma2_0, alpha_mu, alpha_sigma)
        else:
            mu_list[t], sigma2_list[t] = Update_function_Gaussian(regularization, y[t], mu_list[t-1], sigma2_list[t-1], alpha_mu, alpha_sigma)
        
        if mode == 'predict':
            if t == 0:
                y_normalized[t] = (y[t]-mu_0)/np.sqrt(sigma2_0)
            else:
                y_normalized[t] = (y[t]-mu_list[t-1])/np.sqrt(sigma2_list[t-1])
        elif mode == 'update':
            y_normalized[t] = (y[t]-mu_list[t])/np.sqrt(sigma2_list[t])
        else:
            print('Error: mode must be predict or update')
    
    return mu_list, sigma2_list, y_normalized

#Define the Update function for the Gaussian parameters. It updates the mean and the variance at each new observation
def Update_function_Gaussian(regularization, y_t, mu_t, sigma2_t, alpha_mu, alpha_sigma):
    if regularization == 'Full':
        mu_updated= mu_t + alpha_mu*(y_t-mu_t) 
        sigma2_updated= sigma2_t + alpha_sigma*((y_t-mu_t)**2  - sigma2_t)

    elif regularization == 'Root':
        mu_updated =  alpha_mu * (y_t - mu_t) / np.sqrt(sigma2_t) + mu_t
        sigma2_updated =  alpha_sigma * (-np.sqrt(2)/2 + np.sqrt(2)*(y_t-mu_t)**2 / (2*sigma2_t)) + sigma2_t
        
    else:
        print('Error: regularization must be Full or Root')
    return mu_updated, sigma2_updated


#Define the likelihood function for the Gaussian case
def neg_log_likelihood_Gaussian(params, y, regularization):
    epsilon = 1e-8
    alpha_mu, alpha_sigma, mu_0, sigma2_0 = params

    T = len(y)
    mu_list, sigma2_list = np.zeros(T), np.zeros(T)
    log_likelihood_list = np.zeros(T)
    y = np.append(y, y[T-1])

    for t in range(0, T):
        if t == 0:
            #At the first step, we update starting from the inizialization parameters
            mu_list[t], sigma2_list[t] = Update_function_Gaussian(regularization, y[t], mu_0, sigma2_0, alpha_mu, alpha_sigma)
            
        else:
            mu_list[t], sigma2_list[t] = Update_function_Gaussian(regularization, y[t], mu_list[t-1], sigma2_list[t-1], alpha_mu, alpha_sigma)
        
        log_likelihood_list[t] = -0.5 * np.log(2 * np.pi * sigma2_list[t]) - 0.5 * (y[t+1] - mu_list[t]) ** 2 / sigma2_list[t] 
    
        
    neg_log_lokelihood = -np.sum(log_likelihood_list)

    return neg_log_lokelihood/T


#Define the optimization function that optimize the likelihood function

def Optimized_params_Gaussian(y, regularization, initial_guesses= np.array([0.001, 0.001, 0, 1])):

    #The bounds are defined to avoid negative intial variance and learning rates outside the interval (0,1)
    bounds = ((0, 1), (0, 1), (None, None), (0.00001, 1))
    optimal = minimize(lambda params: neg_log_likelihood_Gaussian(params, y, regularization), x0=initial_guesses, bounds=bounds)
    
    alpha_mu, alpha_sigma, mu_0, sigma2_0 = optimal.x
    print('Optimal parameters:  alpha_mu = {},  alpha_sigma = {}, mu_0 = {}, sigma2_0 = {}'.format(alpha_mu, alpha_sigma, mu_0, sigma2_0))

    return alpha_mu, alpha_sigma, mu_0, sigma2_0
  
#######################################################################################################à


#Score Driven Normalization Student-t
#Note that the sigma2 needs a tranformation to be the correct variance of the Student distribution
def SD_Normalization_Student(y, y_train, mode='predict', norm_strength=[0.5, 0.5], degrees_freedom=100):

    nu = degrees_freedom
    alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, mu_0, sigma2_0 = Optimized_params_Student(y_train, norm_strength, nu)
    
    T = len(y)
    mu_list, sigma2_list = np.zeros(T), np.ones(T)
    y_normalized = np.zeros(T)

    for t in range(0, T):
        if t == 0:
            #At the first step, we update starting from the inizialization parameters
            mu_list[t], sigma2_list[t] = Update_function_Student(y[t], mu_0, sigma2_0, alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, nu, norm_strength)
        else:
            mu_list[t], sigma2_list[t] = Update_function_Student(y[t], mu_list[t-1], sigma2_list[t-1], alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, nu, norm_strength)
        
        if mode == 'predict':
            if t == 0:
                y_normalized[t] = (y[t]-mu_0)/np.sqrt(sigma2_0 * nu / (nu - 2))
            else:
                y_normalized[t] = (y[t]-mu_list[t-1])/np.sqrt(sigma2_list[t-1] * nu / (nu - 2))
        elif mode == 'update':
            y_normalized[t] = (y[t]-mu_list[t])/np.sqrt(sigma2_list[t] * nu / (nu - 2))
        else:
            print('Error: mode must be predict or update')
    
    sigma2_list = sigma2_list * nu / (nu - 2)
    return mu_list, sigma2_list, y_normalized, alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, nu


#Define the Update function for the Student parameters. It updates the mean and the variance at each new observation
def Update_function_Student(y_t, mu_t, sigma2_t, alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, nu, norm_strength):
    
    mu_updated= mu_t + ((norm_strength[0]) / (1 - norm_strength[0])) * alpha_mu * (y_t - mu_t) / (1 + (y_t - mu_t) ** 2 / (nu * sigma2_t))
    mu_updated = omega_mu + beta_mu * mu_updated
    sigma2_updated= sigma2_t + ((norm_strength[1]) / (1 - norm_strength[1])) * alpha_sigma*((nu + 1) * (y_t - mu_t)**2 / (nu +  (y_t - mu_t)**2 / sigma2_t)  - sigma2_t)
    sigma2_updated = omega_sigma + beta_sigma * sigma2_updated

    return mu_updated, sigma2_updated


#Define the likelihood function for the Student case
def neg_log_likelihood_Student(params, y, norm_strength, nu):
    alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, mu_0, sigma2_0 = params

    T = len(y)
    mu_list, sigma2_list = np.zeros(T), np.zeros(T)
    log_likelihood_list = np.zeros(T)
    y = np.append(y, y[T-1])



    for t in range(0, T):
        if t == 0:
            #At the first step, we update starting from the inizialization parameters
            mu_list[t], sigma2_list[t] = Update_function_Student(y[t], mu_0, sigma2_0, alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, nu, norm_strength)
            penalty_term_mu = 0.5 * (1 - norm_strength[0]) * (mu_list[t] - mu_0)**2
            penalty_term_sigma2 = 0.5 * (1 - norm_strength[1]) * (sigma2_list[t] - sigma2_0)**2
        else:
            mu_list[t], sigma2_list[t] = Update_function_Student(y[t], mu_list[t-1], sigma2_list[t-1], alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, nu, norm_strength)
            penalty_term_mu = 0.5 * (1 - norm_strength[0]) * (mu_list[t] - mu_list[t-1])**2
            penalty_term_sigma2 = 0.5 * (1 - norm_strength[1]) * (sigma2_list[t] - sigma2_list[t-1])**2

        log_likelihood_list[t] = np.log(gamma((nu + 1) / 2)) - np.log(gamma(nu / 2)) - 0.5 * np.log(np.pi * nu) - 0.5 * np.log(sigma2_list[t]) - ((nu + 1) / 2) * np.log(1 + (y[t+1] - mu_list[t]) ** 2 / (nu * sigma2_list[t])) 
        log_likelihood_list[t] = (norm_strength[0] + norm_strength[1]) * log_likelihood_list[t] - penalty_term_mu - penalty_term_sigma2
        
    neg_log_lokelihood = -np.sum(log_likelihood_list)/T

    return neg_log_lokelihood

#Define the optimization function that optimize the likelihood function

def Optimized_params_Student(y, norm_strength, nu, initial_guesses= np.array([0.001, 0.001, 0.5, 0.5, 0, 1, 0, 1])):

    #The bounds are defined to avoid negative intial variance and learning rates outside the interval (0,1)
    bounds = ((0, 1), (0, 1), (0, 1), (0, 1), (None, None), (0.000001, None), (None, None), (0.000001, None))
    optimal = minimize(lambda params: neg_log_likelihood_Student(params, y, norm_strength, nu), x0=initial_guesses, bounds=bounds, options={'maxiter': 1000}, method='Powell')
    
    alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, mu_0, sigma2_0 = optimal.x
    print('Optimal parameters:  alpha_mu = {},  alpha_sigma = {}, mu_0 = {}, sigma2_0 = {}, beta_mu = {}, beta_sigma = {}, omega_mu = {}, omega_sigma = {}'.format(alpha_mu, alpha_sigma, mu_0, sigma2_0, beta_mu, beta_sigma, omega_mu, omega_sigma))
    return alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, mu_0, sigma2_0