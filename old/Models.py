import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from GAS_norm import Update_function_Student

#Define a neural network class that takes the past 200 time steps as input and outputs the next 50 time steps. Use pytorch.
class Net(nn.Module):
    def __init__(self): #Define the layers
        super(Net, self).__init__()
        self.relu=nn.ReLU()   
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 50)
    
    def encoder(self, x): 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x
        

    def forward(self, x): #Define the forward pass        
        x=self.encoder(x)
        x = self.fc3(x)
        return x
    


class GAS_Net(nn.Module):
    def __init__(self):
        super(GAS_Net, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 50)
        self.sigma_layer = nn.Linear(200, 50)
        self.mu_layer = nn.Linear(200, 50)
    
    def encoder(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
    
        
    def forward(self, x, mu_vector, sigma2_vector, phase):
        
        #In phase 1 we use only the mu_layer by setting the encoded x to zero
        if phase == 0:
            mu_encoded = self.mu_layer(mu_vector)
            output = torch.add(mu_encoded, 0)
        
        #In phase 2 we use both the mu_layer and the encoding of x but we set the encoded sigma to one
        elif phase == 1:
            encoded = self.encoder(x)
            mu_encoded = self.mu_layer(mu_vector)
            output = torch.add(encoded, mu_encoded)

        #In phase 3 we use everything

        elif phase == 2:
            encoded = self.encoder(x)
            #Multiply the encoded x with an new encoding of the standard deviation
            sigma_vector= torch.sqrt(sigma2_vector)        
            sigma_encoded = self.sigma_layer(sigma_vector)
            encoded = torch.mul(encoded, sigma_encoded)
            #Sum the encoded x with an new encoding of the mean
            print(sigma_encoded)
            mu_encoded = self.mu_layer(mu_vector)
            output = torch.add(encoded, mu_encoded)
            
        return output



#REVIN net

from RevIN import RevIN

class Revin_Net(nn.Module):
    def __init__(self): #Define the layers
        super(Revin_Net, self).__init__()
        self.relu=nn.ReLU()     
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 50)
        self.revin_layer = RevIN(1)
    
    def encoder(self, x):    
        x = self.fc1(x) 
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x
        

    def forward(self, x): #Define the forward pass
        x = x.unsqueeze(0).unsqueeze(2)
        x = self.revin_layer(x, 'norm')
        x = x.squeeze(2).squeeze(0)
        x = self.encoder(x)
        x = self.fc3(x)
        x = x.unsqueeze(0).unsqueeze(2)
        x = self.revin_layer(x, 'denorm')
        x = x.squeeze(2).squeeze(0)
        return x
    

#######################################################################################################
#GAS net where the prediction of the mean and the variance are not done with linear layer but autoregressively with GAS

#First of all we define our main deep model
class Non_AR_net(nn.Module):
    def __init__(self, k):
        super(Non_AR_net, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, k)


    def encoder(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
    
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded
    

#Then we define the GAS predictor of the mean and the variance that predicts the next k time steps autoregressively
#We can choose if using or not the prediction of the deep model
class AR_GAS(nn.Module):
    def __init__(self, k, use_deep_preds=True):
        super(AR_GAS, self).__init__()
        self.k = k
        self.use_deep_preds = use_deep_preds

    def forward(self, deep_preds, last_mu, last_sigma, alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, nu, norm_strength):
        mu_pred = torch.zeros(self.k)
        sigma_pred = torch.zeros(self.k)
        if self.use_deep_preds:
            mu_pred[0], sigma_pred[0] = Update_function_Student(deep_preds[0], last_mu, last_sigma, alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, nu, norm_strength)
            for i in range(1, self.k):
                mu_pred[i], sigma_pred[i] = Update_function_Student(deep_preds[i], mu_pred[i-1], sigma_pred[i-1], alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, nu, norm_strength)
        else:
            mu_pred[0], sigma_pred[0] = Update_function_Student(last_mu, last_mu, last_sigma, alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, nu, norm_strength)
            for i in range(1, self.k):
                mu_pred[i], sigma_pred[i] = Update_function_Student(mu_pred[i-1], mu_pred[i-1], sigma_pred[i-1], alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, nu, norm_strength)
        return mu_pred, torch.sqrt(sigma_pred)
    

#Finally we define the GAS net that sum the predictions of the deep model and the GAS model

class AR_GAS_Net(nn.Module):
    def __init__(self, k, use_deep_preds=True):
        super(AR_GAS_Net, self).__init__()
        self.k = k
        self.non_AR_net = Non_AR_net(k)
        self.AR_GAS = AR_GAS(k, use_deep_preds)

    def forward(self, x, last_mu, last_sigma, alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, nu, norm_strength):
        deep_preds = self.non_AR_net(x)
        mu_pred, sigma_pred = self.AR_GAS(deep_preds.detach(), last_mu, last_sigma, alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, nu, norm_strength)
        output = torch.mul(deep_preds, sigma_pred)
        output = torch.add(output, mu_pred)
        return output