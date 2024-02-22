# We modify the LSTM class to make it compatible with GAS normalization

from pytorch_forecasting.models import RecurrentNetwork
from pytorch_forecasting.models.nn import HiddenState
import torch
from typing import Dict, Tuple
import torch.nn as nn
from GAS_norm import Update_function_Student
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, Callable
from pytorch_forecasting.utils import apply_to_list, to_list
from pytorch_forecasting.metrics import (
    MAE,
    MASE,
    SMAPE,
    DistributionLoss,
    MultiHorizonMetric,
    MultiLoss,
    QuantileLoss,
    convert_torchmetric_to_pytorch_forecasting_metric,
)
from pytorch_forecasting.metrics.base_metrics import Metric
import numpy as np



class AR_GAS(nn.Module):
    def __init__(self, k):
        super(AR_GAS, self).__init__()
        self.k = k

    def forward(self, last_mu, last_sigma2, gas_params):
        alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, nu, norm_strength = gas_params
        #transform last_mu and last_sigma2 in numpy
        last_mu = last_mu.detach().numpy()
        last_sigma2 = last_sigma2.detach().numpy()

        
        mu_pred = torch.zeros(self.k)
        sigma2_pred = torch.ones(self.k)
        
        mu_pred[0], sigma2_pred[0] = Update_function_Student(last_mu, last_mu, last_sigma2, alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, nu, norm_strength)
        for i in range(1, self.k):
            mu_pred[i], sigma2_pred[i] = Update_function_Student(mu_pred[i-1], mu_pred[i-1], sigma2_pred[i-1], alpha_mu, alpha_sigma, beta_mu, beta_sigma, omega_mu, omega_sigma, nu, norm_strength)
        return mu_pred, torch.sqrt(sigma2_pred)
    

class GAS_LSTM(RecurrentNetwork):
    def __init__(self, gas_params, **kwargs):
        super().__init__(**kwargs)
        self.gas_params = gas_params
        self.decoder_steps = kwargs.get('decoder_steps', 50)  # Default to 50 if not specified
        self.AR_GAS = AR_GAS(self.decoder_steps)
    


    def output_to_prediction(
        self,
        normalized_prediction_parameters: torch.Tensor,
        target_scale: Union[List[torch.Tensor], torch.Tensor],
        n_samples: int = 1,
        **kwargs,
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor]:
        """
        Convert network output to rescaled and normalized prediction.

        Function is typically not called directly but via :py:meth:`~decode_autoregressive`.

        Args:
            normalized_prediction_parameters (torch.Tensor): network prediction output
            target_scale (Union[List[torch.Tensor], torch.Tensor]): target scale to rescale network output
            n_samples (int, optional): Number of samples to draw independently. Defaults to 1.
            **kwargs: extra arguments for dictionary passed to :py:meth:`~transform_output` method.

        Returns:
            Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor]: tuple of rescaled prediction and
                normalized prediction (e.g. for input into next auto-regressive step)
        """
        single_prediction = to_list(normalized_prediction_parameters)[0].ndim == 2

        if single_prediction:  # add time dimension as it is expected
            normalized_prediction_parameters = apply_to_list(normalized_prediction_parameters, lambda x: x.unsqueeze(1))
        # transform into real space
        prediction_parameters = self.transform_output(
            prediction=normalized_prediction_parameters, target_scale=target_scale, **kwargs
        )
        # todo: handle classification
        # sample value(s) from distribution and  select first sample
        if isinstance(self.loss, DistributionLoss) or (
            isinstance(self.loss, MultiLoss) and isinstance(self.loss[0], DistributionLoss)
        ):
            # todo: handle mixed losses
            if n_samples > 1:                
                prediction_parameters = apply_to_list(
                    prediction_parameters, lambda x: x.reshape(int(x.size(0) / n_samples), n_samples, -1)
                )
                prediction = self.loss.sample(prediction_parameters, 1)
                prediction = apply_to_list(prediction, lambda x: x.reshape(x.size(0) * n_samples, 1, -1))
            else:
                prediction = self.loss.sample(normalized_prediction_parameters, 1)

        else:
            prediction = prediction_parameters
        # normalize prediction prediction
        
        normalized_prediction = normalized_prediction_parameters
        
        if isinstance(normalized_prediction, list):
            input_target = torch.cat(normalized_prediction, dim=-1)
        else:
            input_target = normalized_prediction  # set next input target to normalized prediction

        # remove time dimension
        if single_prediction:
            prediction = apply_to_list(prediction, lambda x: x.squeeze(1))
            input_target = input_target.squeeze(1)

            
        return prediction, input_target
    
    def decode_autoregressive(
        self,
        decode_one: Callable,
        first_target: Union[List[torch.Tensor], torch.Tensor],
        first_hidden_state: Any,
        target_scale: Union[List[torch.Tensor], torch.Tensor],
        n_decoder_steps: int,
        n_samples: int = 1,
        **kwargs,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        # make predictions which are fed into next step
        output = []
        current_target = first_target
        current_hidden_state = first_hidden_state

        normalized_output = [first_target]

        for idx in range(n_decoder_steps):
            # get lagged targets
            current_target, current_hidden_state = decode_one(
                idx, lagged_targets=normalized_output, hidden_state=current_hidden_state, **kwargs
            )

            # get prediction and its normalized version for the next step
            
            prediction, current_target = self.output_to_prediction(
                current_target, target_scale=target_scale[:, idx, :].unsqueeze(1), n_samples=n_samples
            )
            # save normalized output for lagged targets
            normalized_output.append(current_target)
            # set output to unnormalized samples, append each target as n_batch_samples x n_random_samples

            output.append(prediction)

        if isinstance(self.hparams.target, str):
            output = torch.stack(output, dim=1)
        else:
            # for multi-targets
            output = [torch.stack([out[idx] for out in output], dim=1) for idx in range(len(self.target_positions))]
        return output
    
        

    def forward(self, x: Dict[str, torch.Tensor], n_samples: int = None) -> Dict[str, torch.Tensor]:


        # Get the mean and the variance from x
        #the tensor is of shape batch_size x time_series_length x num_features. The first two features are the mean and the variance
        mean = x['encoder_cont'].squeeze(2)[:,:,0]
        variance = x['encoder_cont'].squeeze(2)[:,:,1]

        #We normalize the third feature of the encoder_cont tensor and the decoder_cont with the mean and the variance (first and second feature)
        x['encoder_cont'][:,:,2] = (x['encoder_cont'][:,:,2] - x['encoder_cont'][:,:,0]) / torch.sqrt(x['encoder_cont'][:,:,1])
        


        #What we need is the last value of the mean and the variance in time series
        last_mean = mean[:,-1]
        last_variance = variance[:,-1]
        

        #We predict the next k time steps means and variances with the GAS model for each batch
        #The final shape of the normalization parameters should be batch_size*k*(mean, std)
        pred_norm_params = torch.zeros((x['decoder_cont'].shape[0], self.decoder_steps, 2))
        for i in range(x['decoder_cont'].shape[0]):
            pred_norm_params[i,:,0], pred_norm_params[i,:,1] = self.AR_GAS(last_mean[i], last_variance[i], self.gas_params)

        #We use the predicted mean and variance to normalize the third feature of the decoder_cont tensor
        x['decoder_cont'][:,:,2] = (x['decoder_cont'][:,:,2] - pred_norm_params[:,:,0]) / pred_norm_params[:,:,1]
        

        #We do not want the model to use the mean and the variance as features both in the encoder and the decoder
        #We set them to zero
        x['encoder_cont'][:,:,0] = 0
        x['encoder_cont'][:,:,1] = 0
        x['decoder_cont'][:,:,0] = 0
        x['decoder_cont'][:,:,1] = 0

       # call parent class's forward method to get the deep model predictions
    
        """
        Forward network
        """
        hidden_state = self.encode(x)
        # decode
        input_vector = self.construct_input_vector(
            x["decoder_cat"],
            x["decoder_cont"],
            one_off_target=x["encoder_cont"][
                torch.arange(x["encoder_cont"].size(0), device=x["encoder_cont"].device),
                x["encoder_lengths"] - 1,
                self.target_positions.unsqueeze(-1),
            ].T.contiguous(),
        )

        if self.training:
            assert n_samples is None, "cannot sample from decoder when training"

        

        # Instead of using the static target scale batch_size * 2 we use the predicted time varying params with shape batch_size * time_steps * 2
        output = self.decode(
            input_vector,
            decoder_lengths=x["decoder_lengths"],
            target_scale=pred_norm_params,
            hidden_state=hidden_state,
        )
        
        return self.to_network_output(prediction=output, target_scale=pred_norm_params)
    

#Modify the loss to rescale the parameters in the correct way
from sklearn.base import BaseEstimator

class GAS_MAE(MultiHorizonMetric):
    """
    Mean average absolute error.

    Defined as ``(y_pred - target).abs()``
    """

    def loss(self, y_pred, target):
        loss = (self.to_prediction(y_pred) - target).abs()
        return loss


    def rescale_parameters(
        self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator
    ) -> torch.Tensor:
        """
        Rescale normalized parameters into the scale required for the output.

        Args:
            parameters (torch.Tensor): normalized parameters (indexed by last dimension)
            target_scale (torch.Tensor): scale of parameters (n_batch_samples x (center, scale))
            encoder (BaseEstimator): original encoder that normalized the target in the first place

        Returns:
            torch.Tensor: parameters in real/not normalized space
        """
        # Ensure that target_scale is broadcastable to parameters
        if target_scale.ndim == 2:
            target_scale = target_scale.unsqueeze(1)

        # Extract mean and variance from target_scale
        mean, variance = target_scale[..., 0], target_scale[..., 1]

        # Rescale parameters
        rescaled_parameters = parameters * variance.unsqueeze(-1) + mean.unsqueeze(-1)

        return rescaled_parameters