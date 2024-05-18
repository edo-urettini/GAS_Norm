# We modify the LSTM class adding batch normalization and Revin

from pytorch_forecasting.models import RecurrentNetwork
from pytorch_forecasting.models.nn import HiddenState
import torch
from typing import Dict, Tuple
import torch.nn as nn
from normalizers.GAS_norm import Update_function_Student
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
from normalizers.RevIN import RevIN

class RecurrentNetwork_mod(RecurrentNetwork):
    def __init__(self, use_batch_norm, use_revin, **kwargs):
        super().__init__(**kwargs)
        self.encoder_bn = nn.BatchNorm1d(num_features=len(self.reals))
        self.decoder_bn = nn.BatchNorm1d(num_features=len(self.reals))
        self.revin_layer = RevIN(num_features=len(self.reals))
        self.use_batch_norm = use_batch_norm
        self.use_revin = use_revin
        

    def forward(self, x: Dict[str, torch.Tensor], n_samples: int = None) -> Dict[str, torch.Tensor]:
        """
        Forward network
        """
        if self.use_batch_norm:
            #Apply batch normalization considering the input shape is N x T x C
            x["encoder_cont"] = self.encoder_bn(x["encoder_cont"].transpose(1,2)).transpose(1,2)

        if self.use_revin:
            #Apply RevIN
            x["encoder_cont"] = self.revin_layer(x["encoder_cont"], mode='norm')


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

        output = self.decode(
            input_vector,
            decoder_lengths=x["decoder_lengths"],
            target_scale=x["target_scale"],
            hidden_state=hidden_state,
        )

        if self.use_revin:
            #Apply RevIN
            output = self.revin_layer(output, mode='denorm')
        # return relevant part
        return self.to_network_output(prediction=output)
    
    def decode_all(
        self,
        x: torch.Tensor,
        hidden_state: HiddenState,
        lengths: torch.Tensor = None,
    ):
        if self.use_batch_norm:
            #Apply batch norm 
            x = self.decoder_bn(x.transpose(1,2)).transpose(1,2)
        
        decoder_output, hidden_state = self.rnn(x, hidden_state, lengths=lengths, enforce_sorted=False)
        if isinstance(self.hparams.target, str):  # single target
            output = self.output_projector(decoder_output)
        else:
            output = [projector(decoder_output) for projector in self.output_projector]
        return output, hidden_state
