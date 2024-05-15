#We modify the RNN to be used with the SAN approach

import os
os.chdir('..')
print(os.getcwd())
from pytorch_forecasting.models import RecurrentNetwork
from pytorch_forecasting.models.nn import HiddenState
import torch
from typing import Dict, Tuple
import torch.nn as nn
from normalizers import GAS_norm

from typing import Any, Dict, List, Literal, Optional, Tuple, Union, Callable
from models.SAN_master.models.Statistics_prediction import Statistics_prediction


class RecurrentNetwork_SAN(RecurrentNetwork):
    def __init__(self, *args, **kwargs):
        self.configs = kwargs.pop('configs')
        super().__init__(*args, **kwargs)
        self.station_pretrain_epoch = 5 if configs.station_type == 'adaptive' else 0
        self.statistics_pred = Statistics_prediction(self.configs).to(self.device)
        self.station_criterion = self.loss
        self.automatic_optimization = False
        

    def station_loss(self, y, statistics_pred):
        bs, seq_len, dim = y.shape
        y = y.view(bs, -1, self.hparams.period_len, dim)
        mean = torch.mean(y, dim=2)
        std = torch.std(y, dim=2)
        station_true = torch.cat([mean, std], dim=-1)
        loss = self.station_criterion(statistics_pred, station_true)
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        pred_len= batch_y.shape[1]
        

        if self.trainer.current_epoch + 1 < self.station_pretrain_epoch:

            #Freeze all the parameters except the station parameters
            for param in self.parameters():
                param.requires_grad = False
            for param in self.statistics_pred.parameters():
                param.requires_grad = True

            opt = self.optimizers()
            opt.zero_grad()

            batch_x, statistics_pred = self.statistics_pred.normalize(batch_x)
            f_dim = 0
            batch_y = batch_y[:, -pred_len:, f_dim:].to(self.device)
            loss = self.station_loss(batch_y, statistics_pred)
            
            self.manual_backward(loss)
            opt.step()
            self.log('train_station_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
        
        else:
            #Unfreeze all the parameters and freeze the station parameters
            for param in self.parameters():
                param.requires_grad = True
            for param in self.statistics_pred.parameters():
                param.requires_grad = False

            opt = self.optimizers()
            opt.zero_grad()

            batch_x, statistics_pred = self.statistics_pred.normalize(batch_x)
            f_dim = 0
            batch_y = batch_y[:, -pred_len:, f_dim:].to(self.device)
            log, out = self.step(batch_x, batch_y, batch_idx)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
        
        self.training_step_outputs.append(log)
        return log


#Minimal example to test the model
from pytorch_forecasting import TimeSeriesDataSet
import pandas as pd
from pytorch_forecasting.data.examples import generate_ar_data
import lightning as pl
from collections import namedtuple
import argparse


#generate ar_data
ar_data = generate_ar_data(seasonality=20.0, timesteps=5000, n_series=1, seed=42)
ar_data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(ar_data.time_idx, "D")
ar_data = ar_data.astype(dict(series=str))

#prepare data
max_prediction_length = 10
max_encoder_length = 50
training_cutoff = ar_data["time_idx"].max() - max_prediction_length
context_length = max_encoder_length



#Parse the arguments for configs
parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int, default=max_encoder_length)
parser.add_argument('--pred_len', type=int, default=max_prediction_length)
parser.add_argument('--period_len', type=int, default=12)
parser.add_argument('--features', type=str, default='S')
parser.add_argument('--station_type', type=str, default='adaptive')

configs = parser.parse_args()

print(configs)

training = TimeSeriesDataSet(
    ar_data[ar_data.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="value",
    group_ids=["series"],
    min_encoder_length=context_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=context_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=["value"],
)

validation = TimeSeriesDataSet.from_dataset(training, ar_data, predict=True, stop_randomization=True)

#Create dataloaders
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

model = RecurrentNetwork_SAN.from_dataset(training, configs = configs, hidden_size=10, cell_type='GRU', log_interval=1, log_val_interval=1, learning_rate=1e-3, weight_decay=1e-2)
trainer = pl.Trainer(max_epochs=1, gpus=0, gradient_clip_val=0.1)
trainer.fit(model, train_dataloader, val_dataloader)





