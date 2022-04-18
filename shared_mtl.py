import argparse
import json
import logging
import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from dotenv import load_dotenv
from uuid import uuid4

from torch import nn, Tensor, from_numpy, optim
from torch.nn.functional import nll_loss
from torch.utils.data import DataLoader
from torchviz import make_dot

from data_loader import TurbofanDataset
from arch import HardSharing

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

load_dotenv('../../envs/local.env')


def run_shared_mtl(simulation_type: str,
                   engine_ids: List[int],
                   mode: str,
                   device,
                   model_save_dir="./model_outputs",
                   initial_learning_rate=0.005,
                   num_epochs=10,
                   result_save_dir="./results", is_transform=True):
    # train on the GPU or on the CPU, if a GPU is not available

    train_data_no = {}
    val_data_no = {}

    for engine_id in engine_ids:
        dataset_train = TurbofanDataset(simulation_type=simulation_type, unit_numbers=[engine_id], train=True)
        dataset_val = TurbofanDataset(simulation_type=simulation_type, unit_numbers=[engine_id], train=False)

        data_loader_train = DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=2)
        data_loader_val = DataLoader(dataset_val, batch_size=2, shuffle=True, num_workers=2)

        train_data_no[engine_id] = data_loader_train
        val_data_no[engine_id] = data_loader_val

    optimizer = optim.SGD(model.parameters(), lr=0.005)
    loss_func = torch.nn.BCELoss()
    # view_model(net, "hard_sharing_net", 64, 2, device)
    model.train()
    for epoch in range(1, num_epochs+1):
        for engine_id in engine_ids:
            for batch_idx, (data, target) in enumerate(train_data_no[engine_id]):
                print(data.size(), target.size())
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                print("Output", output.size())
                loss = loss_func(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print(loss.item())


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Input feature size
    input_size = 24

    # Weights in shared hidden layers
    shared_layer_hidden_size = 1

    # No. of shared hidden layers
    shared_layer_n_hidden_layers = 2

    # No. of task-specific layers in each tower
    n_task_specific_layers = 1

    # No. of layers in task-specific layers
    task_specific_hidden_size = None

    # No. of tasks, T; this creates T task-specific stacks
    n_outputs = 10

    model = HardSharing(input_size=input_size,
                        hidden_size=shared_layer_hidden_size,
                        n_hidden=shared_layer_n_hidden_layers,
                        n_outputs=n_outputs,
                        n_task_specific_layers=n_task_specific_layers,
                        task_specific_hidden_size=task_specific_hidden_size,
                        dropout_rate=0.1).to(device)
    # describe_model(model, "./results/turbofan/hard_parameter_sharing_network", 28, 1, device)

    run_shared_mtl("FD001", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "train", device, "./results/model_outputs", 0.005, 10, "./results/")
