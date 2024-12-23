import dataset
from model import FMPrecond, FMLoss
from network import EDMPrecond
import torch
import numpy as np
from training import train
from utils import get_device
import random

IMG_RES = 32


def run_train(params):
    seed = 228
    if 'seed' in params.keys():
        seed = params['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model_params = {
        "channel_mult": [1, 2, 2, 2],
        "model_channels": params["model_channels"],
    }
    model = FMPrecond(EDMPrecond(IMG_RES, 3, use_fp16=params["use_fp16"], **model_params)).to(
        get_device())
    print("number of params", sum([p.numel() for p in model.parameters()]))
    optimizer = None
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    if params['checkpoint'] != 'none':
        model.load_state_dict(torch.load(params['checkpoint']))
    sampling_params = {
        'device': get_device(),
        'sigma_min': 0.02,
        'sigma_max': 80.0,
        'num_steps': 50,
        'vis_steps': 10,
        'rho': 7.0,
        'stochastic': False,
    }
    loss_fn = FMLoss(sampling_params, params["batch_size"], distr=params["t_distr"], use_OT=params["use_OT"])
    dataloader = None
    if params["use_mnist"]:
        dataloader = dataset.get_dataloader_mnist(params["batch_size"])
    else:
        dataloader = dataset.get_dataloader(params["batch_size"])
    model, loss_log, grad_log = train(model, optimizer, dataloader, loss_fn, params["n_epochs"],
                                      sampling_params, params["checkpoint_dir"], eval_every=params["eval_every"],
                                      save_every=params["save_every"], log_fid=params["log_fid"], log_variances_long=params["log_variances_long"])
