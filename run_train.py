import dataset
from model import FMPrecond, FMLoss
from network import EDMPrecond
import torch
from training import train
from utils import get_device

IMG_RES = 32


def run_train(params):
    model = FMPrecond(EDMPrecond(IMG_RES, 3, use_fp16=params["use_fp16"])).to(get_device())
    optimizer = None
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr = params['lr'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr = params['lr'])
    sampling_params = {
        'device': get_device(),
        'sigma_min': 0.02,
        'sigma_max': 80.0,
        'num_steps': 50,
        'vis_steps': 10,
        'rho': 7.0,
        'stochastic': False,
    }
    loss_fn = FMLoss(sampling_params, params["batch_size"], distr='logit', use_OT=params["use_OT"])

    model, loss_log, grad_log = train(model, optimizer, dataset.get_dataloader(params["batch_size"]), loss_fn, params["n_epochs"],
                                      sampling_params, params["checkpoint_dir"], eval_every=params["eval_every"], save_every=params["save_every"])
