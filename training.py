import os

import torch
from tqdm import tqdm
import numpy as np

from fid import calc_fid
from model import FMPrecond, FMLoss
from utils import get_grad_norm, get_device
from vis import visualize_training, send_samples_to_wandb
from IPython.display import clear_output
import wandb
NUM_RUNS = 70
def estimate_variances(model, opt, loss_fn, x):
    losses = torch.zeros(NUM_RUNS)
    res_grads = None
    for i in tqdm(range(NUM_RUNS)):
        opt.zero_grad()
        loss, log_imgs = loss_fn(model, x)
        loss.backward()
        grads = [
            param.grad.detach().flatten()
            for param in model.parameters()
            if param.grad is not None
        ]
        grad_vector = torch.cat(grads)
        if res_grads is None:
            res_grads = torch.zeros((NUM_RUNS, grad_vector.shape[0]))
        res_grads[i] = grad_vector
        losses[i] = loss
    mean_grad_vector = res_grads.mean(dim=0).reshape(1, -1)
    ss = 0
    for i in range(res_grads.shape[0]):
        res_grads[i] -= mean_grad_vector
        ss += (res_grads[i]**2).sum().item()
    ss /= res_grads.shape[0]
    return losses.var().item(), res_grads.var(dim=1).sum().item(), ss

def train(model: FMPrecond, opt, train_dataloader, loss_fn: FMLoss, n_epochs: int, sampling_params, checkpoint_dir: str, eval_every=100,
          save_every=1000, log_fid=False, log_variances_long=False):
    loss_history = []
    grad_history = []
    device = get_device()
    os.makedirs(checkpoint_dir, exist_ok=True)
    it = 0
    loss_variance_history = []
    with tqdm(total=len(train_dataloader) * n_epochs) as pbar:
        for epoch in range(n_epochs):
            for batch in train_dataloader:
                x = batch[0].to(device)
                if it % 500 == 0 and log_variances_long:
                    loss_var, grad_var, grad_var2 = estimate_variances(model, opt, loss_fn, x)
                    wandb.log({
                        "grad_var_long": grad_var,
                        "loss_var_long": loss_var,
                        "grad_var_long2": grad_var2
                    }, step=it)
                if it % 2000 == 0 and log_fid:
                    model.eval()
                    cur_fid = calc_fid(model, "cifar10-32x32.npz", "generated_images", sampling_params, batch_size=128, num_samples=10000)
                    wandb.log({"fid": cur_fid}, step=it)
                    model.train()
                opt.zero_grad()
                loss, log_imgs = loss_fn(model, x)
                loss.backward()
                loss_history.append(loss.item())
                opt.step()
                grad_history.append(get_grad_norm(model).detach().cpu())
                if it % eval_every == 0:
                    model.eval()
                    clear_output(wait=True)
                    visualize_training(model, loss_history, grad_history, log_imgs, sampling_params)
                    model.train()

                if it % save_every == 0 or it == len(train_dataloader) * n_epochs - 1:
                    torch.save(model.state_dict(), os.path.join('checkpoints', 'rec_%d.pth' % (it,)))

                if it % 2000 == 0 or it == len(train_dataloader) * n_epochs - 1:
                    model.eval()
                    send_samples_to_wandb(model, log_imgs, sampling_params, it)
                    model.train()

                pbar.update(1)
                pbar.set_description('Loss: %.4g' % loss.item())
                it += 1
                smoothened_loss = sum(loss_history[max(0, len(loss_history) - 100):]) / min(100, len(loss_history))
                smoothened_grad = sum(grad_history[max(0, len(grad_history) - 100):]) / min(100, len(loss_history))

                grad_variance = np.var(grad_history[max(0, len(grad_history) - 100):])
                loss_variance_history.append(log_imgs["loss_variance"])
                smoothened_loss_variance = sum(loss_variance_history[max(0, len(grad_history) - 100):]) / min(100,
                                                                                                              len(loss_history))
                wandb.log({
                    "loss": loss_history[-1],
                    "grad_norm": grad_history[-1],
                    "smooth_loss": smoothened_loss,
                    "smooth_grad": smoothened_grad,
                    "loss_variance": log_imgs["loss_variance"],
                    "grad_variance": grad_variance,
                    "smooth_loss_variance": smoothened_loss_variance
                })

    return model, loss_history, grad_history
