import os

import torch
from tqdm import tqdm
from utils import get_grad_norm, get_device
from vis import visualize_training
from IPython.display import clear_output
import wandb


def train(model, opt, train_dataloader, loss_fn, n_epochs, sampling_params, checkpoint_dir, eval_every=100,
          save_every=1000):
    loss_history = []
    grad_history = []
    device = get_device()
    os.makedirs(checkpoint_dir, exist_ok=True)
    it = 0
    with tqdm(total=len(train_dataloader) * n_epochs) as pbar:
        for epoch in range(n_epochs):
            for batch in train_dataloader:
                x = batch[0].to(device)
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

                if it % save_every == 0 or it == len(train_dataloader) * n_epochs - 1:
                    torch.save(model.state_dict(), os.path.join('checkpoints', 'rec_%d.pth' % (it,)))
                pbar.update(1)
                pbar.set_description('Loss: %.4g' % loss.item())
                it += 1
                smoothened_loss = sum(loss_history[max(0, len(loss_history) - 100):]) / min(100, len(loss_history))
                smoothened_grad = sum(grad_history[max(0, len(grad_history) - 100):]) / min(100, len(loss_history))
                wandb.log({
                    "loss": loss_history[-1],
                    "grad_norm": grad_history[-1],
                    "smooth_loss": smoothened_loss,
                    "smooth_grad": smoothened_grad
                })

    return model, loss_history, grad_history
