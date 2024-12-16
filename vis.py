from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from sampling import sample_euler, get_timesteps_fm
import torch
import numpy as np
from utils import normalize

def remove_ticks(ax):
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
        left=False,
        labelleft=False
    )


def remove_xticks(ax):
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
        left=True,
        labelleft=True
    )


def visualize_batch(img_vis, title='Семплы из цветного MNIST', nrow=10, ncol=4):
    img_grid = make_grid(img_vis, nrow=nrow)
    fig, ax = plt.subplots(1, figsize=(nrow, ncol))
    remove_ticks(ax)
    ax.set_title(title, fontsize=14)
    ax.imshow(img_grid.permute(1, 2, 0))
    plt.show()





def visualize_mappings(log_imgs, ax, n_pictures=4):
    img_vis = torch.cat((
        normalize(log_imgs['noise'])[:n_pictures],
        normalize(log_imgs['images'])[:n_pictures],
        normalize(log_imgs['x_t'])[:n_pictures],
        normalize(log_imgs['denoised'])[:n_pictures]),
        dim=0
    )
    img_grid = make_grid(img_vis, nrow=n_pictures) * 0.5 + 0.5
    ax.imshow(img_grid.permute(1, 2, 0).detach().cpu())


def visualize_training(model, loss_history, grad_history, log_imgs, sampling_params, n_pictures_sampling=4):
    fig, ax = plt.subplot_mosaic([['loss', 'gradient'],
                                  ['sampling', 'sampling']],
                                 figsize=(11, 9), layout="constrained")

    # loss visualization
    ax['loss'].plot(np.arange(len(loss_history)), loss_history)
    ax['loss'].grid(True)
    ax['loss'].set_title('Лосс на обучении', fontsize=17)
    ax['loss'].set_xlabel('Итерация', fontsize=14)
    ax['loss'].tick_params(labelsize=13)

    ax['gradient'].plot(np.arange(len(grad_history)), grad_history)
    ax['gradient'].grid(True)
    ax['gradient'].set_title('Норма градиента', fontsize=17)
    ax['gradient'].set_xlabel('Итерация', fontsize=14)
    ax['gradient'].tick_params(labelsize=13)

    noise = torch.randn_like(log_imgs['clear_images'][:n_pictures_sampling])
    remove_ticks(ax['sampling'])
    _, trajectory = sample_euler(model, noise, sampling_params, get_timesteps_fm, save_history=True)
    trajectory = torch.cat(trajectory, dim=0) * 0.5 + 0.5
    trajectory = trajectory.reshape(len(trajectory) // n_pictures_sampling, n_pictures_sampling,
                                    *trajectory.shape[-3:]).permute(1, 0, 2, 3, 4).reshape(-1, *trajectory.shape[-3:])
    img_grid = make_grid(trajectory, nrow=len(trajectory) // n_pictures_sampling)
    ax['sampling'].imshow(img_grid.permute(1, 2, 0).detach().cpu())
    ax['sampling'].set_title('Семплы из модели', fontsize=17)
    plt.show()


def visualize_model_samples(model, params, get_timesteps, class_labels=None, title='Семплы из модели', diffusion=True,
                            **model_kwargs):
    if diffusion:
        noise = torch.randn(40, 3, 32, 32).cuda() * params['sigma_max']
    else:
        noise = torch.randn(40, 3, 32, 32).cuda()
    out, trajectory = sample_euler(model, noise, params, get_timesteps, class_labels=class_labels, **model_kwargs)
    out = out * 0.5 + 0.5
    visualize_batch(out.detach().cpu(), title=title)
