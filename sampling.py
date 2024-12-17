import torch

from utils import normalize, get_device


def get_timesteps_diff(params):
    num_steps = params['num_steps']
    sigma_min, sigma_max = params['sigma_min'], params['sigma_max']
    rho = params['rho']
    step_indices = torch.arange(num_steps, device=params['device'])
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
    return t_steps

def get_timesteps_fm(params):
    sigmas = get_timesteps_diff(params)
    return 1 / (sigmas + 1)
def sample_euler(model, noise, params, get_timesteps,  save_history=False, **model_kwargs):
    num_steps = params['num_steps']
    t_steps = get_timesteps(params)
    x = noise
    if save_history:
        vis_steps = params['vis_steps']
        x_history = [normalize(noise)]

    with torch.no_grad():
        for i in range(len(t_steps) - 1):
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]
            t_net = t_steps[i] * torch.ones(x.shape[0], device=params['device'])
            x = x + model.velocity(x, t_net) * (t_next - t_cur)
            if save_history:
                x_history.append(normalize(x).view(-1, 3, *x.shape[2:]))

    if save_history:
        x_history = [x_history[0]] + x_history[::-(num_steps // (vis_steps - 2))][::-1] + [x_history[-1]]
        return x, x_history

    return x, []

