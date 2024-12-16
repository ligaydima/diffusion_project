import torch


class EDMLoss:
    def __init__(self, mean=0.0, std=1.0, sigma_data=0.5):
        self.mean = mean
        self.std = std
        self.sigma_data = sigma_data

    def __call__(self, net, images, cond=None, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.std + self.mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(images) * sigma
        D_yn = net(images + n, sigma, labels, cond=cond)
        loss = (weight * (D_yn - images) ** 2).mean()
        log_imgs = {
            'cond': cond.detach(),
            'clear_images': images.detach(),
            'noisy_images': ((images + n)).detach(),
            'denoised_images': D_yn.detach()
        }
        return loss, log_imgs


class FMLoss:
    def __init__(self, diffusion, sampling_params, batch_dim, distr='rand'):
        self.diffusion = diffusion  # предобученная диффузионная модель
        self.sampling_params = sampling_params  # параметры семплирования из предобученной модели
        self.batch_dim = batch_dim  # размерность одного батча на обучении
        self.distr = distr  # распределение для семплирования моментов времени

    def sample_t(self, n_samples, device='cuda'):
        if self.distr == 'rand':
            return torch.rand(n_samples, device=device)

        if self.distr == 'logit':
            n = torch.randn(n_samples, device=device)
            return torch.sigmoid(n)

    def __call__(self, net, images, use_OT=False):
        noise = torch.randn_like(images)
        x_0 = noise
        x_1 = images
        if use_OT:
            pass
        t = self.sample_t(self.batch_dim[0])[:, None, None, None]

        x_t = t * x_0 + (1 - t) * x_1
        denoiser_pred = net(x_t, t.flatten())
        loss = ((denoiser_pred - images) ** 2).mean()
        log_imgs = {
            'noise': noise.cpu().detach(),
            'images': images.cpu().detach(),
            'x_t': x_t.cpu().detach(),
            'denoised': denoiser_pred.cpu().detach()
        }

        return loss, log_imgs
