import numpy as np
import torch.nn as nn
import torch
import ot
class FMPrecond(nn.Module):
    def __init__(self, model, error_eps=1e-4):
        super().__init__()
        self.model = model
        self.error_eps = error_eps

    def forward(self, x_t, t, class_labels=None):
        sigma = (1 - t) / t# sigma(t)
        x_sigma = x_t / t[:, None, None, None]# phi(x_t, t)
        return self.model(x_sigma, sigma, class_labels)

    def to_velocity(self, x_t, t, denoiser_pred):
        return denoiser_pred / (1 - t[:, None, None, None]) - x_t / (1 - t[:, None, None, None])


    def velocity(self, x_t, t, class_labels=None):
        return self.to_velocity(x_t, t, self.forward(x_t, t, class_labels=class_labels))

class FMLoss:
    def __init__(self, sampling_params, batch_dim, distr='rand', use_OT=False):
        self.sampling_params = sampling_params  # параметры семплирования из предобученной модели
        self.batch_dim = batch_dim  # размерность одного батча на обучении
        self.distr = distr  # распределение для семплирования моментов времени
        self.use_OT = use_OT

    def sample_t(self, n_samples, device='cpu'):
        if self.distr == 'rand':
            return torch.rand(n_samples, device=device)

        if self.distr == 'logit':
            n = torch.randn(n_samples, device=device)
            return torch.sigmoid(n)

    def __call__(self, net, images):
        noise = torch.randn_like(images)
        x_0 = noise
        x_1 = images
        if self.use_OT:
            x_0_tmp = x_0.unsqueeze(1)
            x_1_tmp = x_1.unsqueeze(0)
            cost = ((x_0_tmp - x_1_tmp) ** 2).sum(dim = (2,3,4))
            B = torch.ones(images.shape[0]).to(images.device) / images.shape[0]
            distr = torch.ones(images.shape[0]).to(images.device) / images.shape[0]
            plan = ot.emd(distr, B, cost)
            flattened_plan = plan.flatten()
            if torch.any(torch.isnan(flattened_plan)) or not torch.all(torch.isfinite(flattened_plan)):
                print("BUGG")
                assert False
            probs = flattened_plan / flattened_plan.sum()
            sample_ids = torch.argsort(probs, descending=True)[:images.shape[0]]
            row_index, col_index = np.divmod(sample_ids, plan.shape[1])
            # sampled_id = torch.multinomial(flattened_plan, num_samples = images.shape[0], replacement = True)
            # row_index = sampled_id // plan.shape[1]
            # col_index = sampled_id % plan.shape[1]
            # print(row_index, col_index)
            x_0 = x_0[row_index]
            x_1 = x_1[col_index]
        t = self.sample_t(images.shape[0], device=images.device)[:, None, None, None]

        x_t = t * x_1 + (1 - t) * x_0
        denoiser_pred = net(x_t, t.flatten())
        loss = ((denoiser_pred - x_1) ** 2).mean()
        log_imgs = {
            'noise': noise.cpu().detach(),
            'images': images.cpu().detach(),
            'x_t': x_t.cpu().detach(),
            'denoised': denoiser_pred.cpu().detach()
        }

        return loss, log_imgs