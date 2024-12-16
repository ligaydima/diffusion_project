class FMPrecond(nn.Module):
    def __init__(self, model, error_eps=1e-4):
        super().__init__()
        self.model = model # VE-денойзер
        self.error_eps = error_eps

    # выражаем FM денойзер через VE денойзер и возвращаем предсказание чистой картинки
    def forward(self, x_t, t, class_labels=None):
        sigma = (1 - t) / t# sigma(t)
        x_sigma = x_t / t[:, None, None, None]# phi(x_t, t)
        return self.model(x_sigma, sigma, class_labels)

    # выражаем векторное поле f*_t(x_t) через предсказание оптимального денойзера D*_t(x_t) (pred)
    def to_velocity(self, x_t, t, denoiser_pred):
        return denoiser_pred / (1 - t[:, None, None, None]) - x_t / (1 - t[:, None, None, None])# a(x_t, t) * denoiser_pred + b(x_t, t)

    # комбинируем первое и второе
    def velocity(self, x_t, t, class_labels=None):
        return self.to_velocity(x_t, t, self.forward(x_t, t, class_labels=class_labels))# a(x_t, t) * D*_t(x_t) + b(x_t, t)
