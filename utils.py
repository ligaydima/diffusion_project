import torch
def normalize(x):
    return x / x.abs().max(dim=0)[0][None, ...]
DEVICE = None
def get_device():
    global DEVICE
    if DEVICE:
        return DEVICE
    # Check for CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA acceleration.")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS acceleration.")
    else:
        device = torch.device("cpu")
        print("No acceleration available, using CPU.")
    DEVICE = device
    return device

def get_grad_norm(model):
    grads = [
        param.grad.detach().flatten()
        for param in model.parameters()
        if param.grad is not None
    ]
    return torch.cat(grads).norm()