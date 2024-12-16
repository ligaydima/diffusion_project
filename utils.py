import torch

def get_device():
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

    return device
