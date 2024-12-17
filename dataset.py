import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

CIFAR10_ROOT = './data'

def get_dataloader(batch_size, num_workers=1):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
    ])
    train_dataset = datasets.CIFAR10(root=CIFAR10_ROOT, train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader
