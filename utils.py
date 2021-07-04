from debug import debug

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import conv2d

import torchvision
import torchvision.transforms.functional as TF

import numpy as np
import skimage.filters
import pickle

import matplotlib.pyplot as plt


# def get_blur_kernel(sigma=1):
#     x = np.linspace(-1, 1, 5) / sigma
#     y = np.exp(- x * x)
#     y /= np.sum(y)
#     return torch.as_tensor(y.reshape(1, 1, 1, -1) * y.reshape(1, 1, -1, 1)).float()


def get_params(model):
    return model.parameters() if model._named_parameters is None else list(zip(*model._named_parameters))[1]


def blur(model, x, y, min_pixel_value, max_pixel_value, beta, steps, device='cuda'):
    if beta == 0:
        return x
    blurred = []
    n_ch = x.shape[1]
    for img in x.permute(0, 2, 3, 1).squeeze():
        blurred.append(skimage.filters.gaussian(img, sigma=(beta, beta),
                       truncate=3.5, multichannel=True, mode='reflect'))
    out = torch.as_tensor(np.stack(blurred)).to(device)
    if n_ch == 1:
        return out.unsqueeze(1)
    return out
    # return conv2d(x, get_blur_kernel(beta), stride=1, padding=2).clamp(min_pixel_value, max_pixel_value)
    # return adjust_sharpness(x, beta)


def noise_corrupt(model, x, y, min_pixel_value, max_pixel_value, beta, steps, device='cuda'):
    return (x + beta * torch.randn_like(x)).clamp(min_pixel_value, max_pixel_value)


def lbfgs(model, x, y, min_pixel_value, max_pixel_value, beta, steps, device='cuda'):
    loss = nn.CrossEntropyLoss()
    y = y.to(device)
    x_orig = x.to(device)
    x = x.clone().to(device)
    x.requires_grad = True
    model.to(device)
    lr = 0.02

    # batch_size = 1000

    for n in range(steps):
        # for i in range(math.ceil(len(x) / batch_size)):
        batch = x
        # batch = x[i * batch_size:(i + 1) * batch_size]

        out = model(batch)
        (loss(out, y) - beta * (x - x_orig).norm()).backward()

        x.data.add_(lr, x.grad.sign())
        x.data.clamp_(min_pixel_value, max_pixel_value)

        x.grad.detach_()
        x.grad.zero_()

    return x.detach()


def fgsm(model, x, y, min_pixel_value, max_pixel_value, beta, steps, device='cuda'):
    if beta == 0:
        return x
    loss = nn.CrossEntropyLoss()
    x = torch.as_tensor(x).clone().to(device)
    x.requires_grad = True
    model.to(device)

    # batch_size = 1000

    for n in range(steps):
        # for i in range(math.ceil(len(x) / batch_size)):
        batch = x
        # batch = x[i * batch_size:(i + 1) * batch_size]

        out = model(batch)
        loss(out, y).backward()

        x.data.add_(beta, x.grad.sign())
        x.data.clamp_(min_pixel_value, max_pixel_value)

        x.grad.detach_()
        x.grad.zero_()

    return x.detach()


ltypes = [
    "base",
    "base_fc",
    "linear",
    "base_larger",
    "base_larger2",
    "base_larger3",
    "base_larger3_global_pooling",
    "base_larger4_global_pooling",
    "base_larger4",
]


@torch.no_grad()
def store_samples():
    batches = []
    for it in range(64):
        x, y = g(it)
        batches.append((x.cpu().numpy(), y.cpu().numpy()))
    batches = list(reversed(batches))
    with open('runs/test_samples.pkl', 'wb') as f:
        pickle.dump(batches, f)


def show_sample(X, permute=False, sample_random=False):
    if sample_random:
        X = X[np.random.randint(0, len(X), size=10)]
    X = torch.as_tensor(X)
    if permute:
        X = torch.from_numpy(X).permute(0, 2, 3, 1)

    img_grid = torchvision.utils.make_grid(X, nrow=10)
    plt.figure(figsize=(8, 14))
    plt.axis('off')
    plt.imshow(img_grid.permute(1, 2, 0).numpy())
    plt.show()


# @torch.no_grad()
# def evaluate_set(model, x, y, name):
#     batch_size = 1000
#     pred = []

#     model.eval()
#     for i in range(math.ceil(len(x) / batch_size)):
#         batch_x = x[i * batch_size:(i + 1) * batch_size]
#         # batch_x.cuda()
#         out = model(batch_x)
#         pred.append(out)

#     # accuracy = np.sum(np.argmax(pred, axis=1) == y) / len(y)
#     pred = torch.cat(pred, dim=0)
#     acc = (pred.max(-1).indices == y).float().mean().item()
#     return acc


class DatasetTransform():
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = (x - 0.1307) / 0.3081
        return x, y

    def __len__(self):
        return len(self.dataset)


import pickle
# from train_cgtn import get_dataset as cgtn_get_dataset


import torchvision.datasets as datasets


def cgtn_get_dataset(dataset, data_path, seed, device):
    torch.manual_seed(seed)
    if dataset == "MNIST":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
        trainset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
        trainset, validationset = torch.utils.data.random_split(trainset, [50000, 10000])
        testset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
        img_shape = (1, 28, 28)
    elif dataset == "CIFAR10":
        img_mean = [0.49139968, 0.48215827, 0.44653124]
        img_std = [0.24703233, 0.24348505, 0.26158768]
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(img_mean, img_std),
        ])
        transform_train = transform
        img_mean = torch.as_tensor(img_mean).to(device)[..., None, None]
        img_std = torch.as_tensor(img_std).to(device)[..., None, None]
        trainset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
        validationset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)

        # Split train and validationset
        lengths = [45000, 5000]
        indices = torch.randperm(sum(lengths)).tolist()
        trainset = torch.utils.data.dataset.Subset(trainset, indices[:lengths[0]])
        validationset = torch.utils.data.dataset.Subset(validationset, indices[-lengths[1]:])

        testset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
        img_shape = (3, 32, 32)
    torch.manual_seed(seed)

    return img_shape, trainset, validationset, testset


# def get_datasets(samples='runs/mnist/samples_2000.pkl'):
#     (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
#     mnist_train_x = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
#     mnist_train_y = np.argmax(y_train, axis=1)
#     mnist_test_x = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
#     mnist_test_y = np.argmax(y_test, axis=1)
#     mnist_train = (mnist_train_x, mnist_train_y)
#     mnist_test = (mnist_test_x, mnist_test_y)

#     with open(samples, 'rb') as f:
#         gxy = pickle.load(f)
#     gxy = list(reversed(gxy))
#     # gtn_train = DatasetTransform(gxy)
#     gtn_train = gxy

#     img_shape, trainset, validationset, (testset_x, testset_y) = cgtn_get_dataset(
#         'MNIST', 'data', seed=1, device='cpu', with_augmentation=False)
#     validation_x, validation_y = zip(*validationset)
#     gtn_valid = torch.stack(validation_x).numpy(), torch.as_tensor(validation_y).numpy()
#     gtn_test = testset_x.numpy(), testset_y.numpy()

#     gx_train, gy_train = zip(*gxy)
#     generator_train = np.vstack(gx_train), np.vstack(gy_train)

#     valsets = {
#         'generator_train': generator_train,
#         'gtn_valid': gtn_valid,
#         'gtn_test': gtn_test,
#         'mnist_train': mnist_train,
#         'mnist_test': mnist_test,
#     }

#     return gtn_train, valsets


def prepare_batch(x, y=None, device='cuda'):
    x = torch.as_tensor(x).to(device)
    if y is not None:
        y = torch.as_tensor(y).to(device)
        if len(y.shape) > 1:
            y = y.argmax(dim=1)
        return x, y
    return x


@torch.no_grad()
def evaluate_set(model, valset, device='cuda'):
    model.eval()
    model.to(device)
    correct = []
    for x, y in valset:
        x, y = prepare_batch(x, y, device='cuda')
        out = model(x)
        correct.append((out.argmax(1) == y))
    correct = torch.cat(correct)
    acc = correct.cpu().float().mean().item()
    return acc


# @debug
def fit_generator(model, generator, optimizer, loss, epochs=1, valsets=None, silent=False):
    model.train()

    accuracies = []
    for epoch in range(epochs):
        for step, (x, y) in enumerate(generator):
            optimizer.zero_grad()
            x, y = prepare_batch(x, y)
            out = model(x)
            loss(out, y).backward()
            optimizer.step()

            acc = (out.argmax(dim=1) == y).float().mean().item()

            val_acc = {}
            if not silent:
                if step % 90 == 0 and valsets is not None:
                    for name, valset in valsets.items():
                        acc = evaluate_set(model, valset)
                        val_acc[name] = acc

            acc_step = {'train': acc, **val_acc}

            if not silent:
                if step == 0:
                    print(('{:>15}' * (len(acc_step.keys()) + 1)).format('', *acc_step.keys()))
                print(('{:>15}' + '{:>15.3f}' * len(acc_step.keys())).format(step, *acc_step.values()))
            accuracies.append(acc_step)
    return accuracies


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        self.bn_1 = nn.BatchNorm2d(4)
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
        self.bn_2 = nn.BatchNorm2d(10)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
        self.fc_2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = self.conv_1(x)
        # x = self.bn_1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv_2(x)
        # x = self.bn_2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x
