import os
import torch
import torchvision

from debug import debug

from models import Generator
from train_cgtn import AutoML, CGTN

dataset = 'mnist'
img_shape = (1, 28, 28)

files = [os.path.join('runs', dataset, f)
         for f in os.listdir(os.path.join('runs', dataset))
         if 'samples' in f]

state_file = f'runs/detach/checkpoint_2000.pkl'
# state = torch.load(f'runs/{dataset}/checkpoint_2000.pkl')
samples_file = f'runs/detach/samples_2000.pkl'
state = torch.load(state_file)

model_state = {k: v for k, v in state['model'].items() if not 'optimizers' in k}
optimizer_state = {k: v for k, v in state['model'].items() if 'optimizers' in k}

noise_size = 128
g = CGTN(
    generator=Generator(noise_size + 10, img_shape),
    num_inner_iterations=64,
    generator_batch_size=128,
    noise_size=noise_size,
    evenly_distributed_labels=True,
    meta_learn_labels=False,
)

automl = AutoML(g, optimizers=None)
automl.load_state_dict(model_state)
# automl.to('cuda')


from utils import fit_generator, get_datasets, show_sample
from utils import TestNet
import torch.nn as nn

cgtn_train, valsets = get_datasets(samples=samples_file)
loader = cgtn_train
import numpy as np
torch.manual_seed(1)
np.random.seed(1)

# model = TestNet()
from models import Classifier
model = Classifier(input_shape=(1, 28, 28), batch_norm_momentum=0, use_global_pooling=False)
debug(model.convs[0].weight)

model.train()
model.to('cuda')

optimizer = torch.optim.SGD(model.parameters(), lr=10, momentum=10)

from models import AbstractClassifier


class GTNSGD():
    def __init__(self, params, log_lr, log_momentum):
        self.params = list(params)

        self.lr = torch.exp(log_lr)
        self.momentum = torch.exp(log_momentum)

        self.momentum_buffer = {}
        self.it = 0

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        lr = self.lr[self.it]
        momentum = self.momentum[self.it]

        # debug(lr)
        # debug(momentum)
        # debug(self.params[0].flatten()[:10] + self.params[-1].flatten()[-10:])
        # debug(self.params[0].flatten()[:10] + self.params[-1].flatten()[-10:])

        print(self.it)
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            # if state is None:
            #     state = p.grad
            # state = state * momentum + p.grad
            # out = p - lr * state

            if p not in self.momentum_buffer:
                buf = self.momentum_buffer[p] = p.grad.clone().detach()
            else:
                buf = self.momentum_buffer[p]
                buf.mul_(momentum).add_(p.grad.data)
            p.data.add_(-lr, buf)
            # if self.it < 2:
            #     if i == 0 or i == 9:
            #         print('p', p.flatten()[:10])
            #         print('grad', p.grad.flatten()[:10])
            #         print('buf', buf.flatten()[:10])
            #         print('out', p.flatten()[:10])
        self.it += 1


optimizer = GTNSGD(model.parameters(),
                   log_lr=optimizer_state['optimizers.0.log_lr'],
                   log_momentum=optimizer_state['optimizers.0.log_momentum'])

# import inner_optimizers
# optimizer = inner_optimizers.SGD(0.02, 0.5, 64)
criterion = nn.CrossEntropyLoss()

# from torch.utils.data.dataloader import DataLoader
# loader = DataLoader(list(zip(*valsets['mnist_train'])), batch_size=128)

valsets = {k: v for k, v in valsets.items() if 'train' not in k}

_ = fit_generator(model, loader, optimizer, criterion, epochs=1, valsets=valsets)
debug(model.parameters())

print(model)

# debug(model.modules())


# # %%

# class XXBatchNorm1dMeanOnly(nn.BatchNorm1d):
#     def forward(self, x):
#         self.running_var.fill_(1)
#         x = super().forward(x)
#         return x


# class XXBatchNorm2dMeanOnly(nn.BatchNorm2d):
#     def forward(self, x):
#         self.running_var.fill_(1)
#         x = super().forward(x)
#         return x


# def patch_batchnorm(batchnorm):
#     print(batchnorm.running_mean.shape)
#     # new = XXBatchNorm2dMeanOnly()
#     # new.running_mean = batchnorm.running_mean
#     # return new


# def get_bn_layers(module):
#     # ignore_types = ['activation', 'loss', 'container', 'pooling']
#     # all_layers = []
#     for name, layer in module.named_children():
#         if len(list(layer.children())) == 0:
#             if 'BatchNormMeanOnly' in layer.__class__.__name__:
#                 print(f'patching {module.__class__.__name__}.{name} {layer}')
#                 setattr(module, name, patch_batchnorm(layer))
#                 # all_layers.append(layer)
#         else:
#             get_bn_layers(layer)
#         # else:
#         #     all_layers += get_bn_layers(layer)
#     # return all_layers


# get_bn_layers(model)
