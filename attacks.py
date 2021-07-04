import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from utils import get_params, fgsm, noise_corrupt, blur, lbfgs

import matplotlib.pyplot as plt
from debug import debug
import utils

from train_cgtn import AutoML
automl = AutoML(None, optimizers=[None])
criterion = nn.CrossEntropyLoss()

dataset = 'MNIST'
learner_type = 'base_larger'
root = f'runs/{dataset}/{learner_type}'
device = 'cuda'
seed = 1

import sys
# attack = sys.argv[1]
attack = 'noise'
print(f'attack: {attack}')


input_shape, trainset, validationset, testset = utils.cgtn_get_dataset(dataset, 'data', seed, 'cuda')

min_pixel_value = testset.transform(np.zeros(input_shape)).min().item()
max_pixel_value = testset.transform(np.ones(input_shape)).min().item()

print(f'min {min_pixel_value:.2f} max {max_pixel_value:.2f}')

train_loader = DataLoader(trainset, batch_size=128)
val_loader = DataLoader(validationset, batch_size=128)
test_loader = DataLoader(testset, batch_size=128)

x_test, y_test = tuple(zip(*testset))
x_test = torch.cat(x_test).unsqueeze(1).to(device)
y_test = torch.as_tensor(y_test).to(device)

x_test = x_test[:1000]
y_test = y_test[:1000]

import pandas as pd
cls_results = pd.DataFrame(torch.load(os.path.join(root, 'cls_results.txt')))
filter = cls_results.applymap(np.mean).min(axis=1) > 0.7
df = cls_results.loc[filter]
allowed_models = list(df.index)
if 'base_fc' in allowed_models:
    allowed_models.remove('base_fc')


def defaultdict_list():
    return defaultdict(list)


from collections import defaultdict
results = defaultdict(defaultdict_list)


samples = {}

if os.path.exists(os.path.join(root, f'{attack}_results.txt')):
    results = pd.DataFrame(torch.load(os.path.join(root, f'{attack}_results.txt')))
else:
    for experiment in os.listdir(root):
        if not os.path.isdir(os.path.join(root, experiment)):
            continue
        print(f'\n===== experiment: {experiment} =====')
        path = os.path.join(root, experiment)
        # if not 'vanilla' in experiment:
        #     continue
        for model_path in sorted(os.listdir(path)):
            base_model = '_'.join(model_path.split('_')[:-1])
            if base_model not in allowed_models:
                continue
            # if not '_1' in model_path and '_2' not in model_path:
            #     continue

            model_path = os.path.join(path, model_path)
            model = torch.load(model_path, map_location=device)
            print(f'--> model {base_model}: {model_path}')
            for module in model.modules():
                module.disable_exconv = True

            optimizer = optim.Adam(get_params(model), lr=0.01)
            accuracies = {}

            steps = 10
            if attack == 'noise':
                attack_fn = noise_corrupt
                attack_vals = torch.linspace(0, 1.6, 10)

            if attack == 'blur':
                attack_fn = blur
                attack_vals = torch.linspace(0.01, 5, 10)

            if attack == 'lbfgs':
                attack_fn = lbfgs
                attack_vals = torch.Tensor([20, 50, 100, 200, 500, 1000, 2000])
                steps = 30

            if attack == 'fgsm':
                attack_fn = fgsm
                attack_vals = torch.linspace(0.001, 0.02, 6)
                steps = 30

            with torch.no_grad():
                predictions = model(x_test)
                accuracy = (predictions.argmax(dim=1) == y_test).float().mean().item()
                print(f"Accuracy on benign test examples: {accuracy * 100:.2f}%")

            for beta in attack_vals:
                beta = beta.item()
                x_test_adv = attack_fn(model, x_test, y_test, min_pixel_value,
                                       max_pixel_value, beta=1 / beta if attack == 'lbfgs' else beta, steps=steps, device=device)

                # if model_path == 'base_larger3_1' or attack in ('fgsm', 'lbfgs') and '_1' in model_path:
                #     plt.imshow(make_grid(x_test_adv[:8].cpu()).permute(1, 2, 0))
                #     plt.show()

                with torch.no_grad():
                    predictions = model(x_test_adv)
                    accuracy = (predictions.argmax(dim=1) == y_test).float().mean().item()
                    print(f"beta {beta:.3f} Accuracy: {accuracy * 100:.2f}%")
                    accuracies[beta] = accuracy

                samples[beta] = x_test_adv[3]

            # plt.imshow(make_grid(x_test_adv[:8].cpu()).permute(1, 2, 0))
            # plt.show()

            results[experiment][base_model].append(accuracies)

    torch.save(results, os.path.join(root, f'{attack}_results.txt'))
    results = pd.DataFrame(results)

plt.figure(figsize=(14, 8))
for i, beta in enumerate(samples.keys()):
    plt.subplot(1, len(samples), i + 1)
    plt.axis('off')
    plt.title(r'$\beta=' + f'{beta:.0f}$')
    plt.imshow(samples[beta].cpu().squeeze())
plt.savefig(os.path.join(root, f'samples_{attack}.pdf'))
plt.show()

exp_data = {}
for experiment in results:
    for model in results[experiment].index:
        data = results[experiment].loc[model]
        beta = data[0].keys()
        data = np.stack([list(d.values()) for d in data])
        line = data.mean(axis=0)
        std = data.std(axis=0)
        plt.plot(beta, line, label=model)
        plt.fill_between(beta, line + std, line - std, alpha=0.6)
    plt.title(experiment)
    plt.legend()
    plt.show()

    exp_data[experiment] = results[experiment].apply(lambda x: np.array([list(y.values()) for y in x])).mean()

# %%

for experiment in exp_data:
    data = exp_data[experiment]
    line = data.mean(axis=0)
    std = data.std(axis=0)
    plt.plot(beta, line, label=experiment)
    plt.fill_between(beta, line + std, line - std, alpha=0.6)
if attack == 'lbgfs':
    plt.xscale('log')
plt.xlabel(r'$\beta$')
plt.ylabel('accuracy')
plt.title(attack)
plt.legend()
plt.savefig(os.path.join(root, f'{attack}.pdf'))
plt.show()
