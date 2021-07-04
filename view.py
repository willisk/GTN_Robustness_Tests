import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from art.estimators.classification import PyTorchClassifier

from debug import debug
import utils

from train_cgtn import AutoML
automl = AutoML(None, optimizers=[None])
criterion = nn.CrossEntropyLoss()

# dataset = 'MNIST'
dataset = 'CIFAR10'
# learner_type = 'base_larger'
learner_type = 'base_larger3'
root = f'runs/{dataset}/{learner_type}'
seed = 1

recompute = False


input_shape, trainset, validationset, testset = utils.cgtn_get_dataset(dataset, 'data', seed, 'cuda')

train_loader = DataLoader(trainset, batch_size=128)
val_loader = DataLoader(validationset, batch_size=128)
test_loader = DataLoader(testset, batch_size=128)

valsets = {
    'val': val_loader,
    'test': test_loader,
}

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


def train_vanilla(basepath):
    for ltype in ltypes:
        for i in range(5):
            automl_model, _ = automl.sample_learner(input_shape=input_shape, learner_type=ltype, device='cuda')
            model = automl_model.model
            lr = 0.005
            optimizer = optim.Adam(model.parameters(), lr=lr)
            acc = utils.fit_generator(model, train_loader, optimizer, criterion, epochs=1, valsets=valsets, silent=True)

            print(f'{ltype} with accuracy {acc[-1]["train"]:.3f}%')

            count = len([model for model in os.listdir(basepath) if ltype in model])
            path = os.path.join(basepath, f'{ltype}_{count + 1}.pt')
            torch.save(model, path)


path = os.path.join(root, 'vanilla')
if not os.path.exists(path):
    os.makedirs(path)
    train_vanilla(path)


x_test, labels = tuple(zip(*testset))
x_test = [x.numpy() for x in x_test]
y_test = torch.zeros((len(labels), 10))
y_test.scatter_(1, torch.as_tensor(labels).unsqueeze(-1), 1)
y_test = y_test.numpy()


# import matplotlib.pyplot as plt
# plt.imshow(x_test)


import pandas as pd
from collections import defaultdict

results = {}
# results = pd.DataFrame(torch.load(os.path.join(root, 'cls_results.txt')))
if not recompute and os.path.exists(os.path.join(root, 'cls_results.txt')):
    results = pd.DataFrame(torch.load(os.path.join(root, 'cls_results.txt')))
else:
    for experiment in os.listdir(root):
        if not os.path.isdir(os.path.join(root, experiment)):
            continue
        results[experiment] = defaultdict(list)
        print(f'\n===== experiment: {experiment} =====')
        path = os.path.join(root, experiment)

        for model_path in sorted(os.listdir(path)):
            base_model = '_'.join(model_path.split('_')[:-1])
            # if not 'larger4' in base_model:
            #     continue

            model_path = os.path.join(path, model_path)
            print(f'--> model {base_model}: {model_path}')
            model = torch.load(model_path)

            classifier = PyTorchClassifier(
                model=model,
                loss=criterion,
                input_shape=input_shape,
                nb_classes=10,
            )

            predictions = classifier.predict(x_test)
            accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
            print(f"Accuracy on test examples: {accuracy * 100:.2f}%")

            results[experiment][base_model].append(accuracy)
        #     break
        # break

    torch.save(results, os.path.join(root, 'cls_results.txt'))
    results = pd.DataFrame(results)

# %%
from IPython.display import display
display(results)
print('Applying mean:')
display(results.applymap(np.mean))
print('By applying mean and taking min:')
t = results.applymap(np.mean).min(axis=1)
filter = t > 0.7

print(f'removed:')
display(t.loc[~filter])

print('remaining:')
display(t.loc[filter])

# df = results.loc[filter]

# # df
# df.plot(kind='box', columns=df.index)
