
from debug import debug

import os
import pickle
import json
import time
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np
import torchvision
import torchvision.datasets as datasets
import horovod.torch as hvd
from models import Classifier, Generator, Encoder, sample_model
import inner_optimizers
from gradient_helpers import SurrogateLoss, gradient_checkpointing
import nest
import models
import tabular_logger as tlogger

from mpi4py import MPI


def get_dataset(dataset, data_path, seed, device, with_augmentation=False, cutout_size=16):
    torch.manual_seed(seed)

    if dataset == "MNIST":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
        trainset = datasets.MNIST(root=data_path, train=True, download=hvd.rank() == 0, transform=transform)
        trainset, validationset = torch.utils.data.random_split(trainset, [50000, 10000])
        testset = datasets.MNIST(root=data_path, train=False, download=hvd.rank() == 0, transform=transform)
        img_shape = (1, 28, 28)
    elif dataset == "CIFAR10":
        img_mean = [0.49139968, 0.48215827, 0.44653124]
        img_std = [0.24703233, 0.24348505, 0.26158768]
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(img_mean, img_std),
        ])
        if with_augmentation:
            transform_train = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(img_mean, img_std),
                # Cutout(cutout_size),
            ])
        else:
            transform_train = transform
        img_mean = torch.as_tensor(img_mean).to(device)[..., None, None]
        img_std = torch.as_tensor(img_std).to(device)[..., None, None]
        trainset = datasets.CIFAR10(root=data_path, train=True, download=hvd.rank() == 0, transform=transform_train)
        validationset = datasets.CIFAR10(root=data_path, train=True, download=hvd.rank() == 0, transform=transform)

        # Split train and validationset
        lengths = [45000, 5000]
        indices = torch.randperm(sum(lengths)).tolist()
        trainset = torch.utils.data.dataset.Subset(trainset, indices[:lengths[0]])
        validationset = torch.utils.data.dataset.Subset(validationset, indices[-lengths[1]:])

        testset = datasets.CIFAR10(root=data_path, train=False, download=hvd.rank() == 0, transform=transform)
        img_shape = (3, 32, 32)
    torch.manual_seed(seed + hvd.rank())

    testset_x, testset_y = zip(*testset)
    testset_x = torch.stack(testset_x).to(device)
    testset_y = torch.as_tensor(testset_y).to(device)

    return img_shape, trainset, validationset, (testset_x, testset_y)


def maybe_allreduce_grads(model):
    if hvd.size() > 1:
        tstart_reduce = time.time()
        named_parameters = list(sorted(model.named_parameters(), key=lambda a: a[0]))
        grad_handles = []
        for name, p in named_parameters:
            if p.requires_grad:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                with torch.no_grad():
                    grad_handles.append(hvd.allreduce_async_(p.grad, name=name))
        for handle in grad_handles:
            hvd.synchronize(handle)
        tlogger.record_tabular("TimeElapsedAllReduce", time.time() - tstart_reduce)
        if time.time() - tstart_reduce > 5:
            import socket
            tlogger.info("Allreduce took more than 5 seconds for node {} (rank {})".format(
                socket.gethostname(), hvd.rank()))


def evaluate_set(model, x, y, name):
    with torch.no_grad():
        batch_size = 1000
        validation_pred = []
        model.eval()
        for i in range(math.ceil(len(x) / batch_size)):
            pred = model(x[i * batch_size:(i + 1) * batch_size])
            if isinstance(pred, tuple):
                pred, _ = pred
            validation_pred.append(pred)
        validation_pred = torch.cat(validation_pred, dim=0)
        single_validation_accuracy = (validation_pred.max(-1).indices == y).to(torch.float).mean()

        ensemble_pred = hvd.allreduce(torch.exp(validation_pred), name="{}_ensemble_pred".format(name))
        ensemble_validation_accuracy = (ensemble_pred.max(-1).indices == y).to(torch.float).mean()

        validation_accuracy = hvd.allreduce(single_validation_accuracy, name="{}_accuracy".format(name)).item()
        validation_loss = hvd.allreduce(F.nll_loss(validation_pred, y), name="{}_loss".format(name)).item()
        tlogger.record_tabular('{}_loss'.format(name), validation_loss)
        tlogger.record_tabular('{}_accuracy'.format(name), validation_accuracy)
        tlogger.record_tabular('ensemble_{}_accuracy'.format(name), ensemble_validation_accuracy)
        return validation_loss, single_validation_accuracy, validation_accuracy


class CGTN(nn.Module):
    def __init__(self, generator, num_inner_iterations, generator_batch_size, noise_size, evenly_distributed_labels=False, meta_learn_labels=False):
        super().__init__()
        self.generator_batch_size = generator_batch_size
        self.generator = generator
        if evenly_distributed_labels:
            labels = torch.arange(num_inner_iterations * generator_batch_size) % 10
            labels = torch.reshape(labels, (num_inner_iterations, generator_batch_size))
            self.curriculum_labels = nn.Parameter(labels, requires_grad=False)
        else:
            self.curriculum_labels = nn.Parameter(torch.randint(
                10, size=(num_inner_iterations, generator_batch_size), dtype=torch.int64), requires_grad=False)
        self.curriculum_labels_one_hot = torch.zeros(num_inner_iterations, generator_batch_size, 10)
        self.curriculum_labels_one_hot.scatter_(2, self.curriculum_labels.unsqueeze(-1), 1)
        self.curriculum_labels_one_hot = nn.Parameter(self.curriculum_labels_one_hot, requires_grad=meta_learn_labels)
        # TODO: Maybe learn the soft-labels?
        self.curriculum = nn.Parameter(torch.randn(
            (num_inner_iterations, generator_batch_size, noise_size), dtype=torch.float32))
        self.generator = torch.jit.trace(self.generator, (torch.rand(generator_batch_size, noise_size + 10),))

    def forward(self, it):
        label = self.curriculum_labels_one_hot[it]
        noise = torch.cat([self.curriculum[it], label], dim=-1)
        x = self.generator(noise)
        if not x.requires_grad:
            label = label.detach()
        return x, label


class Learner(nn.Module):
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer


class AutoML(nn.Module):
    def __init__(self, generator, optimizers, initial_batch_norm_momentum=0.9):
        super().__init__()
        self.generator = generator
        self.optimizers = torch.nn.ModuleList(optimizers)
        self.batch_norm_momentum_logit = nn.Parameter(torch.as_tensor(inner_optimizers.inv_sigmoid(0.9)))

    @property
    def batch_norm_momentum(self):
        return torch.sigmoid(self.batch_norm_momentum_logit)

    def sample_learner(self, input_shape, device, allow_nas=False, learner_type="base",
                       iteration_maps_seed=False, iteration=None, deterministic=False, iterations_depth_schedule=100, randomize_width=False):

        if iteration_maps_seed:
            iteration = iteration - 1
            encoding = [iteration % 6, iteration // 6]
        else:
            encoding = None

        if learner_type == "base":
            model = Classifier(input_shape, batch_norm_momentum=0.0, randomize_width=randomize_width)
        elif learner_type == "base_fc":
            model = Classifier(input_shape, batch_norm_momentum=0.0,
                               randomize_width=randomize_width, use_global_pooling=False)
        elif learner_type == "linear":
            model = models.LinearClassifier(input_shape)
        elif learner_type == "base_larger":
            model = models.ClassifierLarger(input_shape, batch_norm_momentum=0.0, randomize_width=randomize_width)
        elif learner_type == "base_larger2":
            model = models.ClassifierLarger2(input_shape, batch_norm_momentum=0.0, randomize_width=randomize_width)
        elif learner_type == "base_larger3":
            model = models.ClassifierLarger3(input_shape, batch_norm_momentum=0.0, randomize_width=randomize_width)
        elif learner_type == "base_larger3_global_pooling":
            model = models.ClassifierLarger3(input_shape, batch_norm_momentum=0.0,
                                             randomize_width=randomize_width, use_global_pooling=True)
        elif learner_type == "base_larger4_global_pooling":
            model = models.ClassifierLarger4(input_shape, batch_norm_momentum=0.0,
                                             randomize_width=randomize_width, use_global_pooling=True)
        elif learner_type == "base_larger4":
            model = models.ClassifierLarger4(input_shape, batch_norm_momentum=0.0,
                                             randomize_width=randomize_width, use_global_pooling=False)
        else:
            raise NotImplementedError()

        return Learner(model=model.to(device), optimizer=np.random.choice(self.optimizers)), encoding


class EndlessDataLoader(object):
    def __init__(self, data_loader):
        self._data_loader = data_loader

    def __iter__(self):
        while True:
            for batch in self._data_loader:
                yield batch


def main(dataset, root, validation_learner_type, seed=1):

    hvd.init()
    # Load dataset
    # dataset = 'MNIST'

    noise_size = 128
    num_inner_iterations = 64
    inner_loop_init_lr = 0.02
    inner_loop_init_momentum = 0.5
    training_iterations_schedule = 0
    # lr = 0.02
    lr = 0
    final_relative_lr = 1e-1
    generator_batch_size = 128
    meta_batch_size = 128
    num_meta_iterations = 1
    # gradient_block_size = 1
    gradient_block_size = 0
    use_intermediate_losses = 1
    data_path = './data'
    logging_period = 50
    learner_type = "base_fc"
    warmup_iterations = None
    warmup_learner = "base"
    iterations_depth_schedule = 100
    load_from = os.path.join(root, 'checkpoint_2000.pkl')
    enable_checkpointing = True
    randomize_width = False
    device = 'cuda'

    img_shape, trainset, validationset, (testset_x, testset_y) = get_dataset(
        dataset, data_path, seed=1, device=device, with_augmentation=False)
    validation_x, validation_y = zip(*validationset)
    validation_x = torch.stack(validation_x).to(device)
    validation_y = torch.as_tensor(validation_y).to(device)

    # # Make each worker slightly different
    torch.manual_seed(seed)
    # np.random.seed(seed + hvd.rank())

    data_loader = torch.utils.data.DataLoader(
        trainset, batch_size=meta_batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True)
    data_loader = EndlessDataLoader(data_loader)

    generator = CGTN(
        generator=Generator(noise_size + 10, img_shape),
        num_inner_iterations=num_inner_iterations,
        generator_batch_size=generator_batch_size,
        noise_size=noise_size,
        evenly_distributed_labels=True,
        meta_learn_labels=False,
    )

    # Create meta-objective models
    optimizers = [inner_optimizers.SGD(inner_loop_init_lr, inner_loop_init_momentum, num_inner_iterations)]

    automl = AutoML(
        generator=generator,
        optimizers=optimizers,
    )
    automl = automl.to(device)

    optimizer = torch.optim.Adam(automl.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_meta_iterations, lr * final_relative_lr)
    if hvd.rank() == 0:
        if load_from:
            state = torch.load(load_from)
            automl.load_state_dict(state["model"])
            if lr > 0:
                optimizer.load_state_dict(state["optimizer"])
            del state
            tlogger.info("loaded from:", load_from)
        # total_num_parameters = 0
        # for name, value in automl.named_parameters():
        #     tlogger.info("Optimizing parameter:", name, value.shape)
        #     total_num_parameters += np.prod(value.shape)
        # tlogger.info("Total number of parameters:", int(total_num_parameters))

    def compute_learner(learner, iterations=num_inner_iterations, callback=None):
        learner.model.train()

        names, params = list(zip(*learner.model.get_parameters()))
        buffers = list(zip(*learner.model.named_buffers()))
        if buffers:
            buffer_names, buffers = buffers
        else:
            buffer_names, buffers = None, ()
        param_shapes = [p.shape for p in params]
        param_sizes = [np.prod(shape) for shape in param_shapes]
        param_end_point = np.cumsum(param_sizes)

        buffer_shapes = [p.shape for p in buffers]
        buffer_sizes = [np.prod(shape) for shape in buffer_shapes]
        buffer_end_point = np.cumsum(buffer_sizes)

        def split_params(fused_params):
            # return fused_params
            assert len(fused_params) == 1
            return [fused_params[0][end - size:end].reshape(shape) for end, size, shape in zip(param_end_point, param_sizes, param_shapes)]

        def split_buffer(fused_params):
            if fused_params:
                # return fused_params
                assert len(fused_params) == 1
                return [fused_params[0][end - size:end].reshape(shape) for end, size, shape in zip(buffer_end_point, buffer_sizes, buffer_shapes)]
            return fused_params

        # test = split_params(torch.cat([p.flatten() for p in params]))
        # assert all([np.allclose(params[i].detach().cpu(), test[i].detach().cpu()) for i in range(len(test))])
        params = [torch.cat([p.flatten() for p in params])]
        buffers = [torch.cat([p.flatten() for p in buffers])] if buffers else buffers
        optimizer_state = learner.optimizer.initial_state(params)

        params = params, buffers
        initial_params = nest.map_structure(lambda p: None, params)

        losses = {}
        accuracies = {}

        # @debug
        def intermediate_loss(params):
            params = nest.pack_sequence_as(initial_params, params[1:])
            params, buffers = params
            x, y = next(meta_generator)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # debug((x, y))
            learner.model.set_parameters(list(zip(names, split_params(params))))
            if buffer_names:
                learner.model.set_buffers(list(zip(buffer_names, split_buffer(buffers))))
            learner.model.eval()
            pred = learner.model(x)
            if isinstance(pred, tuple):
                pred, aux_pred = pred
                loss = F.nll_loss(pred, y) + F.nll_loss(aux_pred, y)
            else:
                loss = F.nll_loss(pred, y)
            # debug(loss)
            return loss

        learner.it = 0

        def body(args):
            it, params, optimizer_state = args
            x, y_one_hot = automl.generator(it)
            learner.it = it.item()

            with torch.enable_grad():
                if use_intermediate_losses > 0 and (it >= use_intermediate_losses and it % use_intermediate_losses == 0):
                    params = SurrogateLoss.apply(intermediate_loss, it, *nest.flatten(params))
                    params = nest.pack_sequence_as(initial_params, params[1:])
                params, buffers = params
                for p in params:
                    if not p.requires_grad:
                        p.requires_grad = True

                # if it < 4:
                #     parx = list(split_params(params))
                #     debug(parx)

                learner.model.set_parameters(list(zip(names, split_params(params))))
                if buffer_names:
                    learner.model.set_buffers(list(zip(buffer_names, split_buffer(buffers))))
                learner.model.train()
                # debug(x)
                output = learner.model(x)
                # debug(output)
                loss = -(output * y_one_hot).sum() * (1 / output.shape[0])
                pred = output
                # debug(loss)
                if it.item() not in losses:
                    losses[it.item()] = loss.detach().cpu().item()
                    accuracies[it.item()] = (pred.max(-1).indices ==
                                             y_one_hot.max(-1).indices).to(torch.float).mean().item()

                grads = grad(loss, params, create_graph=x.requires_grad, allow_unused=True)
            # assert len(grads) == len(names)
            new_params, optimizer_state = learner.optimizer(it, params, grads, optimizer_state, split_params)

            buffers = list(learner.model.buffers())
            buffers = [torch.cat([b.flatten() for b in buffers])] if buffers else buffers
            if callback is not None:
                learner.model.set_parameters(list(zip(names, split_params(params))))
                if buffer_names:
                    learner.model.set_buffers(list(zip(buffer_names, split_buffer(buffers))))
                callback(learner)

            return (it + 1, (new_params, buffers,), optimizer_state)

        # state = body_fn(state)
        # res = body((1, optimizer_state))
        last_state, params, optimizer_state = gradient_checkpointing((torch.as_tensor(0), params, optimizer_state), body, iterations,
                                                                     block_size=gradient_block_size)

        # import ipdb
        # ipdb.set_trace()
        assert last_state.item() == iterations
        params, buffers = params
        learner.model.set_parameters(list(zip(names, split_params(params))))
        if buffer_names:
            learner.model.set_buffers(list(zip(buffer_names, split_buffer(buffers))))

        return learner, losses, accuracies

    tstart = time.time()
    meta_generator = iter(data_loader)
    hvd.broadcast_parameters(automl.state_dict(), root_rank=0)
    best_optimizers = {}
    validation_accuracy = None
    total_inner_iterations_so_far = 0
    for iteration in range(1, num_meta_iterations + 1):
        last_iteration = time.time()
        # basic logging
        tlogger.record_tabular('Iteration', iteration)
        tlogger.record_tabular('lr', optimizer.param_groups[0]['lr'])

        # Train learner
        if training_iterations_schedule > 0:
            training_iterations = int(min(num_inner_iterations, 4 + (iteration - 1) // training_iterations_schedule))
        else:
            training_iterations = num_inner_iterations
        tlogger.record_tabular('training_iterations', training_iterations)
        total_inner_iterations_so_far += training_iterations
        tlogger.record_tabular('training_iterations_so_far', total_inner_iterations_so_far * hvd.size())

        optimizer.zero_grad()

        for _ in range(1):
            torch.cuda.empty_cache()
            meta_x, meta_y = next(meta_generator)
            meta_x = meta_x.to('cuda', non_blocking=True)
            meta_y = meta_y.to('cuda', non_blocking=True)

            tstart_forward = time.time()

            sample_learner_type = learner_type
            if warmup_iterations is not None and iteration < warmup_iterations:
                sample_learner_type = warmup_learner
            # torch.manual_seed(1)
            # np.random.seed(1)
            learner, encoding = automl.sample_learner(img_shape, device,
                                                      allow_nas=False,
                                                      randomize_width=randomize_width,
                                                      learner_type=sample_learner_type,
                                                      iteration_maps_seed=False,
                                                      iteration=iteration,
                                                      deterministic=False,
                                                      iterations_depth_schedule=iterations_depth_schedule)
            automl.train()

            learner, intermediate_losses, intermediate_accuracies = compute_learner(
                learner, iterations=training_iterations)
            # TODO: remove this requirement
            params = list(learner.model.get_parameters())
            learner.model.eval()

            # Evaluate learner on training and back-prop
            torch.cuda.empty_cache()
            pred = learner.model(meta_x)
            if isinstance(pred, tuple):
                pred, aux_pred = pred
                loss = F.nll_loss(pred, meta_y) + F.nll_loss(aux_pred, meta_y)
            else:
                loss = F.nll_loss(pred, meta_y)
            accuracy = (pred.max(-1).indices == meta_y).to(torch.float).mean()
            tlogger.record_tabular("TimeElapsedForward", time.time() - tstart_forward)
            num_parameters = sum([a[1].size().numel() for a in params])
            tlogger.record_tabular("TrainingLearnerParameters", num_parameters)
            tlogger.record_tabular("optimizer", type(learner.optimizer).__name__)
            tlogger.record_tabular('meta_training_loss', loss.item())
            tlogger.record_tabular('meta_training_accuracy', accuracy.item())
            tlogger.record_tabular('training_accuracies', intermediate_accuracies)
            tlogger.record_tabular('training_losses', intermediate_losses)
            tlogger.record_tabular("dag", encoding)

            if lr > 0.0:
                tstart_backward = time.time()
                # if semisupervised_student_loss:
                loss.backward()

                tlogger.record_tabular("TimeElapsedBackward", time.time() - tstart_backward)

        if lr > 0.0:
            # If using distributed training aggregard gradients with Horovod
            maybe_allreduce_grads(automl)
            optimizer.step()
            scheduler.step(iteration - 1)

        is_last_iteration = iteration == num_meta_iterations
        if np.isnan(loss.item()):
            tlogger.info("NaN training loss, terminating")
            is_last_iteration = True
        is_last_iteration = MPI.COMM_WORLD.bcast(is_last_iteration, root=0)
        if iteration == 1 or iteration % logging_period == 0 or is_last_iteration:
            tstart_validation = time.time()

            val_loss, val_accuracy = [], []
            test_loss, test_accuracy = [], []

            # print('Saving learner model')
            path = os.path.join(root, '10_gtn')
            if not os.path.exists(path):
                os.makedirs(path)
            count = len([model for model in os.listdir(path) if validation_learner_type in model])
            path = f'{path}/{validation_learner_type}_{count + 1}.pt'

            def compute_learner_callback(learner):
                if hasattr(learner, 'it') and learner.it == 10:
                    torch.save(learner.model, path)
                # Validation set
                validation_loss, single_validation_accuracy, validation_accuracy = evaluate_set(
                    learner.model, validation_x, validation_y, "validation")
                val_loss.append(validation_loss)
                val_accuracy.append(validation_accuracy)
                best_optimizers[type(learner.optimizer).__name__] = single_validation_accuracy.item()
                # Test set
                loss, _, accuracy = evaluate_set(learner.model, testset_x, testset_y, "test")
                test_loss.append(loss)
                test_accuracy.append(accuracy)

            tlogger.info("sampling another learner_type ({}) for validation".format(validation_learner_type))
            # debug('before', learner.model.convs[0].weight)
            learner, _ = automl.sample_learner(img_shape, device,
                                               allow_nas=False,
                                               learner_type=validation_learner_type,
                                               iteration_maps_seed=False,
                                               iteration=iteration,
                                               deterministic=False,
                                               iterations_depth_schedule=iterations_depth_schedule
                                               )
            compute_learner_callback(learner)
            with torch.no_grad():
                learner, _, _ = compute_learner(learner, iterations=training_iterations,
                                                callback=compute_learner_callback)
            # debug(learner.model._named_parameters)
            # debug.embed()

            tlogger.record_tabular("validation_losses", val_loss)
            tlogger.record_tabular("validation_accuracies", val_accuracy)
            validation_accuracy = val_accuracy[-1]
            tlogger.record_tabular("test_losses", test_loss)
            tlogger.record_tabular("test_accuracies", test_accuracy)

            # Extra logging
            tlogger.record_tabular('TimeElapsedIter', tstart_validation - last_iteration)
            tlogger.record_tabular('TimeElapsedValidation', time.time() - tstart_validation)
            tlogger.record_tabular('TimeElapsed', time.time() - tstart)

            for k, v in best_optimizers.items():
                tlogger.record_tabular("{}_last_accuracy".format(k), v)

            # print(learner.model)
            print()
            print(('{:>15}' * 4).format('', 'train', 'val', 'test'))
            for i, (train, val, test) in enumerate(zip(intermediate_accuracies.values(), val_accuracy, test_accuracy)):
                print(('{:>15}' + '{:>15.3f}' * 3).format(i, train, val, test))

            print('Saving learner model')
            path = os.path.join(root, 'gtn')
            if not os.path.exists(path):
                os.makedirs(path)
            count = len([model for model in os.listdir(path) if validation_learner_type in model])
            path = f'{path}/{validation_learner_type}_{count + 1}.pt'
            torch.save(learner.model, path)

            # # EXTRA VALIDATION
            # _, trainset2, validationset2, _ = get_dataset(
            #     dataset, data_path, seed=7777, device=device, with_augmentation=False)
            # validation_x2, validation_y2 = zip(*validationset2)
            # validation_x2 = torch.stack(validation_x2).to(device)
            # validation_y2 = torch.as_tensor(validation_y2).to(device)

            # train_x2, train_y2 = zip(*trainset2)
            # train_x2 = torch.stack(train_x2).to(device)
            # train_y2 = torch.as_tensor(train_y2).to(device)

            # valsets = {'validation': (validation_x, validation_y),
            #            'trainset2': (train_x2, train_y2),
            #            'validation2': (validation_x2, validation_y2),
            #            'test': (testset_x, testset_y), }
            # from utils import evaluate_set as evalx

            # for name, (x, y) in valsets.items():
            #     acc = evalx(learner.model, x, y)
            #     print(f'{name} accuracy: {acc:.3f}')

            if is_last_iteration:
                break
        elif hvd.rank() == 0:
            tlogger.info("training_loss:", loss.item())
    return validation_accuracy


if __name__ == "__main__":
    from tabular_logger import set_tlogger
    set_tlogger("default")

    dataset = 'CIFAR10'
    # dataset = 'MNIST'
    # learner_type = 'base_larger'
    learner_type = 'base_larger3'
    root = f'runs/{dataset}/{learner_type}'

    for ltype in [
        'base',
        #   'base_fc',
        'linear',
        'base_larger',
        'base_larger2',
        'base_larger3',
        'base_larger3_global_pooling',
        'base_larger4_global_pooling',
        'base_larger4',
    ]:

        for i in range(5):
            main(dataset, root, validation_learner_type=ltype, seed=i)
