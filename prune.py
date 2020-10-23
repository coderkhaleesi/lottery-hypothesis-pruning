import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from __settings__ import *
from data_loader import dataset
from distinctiveness_pruning import distinctiveness_prune
from training import full_train, run_epoch, run_one_pass


model_init = torch.load(os.path.join('saved_models', 'original', 'initial.pth')).to(dev)
initial_values = [mod.weight.data for mod in model_init.modules() if is_pruneable(mod)]


def random_prune(model, percentage):
    params_to_prune = [(mod, 'weight') for mod in model.modules() if is_pruneable(mod)]
    prune.global_unstructured(params_to_prune, prune.RandomUnstructured, amount=percentage)


def magnitude_prune(model, percentage):
    params_to_prune = [(mod, 'weight') for mod in model.modules() if is_pruneable(mod)]
    prune.global_unstructured(params_to_prune, prune.L1Unstructured, amount=percentage)


def prune_continue(prune_method, name):
    model = torch.load(os.path.join('saved_models', 'original', 'best.pth')).to(dev)
    mask = prune_method(model)
    if mask is not None:
        torch.save(mask, os.path.join('experiment_results', f'{name}_mask'))
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    full_train(model, optimizer, dataset, f'continue_{name}', max_epochs=max_epochs, mask=mask)


def prune_reinit(prune_method, name):
    model = torch.load(os.path.join('saved_models', 'original', 'best.pth')).to(dev)
    mask = prune_method(model)
    if mask is not None:
        torch.save(mask, os.path.join('experiment_results', f'{name}_mask'))

    i = 0
    for mod in model.modules():
        if is_pruneable(mod):
            if hasattr(mod, 'weight_orig'):
                mod.weight_orig.data = initial_values[i].detach().clone()
            else:
                mod.weight.data = initial_values[i].detach().clone()
            i += 1

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    full_train(model, optimizer, dataset, f'reinit_{name}', max_epochs=max_epochs, mask=mask)


#prune_reinit(model_ft, lambda m: magnitude_prune(m, magnitude_percent), f'magnitude_{magnitude_percent}')
#prune_reinit(model_ft, lambda m: distinctiveness_prune(m, min_angle), f'distinct_{min_angle}')