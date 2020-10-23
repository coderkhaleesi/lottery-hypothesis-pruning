from collections import defaultdict
import torch
import os
import pickle
import numpy as np
import shutil

from __settings__ import is_pruneable
from training import run_one_pass
from data_loader import dataset


def compute_angle(x, y):
    dot_norm = (x * y).sum() / (torch.norm(x, p=2) * torch.norm(y, p=2))
    return torch.acos(dot_norm).abs() * 180 / np.pi


def distinctiveness_prune(model, min_angle):
    shutil.rmtree('saved_activations')
    os.mkdir('saved_activations')
    run_one_pass(model, dataset)

    module_to_match_indexes = defaultdict(lambda: defaultdict(list))
    module_to_mask = dict()
    for fp in os.listdir('saved_activations'):
        with open(os.path.join('saved_activations', fp), 'rb') as f:
            acts = pickle.load(f)
            acts = torch.cat([torch.from_numpy(a) for a in acts]).t()
            for i in range(len(acts)):
                for j in range(i):
                    if compute_angle(acts[i], acts[j]) < min_angle:
                        module_to_match_indexes[fp[:-4]][i].append(j)

    for name, mod in model.named_modules():
        if not is_pruneable(mod):
            continue
        js_set = set()
        for i, js in module_to_match_indexes[name].items():
            for j in js:
                mod.weight.data[i] += mod.weight.data[j]
                mod.weight.data[j] = 0
            js_set = js_set.union(set(js))
        mask = torch.ones_like(mod.weight.data)
        for j in js_set:
            mask[j] = 0
        module_to_mask[name] = mask

    return module_to_mask

