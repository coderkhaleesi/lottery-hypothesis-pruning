from collections import defaultdict
import os
import pickle
import torch
import torch.nn.functional as F

from __settings__ import dev, random_seed, is_pruneable


def run_epoch(model, optimizer, datasource, is_training, loss_func=F.cross_entropy, mask=None):
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_acc = 0
    count = 0
    with torch.set_grad_enabled(is_training):
        for x, y in datasource:
            x = x.to(dev)
            y = y.squeeze(-1).to(dev)
            yh = model(x)
            loss = loss_func(yh, y)
            total_loss += loss.item() * len(y)
            total_acc += (yh.argmax(-1) == y).sum().float().item()
            count += len(y)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if mask is not None:
                    for name, mod in model.named_modules():
                        if not is_pruneable(mod):
                            continue
                        m = mask[name]
                        mod.weight.data *= m

    return total_loss/count, total_acc/count


def run_one_pass(model, datasets):
    def save_activations(acts, name):
        fp = os.path.join('saved_activations', f'{name}.pkl')
        if os.path.exists(fp):
            with open(fp, 'rb') as f:
                all_acts = pickle.load(f)
        else:
            all_acts = []
        with open(fp, 'wb') as f:
            all_acts.append(acts)
            pickle.dump(all_acts, f)
        del all_acts
        del acts
    hooks = []
    for name, mod in model.named_modules():
        if is_pruneable(mod):
            hooks.append(mod.register_forward_hook(lambda self, input, output, name=name:
                                      save_activations((output if len(output.shape) == 2 else
                                                                output.mean(-1).mean(-1)).detach().cpu().numpy(), name)))

    for x, y in datasets[0]:
        model(x.to(dev))
        del x
        del y

    for h in hooks:
        h.remove()


def full_train(model, optimizer, datasets, name, max_epochs=25, mask=None):
    if not os.path.exists(os.path.join('saved_models', name)):
        os.mkdir(os.path.join('saved_models', name))
    torch.save(model, os.path.join('saved_models', name, f'initial.pth'))

    train, valid, test = datasets
    train_scores = []
    valid_scores = []
    best_epoch = 0
    best_acc = 0
    for epoch in range(max_epochs):
        train_scores.append(run_epoch(model, optimizer, train, is_training=True, mask=mask))
        valid_scores.append(run_epoch(model, None, valid, is_training=False))
        print(f'Epoch {epoch}: train {train_scores[-1][1]}, valid {valid_scores[-1][1]}')
        if valid_scores[-1][1] > best_acc:
            best_acc = valid_scores[-1][1]
            best_epoch = epoch
            torch.save(model, os.path.join('saved_models', name, f'{epoch}.pth'))

    model = torch.load(os.path.join('saved_models', name, f'{best_epoch}.pth'))
    test_scores = run_epoch(model, None, test, is_training=False)
    print()
    print(f'Epoch {best_epoch}: test {test_scores[1]}')

    with open(os.path.join('experiment_results', name), 'wb') as f:
        pickle.dump((train_scores, valid_scores, test_scores, best_epoch), f)
    torch.save(model, os.path.join('saved_models', name, 'best.pth'))

