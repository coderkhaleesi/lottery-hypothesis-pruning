from collections import defaultdict
import matplotlib.pyplot as plt
import os
import pickle


name_to_test_accs = defaultdict(list)
for fp in os.listdir('experiment_results'):
    if '_iter_' not in fp:
        continue
    name, rep = fp.split('_iter_')
    with open(os.path.join('experiment_results', fp), 'rb') as f:
        try:
            train_scores, valid_scores, test_scores, best_acc = pickle.load(f)
        except TypeError:
            continue

    metrics = ['loss', 'acc']
    modes = ['train', 'valid']
    metric_to_mode = defaultdict(lambda: defaultdict(list))
    mode_to_scores = {'train': train_scores, 'valid': valid_scores}
    for i in range(len(metrics)):
        m = metrics[i]
        for mode in modes:
            for epoch in range(len(train_scores)):
                metric_to_mode[m][mode].append(mode_to_scores[mode][epoch][i])

    if not os.path.exists(os.path.join('outputs', name)):
        os.mkdir(os.path.join('outputs', name))

    for met in metrics:
        for mode in modes:
            plt.plot(metric_to_mode[met][mode])
            plt.xlabel('Epoch')
            plt.ylabel(met)
            fig_name = f'{name} {mode}'
            plt.title(fig_name)
            plt.savefig(os.path.join('outputs', name, f'{mode}.png'))
            plt.clf()

    name_to_test_accs[name].append(test_scores[1])

for name in name_to_test_accs:
    with open(os.path.join('outputs', name,  'average test accuracy.txt'), 'w') as f:
        accs = name_to_test_accs[name]
        f.write(f'{sum(accs) / len(accs)}')