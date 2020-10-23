from prune import prune_reinit, prune_continue, magnitude_prune, random_prune
from distinctiveness_pruning import distinctiveness_prune

n_reps = 5
angles = [1, 5, 10, 15, 30]
percentages = [0.5, 0.9, 0.95, 0.99, 0.999]


for n in range(n_reps):
    print(f'iter {n}')
    for p in percentages:
        print(f'p: {p}')
        prune_reinit(lambda m: magnitude_prune(m, p), f'magnitude_{p}_percent_iter_{n}')
        prune_continue(lambda m: magnitude_prune(m, p), f'magnitude_{p}_percent_continue_iter_{n}')
        prune_reinit(lambda m: random_prune(m, p), f'random_{p}_percent_iter_{n}')
    for a in angles:
        print(f'a: {a}')
        prune_reinit(lambda m: distinctiveness_prune(m, a), f'distinct_{a}_angle_iter_{n}')
        prune_continue(lambda m: distinctiveness_prune(m, a), f'distinct_{a}_angle_continue_iter_{n}')