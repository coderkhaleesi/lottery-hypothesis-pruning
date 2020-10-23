import os
import pickle

with open(os.path.join('experiment_results', 'original'), 'rb') as f:
    x = pickle.load(f)
    print(x)