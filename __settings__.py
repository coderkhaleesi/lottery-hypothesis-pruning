import torch
import torch.nn as nn

random_seed = 8420

train_portion = 0.7
valid_portion = 0.15

img_size = 224

max_epochs = 50
batch_size = 64

learning_rate = 0.02
momentum = 0.9

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#dev = 'cpu'
print(f'using {dev}')


def is_pruneable(x):
    return isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear)
