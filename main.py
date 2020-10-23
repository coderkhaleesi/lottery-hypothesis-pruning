import torch

from models import resnet
from training import full_train
from data_loader import dataset, num_classes

from __settings__ import *

model = resnet(num_classes).to(dev)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

full_train(model, optimizer, dataset, 'original', max_epochs=max_epochs)
