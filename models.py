import torch
import torchvision

resnet18 = torchvision.models.resnet18(pretrained=True)


def replace_last_layer(model, num_classes):
    d_in = model.fc.in_features
    model.fc = torch.nn.Linear(d_in, num_classes)
    return model


def resnet(num_classes):
    return replace_last_layer(resnet18, num_classes)
