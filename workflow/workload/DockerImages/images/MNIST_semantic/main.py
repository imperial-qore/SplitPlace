import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from sys import argv
import os
import pickle

dirname = 'assets/'
model_name = os.listdir('assets')[0]
model_path = dirname + model_name

data_type = model_name.split('_')[0]
model_type = model_name.split('_')[1]

in_dim = {'MNIST': 28, 'FashionMNIST': 28, 'CIFAR100': 32}
out_dim = {'MNIST': 10, 'FashionMNIST': 10, 'CIFAR100': 100}
chan_dim = {'MNIST': 1, 'FashionMNIST': 1, 'CIFAR100': 3}
multiplier = {'MNIST': 1, 'FashionMNIST': 2, 'CIFAR100': 2}

input_filename, output_filename = argv[1], argv[2]
split = int(output_filename.split("_")[1])

class LayerSplitNet(nn.Module):
    def __init__(self):
        super(LayerSplitNet, self).__init__()
        self.conv1 = nn.Conv2d(chan_dim[data_type], 32*multiplier[data_type], 3, 1)
        self.conv2 = nn.Conv2d(32*multiplier[data_type], 64*multiplier[data_type], 3, 1)
        self.fc1 = nn.Linear(16 * multiplier[data_type] * (in_dim[data_type] - 4) ** 2, 128*multiplier[data_type])
        self.fc2 = nn.Linear(128*multiplier[data_type], out_dim[data_type])

    def forward(self, x, split=0):
        if split in [0, 1]:
            x = self.conv1(x)
            x = F.relu(x)
        if split in [0, 2]:
            x = self.conv2(x)
            x = F.relu(x)
        if split in [0, 3]:
            x = F.max_pool2d(x, 2)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
        if split in [0, 4]:
            x = self.fc2(x)
            x = F.log_softmax(x, dim=1)
        return x

class SemanticSplitNet(nn.Module):
    def __init__(self):
        super(SemanticSplitNet, self).__init__()
        self.semantic = [nn.Sequential(
                nn.Conv2d(chan_dim[data_type], 8*multiplier[data_type], 3, 1),
                nn.ReLU(),
                nn.Conv2d(8*multiplier[data_type], 16*multiplier[data_type], 3, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(4 * multiplier[data_type] * (in_dim[data_type] - 4) ** 2, 64*multiplier[data_type]),
                nn.ReLU(),
                nn.Linear(64*multiplier[data_type], out_dim[data_type]//5)
            ) for _ in range(5)]
        self.semantic = nn.ModuleList(self.semantic)

    def forward(self, x, split=0):
        if split == 0:
            y = [split(x) for split in self.semantic]
            x = F.log_softmax(torch.cat(y, dim=1), dim=1)
        else:
            return self.semantic[split](x)
        return x

if __name__ == '__main__':
    ######## Load Model ########
    assert os.path.exists(model_path)
    checkpoint = torch.load(model_path)
    model = LayerSplitNet() if 'layer' in model_type else SemanticSplitNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    ######## Load Input ########
    with open(input_filename, 'rb') as f:
        inp = pickle.load(f)
    ####### Process Input #######
    out = model(inp, split)
    # print(out)
    ######## Dump Output ########
    with open(output_filename, 'wb') as f:
        pickle.dump(out, f)