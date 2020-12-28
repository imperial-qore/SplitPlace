import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from sys import argv
import os
from utils import *
import pickle

data_type = argv[1]
model_type = argv[2]
operation = argv[3]

in_dim = {'MNIST': 28, 'FashionMNIST': 28, 'CIFAR100': 32}
out_dim = {'MNIST': 10, 'FashionMNIST': 10, 'CIFAR100': 100}
chan_dim = {'MNIST': 1, 'FashionMNIST': 1, 'CIFAR100': 3}
multiplier = {'MNIST': 1, 'FashionMNIST': 2, 'CIFAR100': 2}

path = 'images/'+data_type+'_'+model_type+'/assets/'+data_type+'_'+model_type+"_split"

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

def train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss, num = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx > 100: break
        data, target = data, target
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        total_loss += loss.item(); num += 1
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return total_loss / num

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct / len(test_loader.dataset)

def save_model(filename, model, optimizer, epoch, accuracy_list):
    file_path = filename + ".ckpt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(filename, model):
    optimizer = torch.optim.Adam(model.parameters() , lr=0.0001)
    file_path = filename + ".ckpt"
    if os.path.exists(file_path):
        print(color.GREEN+"Loading pre-trained model: "+file_path+color.ENDC)
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        epoch = -1; accuracy_list = []
        print(color.GREEN+"Creating new model: "+filename+color.ENDC)
    return model, optimizer, epoch, accuracy_list

def setup():
    # Training settings
    torch.manual_seed(1)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = eval("datasets."+data_type+"('data', train=True, download=True,transform=transform)")
    dataset2 = eval("datasets."+data_type+"('data', train=False, transform=transform)")
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2,  batch_size=1000, shuffle=True)

    model = LayerSplitNet() if 'layer' in model_type else SemanticSplitNet()
    model, optimizer, epoch, accuracy_list = load_model(path, model)

    return model, train_loader, test_loader, optimizer, epoch, accuracy_list

def main(model, train_loader, test_loader, optimizer, start_epoch, accuracy_list):
    epochs = 10
    for epoch in range(start_epoch+1, start_epoch + epochs + 1):
        training_loss = train(model, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, test_loader)
        accuracy_list.append((training_loss, test_loss, test_acc))

if __name__ == '__main__':
    model, train_loader, test_loader, optimizer, epoch, accuracy_list = setup()
    with open('test.pt', 'wb') as f:
        data, target = list(test_loader)[0]
        pickle.dump(data, f)
    if 'test' in operation:
        test(model, test_loader)
        plot_accuracies(accuracy_list, data_type+'_'+model_type+"_split")
    else:
        main(model, train_loader, test_loader, optimizer, epoch, accuracy_list)
    save_model(path, model, optimizer, epoch, accuracy_list)