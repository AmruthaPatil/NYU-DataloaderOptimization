import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import warnings

parser = argparse.ArgumentParser()
parser.add_argument("--num-workers", type=int, default=2)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--data-path-train", type=str, default="data/cifar10/train")
parser.add_argument("--data-path-test", type=str, default="data/cifar10/test")
parser.add_argument("--optimizer", type=str, default="sgd")
parser.add_argument("--disable-batchnorm", action="store_true", default=False)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight-decay", type=float, default=5e-4)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--run-test", action="store_true", default=False)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, disable_batchnorm=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if disable_batchnorm:
            self.bn1 = Identity()
            self.bn2 = Identity() 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes) if not disable_batchnorm else Identity())
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, disable_batchnorm=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if disable_batchnorm:
            self.bn1 = Identity()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, disable_batchnorm=disable_batchnorm)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, disable_batchnorm=disable_batchnorm)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, disable_batchnorm=disable_batchnorm)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, disable_batchnorm=disable_batchnorm)
        self.linear = nn.Linear(512*block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride, disable_batchnorm=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, disable_batchnorm=disable_batchnorm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(disable_batchnorm=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], disable_batchnorm=disable_batchnorm)

class Trainer:
    def __init__(self, args):
        self.cnn = ResNet18(args.disable_batchnorm)
        self.cnn.to(args.device)
        self.num_workers = args.num_workers
        self.device = args.device
        self.epochs = args.epochs
        self.run_test = args.run_test
        self.trainable_params = None
        self.gradients = None
        if args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.cnn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.cnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError(f"Error: optimizer {args.optimizer}")
    def get_train_data(self):
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=.5), transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        trainset = torchvision.datasets.CIFAR10(root=args.data_path_train, train=True, download=True, transform=transform_train)
        self.train_dl = torch.utils.data.DataLoader(trainset,batch_size = 128,shuffle = True, drop_last = True,
            num_workers = self.num_workers, pin_memory = self.device == "cuda")
    def get_test_data(self):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        testset = torchvision.datasets.CIFAR10(root=args.data_path_test, train=False, download=True, transform=transform_test)
        self.test_dl = torch.utils.data.DataLoader(dataset = testset, batch_size = 128, shuffle=False, drop_last = False,
                num_workers = self.num_workers, pin_memory = self.device == "cuda")
    def train_loop(self):
        print_out = ""
        self.cnn.train()
        self.trainable_params = sum(p.numel() for p in self.cnn.parameters() if p.requires_grad)
        for i in range(self.epochs):
            n_batches = len(self.train_dl)
            train_dl_i = iter(self.train_dl)
            for j in range(n_batches):
                x,y = next(train_dl_i)
                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                predictions = self.cnn(x)
                loss = F.cross_entropy(predictions,y)
                loss.backward()
                self.optimizer.step()
                correct = (predictions.argmax(dim=1) == y).int().sum()
            if self.run_test:
                self.test_loop()
        self.gradients = sum(p.grad.numel() for p in self.cnn.parameters() if p.requires_grad and p.grad is not None)
        return print_out
    def test_loop(self):
        self.cnn.eval()
        for x,y in self.test_dl:
            with torch.no_grad():
                return self.cnn(x)

def main(args):
    trainer = Trainer(args)
    trainer.get_train_data()
    trainer.get_test_data()
    trainer.train_loop()
    print(f"\n----------------------------------------------------------------------------------------------")
    print(f"Number of Workers= {args.num_workers}, Device= {args.device}, Optimizer= {args.optimizer}, Disable Batch Norm= {args.disable_batchnorm}")
    print(f"----------------------------------------------------------------------------------------------")
    print(f"Trainable Parameters: {trainer.trainable_params}, Gradients: {trainer.gradients}")
    print(f"----------------------------------------------------------------------------------------------")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
        
