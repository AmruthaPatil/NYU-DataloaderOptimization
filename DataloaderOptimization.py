# Import necessary libraries
import os
import time 
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import warnings

# Ignore the specific UserWarning related to DataLoader worker processes
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=UserWarning, message=".*This DataLoader will create.*")

# Parse command-line arguments
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
parser.add_argument("--profile", action="store_true", default=False)
parser.add_argument("--profiler-name", type = str, default="resnet18")

# Define a basic identity module
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
    
# Define a basic block for the ResNet architecture
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

# Define the ResNet architecture
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, disable_batchnorm=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        # The first convolutional layer with 3 input channels, 64 output channels, 3×3 kernel, with stride=1 and padding=1.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if disable_batchnorm:
            self.bn1 = Identity()
        # first sub-group contains convolutional layer with 64 output channels, 3×3 kernel, stride=1, padding=1
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, disable_batchnorm=disable_batchnorm)
        # second sub-group contains convolutional layer with 128 output channels, 3×3 kernel, stride=2, padding=1
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, disable_batchnorm=disable_batchnorm)
        # third sub-group contains convolutional layer with 256 output channels, 3×3 kenel, stride=2, padding=1
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, disable_batchnorm=disable_batchnorm)
        # forth sub-group contains convolutional layer with 512 output channels, 3×3 kernel, stride=2, padding=1
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, disable_batchnorm=disable_batchnorm)
        # final linear layer is of 10 output classes
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

# Function to create a ResNet18 model
def ResNet18(disable_batchnorm=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], disable_batchnorm=disable_batchnorm)

# Class for training the model
class Trainer:
    def __init__(self, args):
        self.cnn = ResNet18(args.disable_batchnorm)
        self.cnn.to(args.device)
        self.num_workers = args.num_workers
        self.device = args.device
        self.epochs = args.epochs
        self.run_test = args.run_test
        self.profile = args.profile
        self.profiler_name = args.profiler_name
        self.dl_time_elapsed = 0
        self.dl_time_counter = 0
        self.mb_time_elapsed = 0
        self.mb_time_counter = 0
        self.epoch_time_elapsed = 0
        self.epoch_time_counter = 0
        self.last_total_accuracy=0
        
        # Initialize optimizer based on specified type
        if args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.cnn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == "nesterov":
            self.optimizer = torch.optim.SGD(self.cnn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        elif args.optimizer == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.cnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == "adadelta":
            self.optimizer = torch.optim.Adadelta(self.cnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.cnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError(f"Error: optimizer {args.optimizer}")
    
    # Method to get training data
    def get_train_data(self):
        # Random cropping, with size 32×32 and padding 4
        # Random horizontal flipping with a probability 0.5
        # Normalize each image’s RGB channel with mean(0.4914, 0.4822, 0.4465) and variance (0.2023, 0.1994, 0.2010)
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=.5), transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        # CIFAR10 dataset, which contains 50K 32×32 color images
        trainset = torchvision.datasets.CIFAR10(root=args.data_path_train, train=True, download=True, transform=transform_train)
        # minibatch size of 128 and 3 IO processes
        train_dl = torch.utils.data.DataLoader(trainset,batch_size = 128,shuffle = True, drop_last = True,
            num_workers = self.num_workers, pin_memory = self.device == "cuda")
        return train_dl
    
    # Method to get test data
    def get_test_data(self):
        # Normalize each image’s RGB channel with mean(0.4914, 0.4822, 0.4465) and variance (0.2023, 0.1994, 0.2010)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        testset = torchvision.datasets.CIFAR10(root=args.data_path_test, train=False, download=True, transform=transform_test)

        test_dl = torch.utils.data.DataLoader(dataset = testset, batch_size = 128, shuffle=False, drop_last = False,
                num_workers = self.num_workers, pin_memory = self.device == "cuda")
        return test_dl
    
    # Method to perform the training loop
    def train_loop(self, train_dl):
        print_out = ""
        accuracies = []
        losses = []

        # Using PyTorch Profiler to analyze the performance 
        profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profiler_logs/{self.profiler_name}'),
            record_shapes=True, with_stack=True)
        if self.profile:
            profiler.start()

        for i in range(self.epochs):
            dl_times = []
            mb_times = []
            epoch_times = []
            self.cnn.train()
            n_batches = len(train_dl)
            train_dl_i = iter(train_dl)
            
            torch.cuda.synchronize()
            epoch_start = time.perf_counter_ns()
            for j in range(n_batches):
                if self.profile:
                    profiler.step()
                
                # Loading time
                torch.cuda.synchronize()
                dl_start = time.perf_counter_ns()
                x,y = next(train_dl_i)
                
                torch.cuda.synchronize()
                dl_end = time.perf_counter_ns()
                dl_time = dl_end - dl_start
                dl_times.append(dl_time)
                self.dl_time_elapsed += dl_time
                self.dl_time_counter += 1

                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Training (MB) time 
                torch.cuda.synchronize()
                mb_start = time.perf_counter_ns()
                predictions = self.cnn(x)
                loss = F.cross_entropy(predictions,y)
                loss.backward()
                self.optimizer.step()
                       
                torch.cuda.synchronize()
                mb_end = time.perf_counter_ns()
                mb_time = mb_end - mb_start
                mb_times.append(mb_time)
                self.mb_time_elapsed += mb_time
                self.mb_time_counter += 1
                correct = (predictions.argmax(dim=1) == y).int().sum()
                accuracy = correct / len(y)
                accuracies.append(accuracy.item())
                losses.append(loss.item())
                # Print the Batch progress (uncomment the below line when running C1)
                # print(f"\nEpoch {i}: Loss={loss} , Accuracy={accuracy}", flush=True)
            
            torch.cuda.synchronize()
            epoch_end = time.perf_counter_ns()
            epoch_time = epoch_end - epoch_start
            epoch_times.append(epoch_time)
            self.epoch_time_elapsed += epoch_time
            self.epoch_time_counter += 1

            avg_dl_time = (sum(dl_times) / len(dl_times)) / 1e9
            avg_mb_time = (sum(mb_times) / len(mb_times)) / 1e9
            avg_epoch_time = (sum(epoch_times) / self.epochs) / 1e9
            avg_loss = sum(losses) / len(losses)
            avg_accuracy = sum(accuracies) / len(accuracies)

            # Print the Epoch progress
            print_out += f"\nEpoch {i}:"
            print_out += f"\nData-loading Time={sum(dl_times)/1e9}s, Training Time={sum(mb_times)/1e9}s, Total Running Time={epoch_time/1e9}s"
            print_out += f"\nAvg Data-loading Time={avg_dl_time}s, Avg Training Time={avg_mb_time}s, Avg Running Time={avg_epoch_time}s"
            print_out += f"\nAvg Training Loss={avg_loss} , Accuracy={avg_accuracy}"
            print_out += f"\n-------------------------------------------\n"

            if self.run_test:
                self.test_loop()
        if self.profile:
            profiler.stop()
        return print_out
    
    # Method to perform the testing loop
    def test_loop(self):
        self.cnn.eval()
        test_dl = self.get_test_data()
        for x,y in test_dl:
            with torch.no_grad():
                return self.cnn(x)

# Main function to initialize the Trainer class and start training
def main(args):
    trainer = Trainer(args)
    train_dl = trainer.get_train_data()
    trainer_output = trainer.train_loop(train_dl)
    avg_dl_time = (trainer.dl_time_elapsed / trainer.dl_time_counter) / 1e9
    avg_mb_time = (trainer.mb_time_elapsed / trainer.mb_time_counter) / 1e9
    avg_epoch_time = (trainer.epoch_time_elapsed / trainer.epoch_time_counter) /1e9
    
    print(f"\n----------------------------------------------------------------------------------------------")
    print(f"Number of Workers= {args.num_workers}, Device= {args.device}, Optimizer= {args.optimizer}, Disable Batch Norm= {args.disable_batchnorm}")
    print(f"----------------------------------------------------------------------------------------------")
    print(f"Data-loading Time={trainer.dl_time_elapsed/1e9}s, Training Time={trainer.mb_time_elapsed/1e9}s, Total Running Time={trainer.epoch_time_elapsed/1e9}s")
    print(f"Avg Data-loading Time={avg_dl_time}s, Avg Training Time={avg_mb_time}s, Avg Running Time={avg_epoch_time}s")
    print(f"----------------------------------------------------------------------------------------------")
    print(trainer_output)

# Parse arguments and start training
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

        
