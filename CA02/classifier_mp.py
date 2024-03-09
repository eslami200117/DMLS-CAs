import os
import time
import torch
import datetime
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import FashionMNIST
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 4

def setup(rank, world_size, master_port, backend, timeout):
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, timeout=timeout)

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


class Fasion_MNIST_classifier(nn.Module):
    def __init__(self):
        super(Fasion_MNIST_classifier, self).__init__()
        self.c1 = nn.Sequential(OrderedDict([
            ('Conv1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), padding=2)),
            ('Relu1', nn.ReLU()),
            ('Pool1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))
        self.c2 = nn.Sequential(OrderedDict([
            ('Conv2', nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))),
            ('Relu2', nn.ReLU()),
            ('Pool2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))  # Output:16*5*5
        ]))
        self.c3 = nn.Sequential(OrderedDict([
            ('FullCon3', nn.Linear(in_features=400, out_features=120)),
            ('Relu3', nn.ReLU()),
        ]))
        self.c4 = nn.Sequential(OrderedDict([
            ('FullCon4', nn.Linear(in_features=120, out_features=84)),
            ('Relu4', nn.ReLU()),
        ]))
        self.c5 = nn.Sequential(OrderedDict([
            ('FullCon5', nn.Linear(in_features=84, out_features=10)),
            ('Sig5', nn.LogSoftmax(dim=-1)),
        ]))

    def forward(self, img):
        output = self.c1(img)
        output = self.c2(output)

        output = output.view(-1, 16*5*5)
        output = self.c3(output)
        output = self.c4(output)
        output = self.c5(output)

        return output

def load_data(rank, world_size):
    train_set = FashionMNIST(
        root="data",
        train=True,
        download=True, 
        transform=ToTensor()
    )
    
    train_sampler = DistributedSampler(
        train_set, 
        num_replicas=world_size, 
        rank=rank
    )
    
    train_loader = DataLoader(
        dataset=train_set,
        sampler=train_sampler, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        persistent_workers=True,
        num_workers=1, 
        pin_memory=True
    )
    
    test_set = FashionMNIST(
        root="data",
        download=True, 
        train=False, 
        transform=ToTensor()
    )
    
    test_sampler = DistributedSampler(
        test_set, 
        num_replicas=world_size, 
        rank=rank
    )
    
    test_loader = DataLoader(
        dataset=test_set, 
        sampler=test_sampler, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        persistent_workers=True,
        num_workers=1, 
        pin_memory=True
    )
    return train_loader, test_loader

def train(rank, world_size, master_port, backend, timeout):
    setup(rank, world_size, master_port, backend, timeout)
    torch.cuda.set_device(rank)
    train_loader, test_loader = load_data(rank, world_size)
    model = Fasion_MNIST_classifier().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=LEARNING_RATE)
    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []
    predictions_list = []
    labels_list = []
    start_time = time.time()
    for _ in range(NUM_EPOCHS):
        for images, labels in train_loader:
            images, labels = images.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = ddp_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            cuda_mem = torch.cuda.max_memory_allocated(device=rank)
            count += 1
            if not (count % 50):
                total = 0
                correct = 0
                for images, labels in test_loader:
                    images, labels = images.to(rank), labels.to(rank)
                    labels_list.append(labels)
                    outputs = ddp_model(images)
                    predictions = torch.max(outputs, 1)[1].to(rank)
                    predictions_list.append(predictions)
                    correct += (predictions == labels).sum()
                    total += len(labels)
                accuracy = correct * 100 / total
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)
            if not (count % 500):
                print("Rank: {}, Iteration: {}, Loss: {}, Accuracy: {}%".format(rank, count, loss.data, accuracy))
    end_time = time.time()
    print("Rank: {}, Training Time: {}".format(rank, end_time - start_time))
    print("Rank: {}, Max Memory Allocated: {} MB".format(rank, cuda_mem / (1024 ** 2)))


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    master_port = find_free_port()
    backend = 'nccl'
    timeout = datetime.timedelta(seconds=10)
    start_time = time.time()
    mp.spawn(train, nprocs=world_size, args=(world_size, master_port, backend, timeout), join=True)
    end_time = time.time()
    print("Total time: {}".format(end_time - start_time))