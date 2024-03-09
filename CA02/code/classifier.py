import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import FashionMNIST
from torch.optim import Adam
from torch.cuda import max_memory_allocated
import time

device = "cuda:0"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 2


def load_data():
    train_set = FashionMNIST(
        root="data",
        train=True,
        download=True, 
        transform=ToTensor()
    )
    
    train_loader = DataLoader(
        dataset=train_set,
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
    
    test_loader = DataLoader(
        dataset=test_set, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        persistent_workers=True,
        num_workers=1, 
        pin_memory=True
    )
    
    return train_loader, test_loader

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

if __name__ == "__main__":
    train_loader, test_loader = load_data()
    model = Fasion_MNIST_classifier()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []
    predictions_list = []
    labels_list = []

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            cuda_mem = max_memory_allocated(device=device)
            count += 1
            if not (count % 50):
                total = 0
                correct = 0
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    labels_list.append(labels)
                    outputs = model(images)
                    predictions = torch.max(outputs, 1)[1].to(device)
                    predictions_list.append(predictions)
                    correct += (predictions == labels).sum()
                    total += len(labels)
                accuracy = correct * 100 / total
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)
            if not (count % 500):
                print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))
                
    end_time = time.time()
    print(f"Training time(s): {end_time - start_time}")
    print(f"Cuda Memory Usage: {cuda_mem / (1024 ** 2)} MB")