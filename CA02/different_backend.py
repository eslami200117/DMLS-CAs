import os
import time
import torch
import datetime
import torch.nn as nn
import torch.nn.functional as F
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
SAVE_EVERY = 2
ACCUMULATE_ITER = 2

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
      
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._loading_snapshot(snapshot_path)
        self.model = DDP(self.model, device_ids=[self.gpu_id])


    def _loading_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot_path["EPOCHS_RUN"]
        
    def _run_batch(self, source, targets, is_accumulate):
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss = loss/ACCUMULATE_ITER
        loss.backward()
        if is_accumulate:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _run_epoch(self, epoch):
        accumulate = 0
        b_sz = len(next(iter(self.train_data))[0])
        # print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            accumulate += 1
            self._run_batch(source, targets, (accumulate%ACCUMULATE_ITER == 0))

    def _save_snapshot(self, epoch, accuracy):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        snapshot["ACCURACY"] = accuracy
        PATH = f"snapshot_epoch_{epoch}.pt"
        torch.save(snapshot, PATH)
        # print(f"Epoch {epoch} | Accuracy: {accuracy} | Training checkpoint saved at {PATH}")

    def train_and_evaluate(self, max_epochs: int):
        start_time = time.time()
        accuracy = 0
        loss = 0
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            accuracy, loss = self._evaluate(self.test_data)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch, accuracy)
        end_time = time.time()
        print(f"[GPU{self.gpu_id}] Evaluation Loss: {loss:.4f} | Accuracy: {accuracy * 100:.2f}% ")
        # print(f"[GPU{self.gpu_id}] Traning Time: {end_time-start_time}")

    def _evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        iterations = 0
        accuracy = 0
        average_loss = 0
        
        with torch.no_grad():
            for source, targets in data_loader:
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                output = self.model(source)
                loss = F.cross_entropy(output, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                iterations += 1
                
        average_loss = total_loss / iterations
        accuracy = correct / total
        # print(f"[GPU{self.gpu_id}] Evaluation Loss: {average_loss:.4f} | Accuracy: {accuracy * 100:.2f}% | Iterations: {iterations}")
        
        return accuracy, average_loss

    
    
def load_data():
    train_set = FashionMNIST(
        root="data",
        train=True,
        download=True, 
        transform=ToTensor()
    )
    
    
    train_loader = DataLoader(
        dataset=train_set,
        sampler=DistributedSampler(train_set),
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
        sampler=DistributedSampler(test_set),
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        persistent_workers=True,
        num_workers=1, 
        pin_memory=True
    )
    return train_loader, test_loader

def load_train_objs():
    model =  Fasion_MNIST_classifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return model, optimizer
    
def main(save_every: int, total_epochs: int, backend: str, snapshot_path: str = "snapshot.pt"):
    dist.init_process_group(backend=backend)
    model, optimizer = load_train_objs()
    train_data ,test_data = load_data()
    trainer = Trainer(model, train_data, test_data, optimizer, save_every, snapshot_path)
    trainer.train_and_evaluate(total_epochs)
    dist.destroy_process_group()

if __name__ == "__main__":
    import sys
    BATCH_SIZE = int(sys.argv[1])
    backend = sys.argv[2]
    timeout = datetime.timedelta(seconds=10)
    start_time = time.time()
    rank = int(os.environ["LOCAL_RANK"])
    if not rank:
        print(f"batch size is: {BATCH_SIZE}")
        print(f"backend is: {backend}")
    main(SAVE_EVERY, NUM_EPOCHS, backend)
    cuda_mem = torch.cuda.max_memory_allocated(device=rank)
    end_time = time.time()
    print(f"[GPU{rank}] Max Memory Allocated: {cuda_mem / (1024 ** 2):.2f} MB")
    print(f"[GPU{rank}] Total time: {end_time - start_time:.2f}")
    