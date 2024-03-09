from mpi4py import MPI
from random import random
from tqdm import tqdm
import time

IT = 4000000

def f() -> int:
    s = 0
    i = 0
    while s < 1:
        s += random()
        i += 1
    return i

start = time.time()
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
partition = IT // size

e = [f() for _ in tqdm(range(partition))]

mean = sum(e) / partition
mean_sum = comm.reduce(mean, op=MPI.SUM, root=0)

if rank == 0:
    result = mean_sum / size
    end = time.time()
    print(result)
    print("runtime:", end - start)
