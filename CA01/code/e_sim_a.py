from mpi4py import MPI
from random import random
from tqdm import tqdm
import time

IT = 4000000

comm = MPI.COMM_WORLD

def f() -> int:
    s = 0
    i = 0
    while s < 1:
        s+=random()
        i+=1
    
    return i

start = time.time()

e = [f() for _ in tqdm(range(IT))]   
mean = sum(e)/IT
    
end = time.time()

print(mean)
print("runtime: ", end-start)
