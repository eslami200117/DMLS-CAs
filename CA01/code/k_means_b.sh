#!/bin/bash

#SBATCH --job-name=question_1_a
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2

mpirun -n 2 python3 k_means_b.py

