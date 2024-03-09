#!/bin/bash

#SBATCH --job-name=question_1_a
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2

mpirun -n 4 python3 k_means_c.py

