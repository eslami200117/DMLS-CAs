#!/bin/bash

#SBATCH --job-name=question_1_a
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

mpirun -n 1 python3 e_sim_a.py 
