#!/bin/bash

#SBATCH --job-name=question_1_a
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

python3 k_means_a.py
