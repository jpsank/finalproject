#!/bin/bash
#SBATCH --job-name=seq-gen
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH -p general

module load Seq-Gen

# JC69 is a special case of HKY and can be obtained by setting the nucleotide frequencies equal and the transition transversion ratio to 0.5.
seq-gen -mHKY -t0.5 -s0.001 -l40 -of < mammal.tre > mammal_jc.fa
