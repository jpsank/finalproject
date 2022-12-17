#!/bin/bash

#SBATCH --job-name=siph_iqtree
#SBATCH --time=30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

module load Seq-Gen
module load IQ-TREE/1.6.12

# the following lines run the analyses, and can be executed on your own computer if the sopftware is installed or on the cluster

# JC - equal nucleotid frequencies with C=G=T=A=0.25 and all relative rates = 1
seq-gen -mGTR -f0.25,0.25,0.25,0.25 -r1,1,1,1,1,1 -s0.001 -l1000 -of < mammal.tre > mammal_jc.fasta
iqtree -s mammal_jc.fasta -nt AUTO
# HKY - unequal nucleotide frequencies with C=G=0.205 and A=T=0.295; rates for transitions=2, transversions=0.5 
seq-gen -mGTR -f0.295,0.205,0.205,0.295 -r0.5,2,0.5,0.5,2,0.5 -s0.001 -l1000 -of < mammal.tre > mammal_hky.fasta
iqtree -s mammal_hky.fasta -nt AUTO
# GTR - 
seq-gen -mGTR -f0.3841,0.1182,0.1646,0.333 -r1.0051,3.5164,2.2921,0.9079,5.9252,1.0000 -s0.001 -l1000 -of < mammal.tre > mammal_gtr.fasta
iqtree -s mammal_gtr.fasta -nt AUTO
# GTR - 
seq-gen -mGTR -f0.3841,0.1182,0.1646,0.333 -r1.0051,3.5164,2.2921,0.9079,5.9252,1.0000 -i0.1363 -a0.9978 -g4  -s0.001 -l1000 -of < mammal.tre > mammal_gtr_ig.fasta
iqtree -s mammal_gtr_ig.fasta -nt AUTO

