# Phylogenetic Biology - Final Project

# NEAT-based Phylogenetic Inference

## Lit Review

Gaphyl, GARLI (Genetic Algorithm for Rapid Likelihood Inference) and GARLI 2.0, Recursive-Iterative-DCM3, and GAML (genetic algorithm for maximum likelihood).
“Harnessing machine learning to guide phylogenetic-tree search algorithms”, multi-objective optimization + evolutionary algorithms (MOEA), paper describing a "new phylogenetic protocol."

## Introduction and Goals

Research Question: To what extent are genetic algorithms useful and effective for phylogenetic tree search, and how do they compare to standard inference methods?

Methods: Python using `biopython` library and/or `phylotreelib` to process trees and perform subtree pruning and regrafting operations. Expecting to use SPR moves as mutations in the final genetic algorithm, although other methods such as swapping two nodes/clades may also be implemented as the project evolves. Rather than completely excluded, likelihood maximization could be incorporated into the genetic algorithm, i.e. as a way to bias mutations towards beneficial topological changes (e.g. by swapping branches with lowest bootstrap scores). The genetic algorithm will be written in Python as well, using Newick format as genomes and implementing strategies from the field of evolutionary computation to solve common pitfalls of genetic algorithms (e.g. premature feature loss resolved by elitism, local optima overcome by dividing population into species). Specifically, some elements of the NEAT algorithm (Neuroevolution of Augmenting Topologies), such as the method for keeping track of specific branch nodes' origins as a proxy for homology when performing genetic crossover.
Standard phylogenetic inference methods will be tested using the `biopython` library and/or IQ-TREE command line tool. The results will be compared by running likelihood evaluations on the final trees generated.

Data will consist of aligned sequences; the Harnessing Machine Learning paper includes a few good datasets for testing phylogenetic inference methods, so those, including the algae protein-coding genes dataset, will be used here. No specific dataset is necessary for testing the new method; instead, several representational datasets like the protein-coding dataset, will be employed to evaluate different phylogenetic inference methods.

## Methods

The tools I used were... See analysis files at (links to analysis files).

## Results

The tree in Figure 1...

## Discussion

These results indicate...

The biggest difficulty in implementing these analyses was...

If I did these analyses again, I would...

## References

