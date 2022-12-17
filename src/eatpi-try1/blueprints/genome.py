"""
Defines blueprint for configuration and functions applied to genomes, plus counter for genome id's.
Holds methods for creating, mutating, copying, crossover, and calculating genomic distance.
"""

from itertools import count
from dataclasses import dataclass, field
import random
from typing import *
import phylotreelib as pt
import itertools

from neat.model import *
from neat.blueprints.primitives import Blueprint
from neat.blueprints.genes import GeneBP, NodeBP, ConnBP


# --------------- GENOME CONFIGURABLES ---------------

@dataclass
class GenomeBP(Blueprint):
    """ Contains genome configuration and counters for a simulation """

    __constructor__ = Genome
    __primary_keys__ = ("id",)

    # Mutation probabilities
    mutate_spr_prob: float

    # Genome compatibility options
    compatibility_disjoint_coefficient: float # c2, takes the place of both c1 and c2
    compatibility_weight_coefficient: float  # c3

    # Genome ID counter
    __id_counter: count = field(default_factory=count)

    def create(self, **kwargs) -> Genome:
        """ Create a new genome with the given attributes. """
        if "id" not in kwargs:
            kwargs["id"] = next(self.__id_counter)
        if "tree" not in kwargs:
            kwargs["tree"] = pt.Tree.randtree()  # TODO: specify list of tree leaf names
        return Genome(**kwargs)

    def __mutate_spr(self, genome: Genome):
        """ Mutate a genome by performing subtree pruning and reconnection. """
        subtrees = [subtree for subtree in genome.tree if len(subtree) > 1]
        if not subtrees:
            return 0
        subtree_node = random.choice(subtrees)
        regraft_node = random.choice(subtrees)
        genome.tree.spr(subtree_node, regraft_node)
        return 1

    def mutate(self, genome: Genome):
        """ Mutate a genome. """
        mut_func = random.choices(
            population=(self.__mutate_spr,),
            weights=(self.mutate_spr_prob,),
            k=1,
        )[0]
        mut_func(genome)
    
    def copy(self, genome: Genome) -> Genome:
        """ Copy a genome. """
        return Genome(
            id=genome.id,
            tree=genome.tree.copy_treeobject(),
        )

    def crossover(self, a: Genome, b: Genome) -> Genome:
        """
        Create a new genome by crossover from two parent genomes.
        Pre-condition: a is fitter than b
        """
        tree = a.tree.copy_treeobject()
        subtree_a = random.choice([subtree for subtree in tree if len(subtree) > 1])
        for leaf in self.remote_children(subtree_a):
            self.remove_leaf(leaf)

        subtree_b = random.choice([subtree for subtree in b.tree if len(subtree) > 1])
        tree.graft(subtree_b, subtree_a)
        return self.create(tree=tree)

    def __compare_genes(self, a: 'dict[Any, Gene]', b: 'dict[Any, Gene]', gene_bp: GeneBP):
        """ Compare two maps of the same type of gene. """
        homologous_distance, disjoint = 0.0, 0
        if a or b:
            for i in b.keys():
                if i not in a:
                    disjoint += 1

            for i, gene1 in a.items():
                if (gene2 := b.get(i)) is None:
                    disjoint += 1
                else:
                    # Homologous genes compute their own distance value.
                    homologous_distance += gene_bp.distance(gene1, gene2)
        return homologous_distance, disjoint

    def distance(self, a: Genome, b: Genome) -> float:
        """
        Returns the genetic distance between two genomes. This distance value
        is used to compute genome compatibility for speciation.
        """

        total_distance = 0
        for a_genes, b_genes, field in ( (a.nodes, b.nodes, self.node), (a.conns, b.conns, self.conn) ):
            if a_genes or b_genes:
                dist, disjoint = self.__compare_genes(a_genes, b_genes, field)
                dist += disjoint * self.compatibility_disjoint_coefficient
                total_distance += dist / max(len(a_genes), len(b_genes))
        
        return total_distance
