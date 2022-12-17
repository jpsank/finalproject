from typing import *
import random
from dataclasses import dataclass, field
from anytree import Node

from neat.model.genes import NodeGene


@dataclass
class Genome:
    """ Base class for genomes. """
    id: int
    nodes: 'dict[int, NodeGene]'
