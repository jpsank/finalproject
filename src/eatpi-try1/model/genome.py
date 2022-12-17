from typing import *
import random
from dataclasses import dataclass, field
import phylotreelib as pt

from neat.model.genes import NodeGene, ConnGene


@dataclass
class Genome:
    id: int
    tree: pt.Tree
    
