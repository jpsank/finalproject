from dataclasses import dataclass, field
from anytree import NodeMixin


class Gene:
    """ Abstract base class for all genes. """


@dataclass
class NodeGene(Gene, NodeMixin):
    """ Base class for CPPN node genes. """
    id: int
    length: float
    name: str = field(default_factory=str)
    parent: 'NodeGene' = None
    children: list = field(default_factory=list)

