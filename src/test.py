from Bio import Phylo
import phylotreelib as pt
import random
from dataclasses import dataclass, field


# Load the tree
# tree = Phylo.read('data/mammal.tre', 'newick')
treefile = pt.Newicktreefile('data/mammal.tre')
tree = treefile.readtree()

# Print the tree
print(tree)

# Do subtree pruning and regrafting with random nodes
node1 = random.choice(list(tree.nodes))
node2 = random.choice(list(tree.nodes - set([node1])))
tree.spr(node1, node2)

# Print the tree
print(tree)
