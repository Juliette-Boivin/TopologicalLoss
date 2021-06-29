# -*- coding: utf-8 -*-
import numpy as np
import higra as hg
import torch as tc
from LossTopo.Attribute import attribute_depth, attribute_saddle_nodes
import LossTopo.ComponentTree





#Define a diagram with num_maximas points superposed in the right-down corner
def NbMaximas2PersDiag(num_maximas, alt_min, alt_max):
    Dgm = np.zeros((num_maximas, 2))
    for i in range(num_maximas):
        Dgm[i] = [alt_min, alt_max]
    return tc.tensor(Dgm)


#Define a persistance diagram calculated on a target image 
def GT2PersDiag(graph, image):
  tree, altitudes = ComponentTree.max_tree(graph, image)
  altitudes_np = altitudes.detach().numpy()

  extrema = hg.attribute_extrema(tree, altitudes_np)
  extrema_indices = np.arange(tree.num_vertices())[extrema]
  extrema_altitudes = altitudes[tc.from_numpy(extrema_indices)]

  depth = attribute_depth(tree, altitudes_np)
  saddle_nodes = tc.from_numpy(attribute_saddle_nodes(tree, depth)[0])
  birth = extrema_altitudes
  death = altitudes[saddle_nodes[extrema_indices]]

  return tc.stack((birth, death), 1) 






