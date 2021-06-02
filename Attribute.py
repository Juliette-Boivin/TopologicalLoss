# -*- coding: utf-8 -*-
import numpy as np
import higra as hg


def attribute_depth(tree, altitudes):
  """
  Compute the depth of any node of the tree which is equal to the largest altitude 
  in the subtree rooted in the current node. 

  :param tree: input tree
  :param altitudes: np array (1d), altitudes of the input tree nodes
  :return: np array (1d), depth of the tree nodes
  """
  return hg.accumulate_sequential(tree, altitudes[:tree.num_leaves()], hg.Accumulators.max)

def attribute_saddle_nodes(tree, attribute):
  """
  Let n be a node and let an be an ancestor of n. The node an has a single child node that contains n denoted by ch(an -> n). 
  The saddle and base nodes associated to a node n for the given attribute values are respectively the closest ancestor an  
  of n and the node ch(an -> n) such that there exists a child c of an with attr(ch(an -> n)) < attr(c). 

  :param tree: input tree
  :param attribute: np array (1d), attribute of the input tree nodes
  :return: (np array, np array), saddle and base nodes of the input tree nodes for the given attribute
  """

  max_child_index = hg.accumulate_parallel(tree, attribute, hg.Accumulators.argmax)
  child_index = hg.attribute_child_number(tree)
  main_branch = child_index == max_child_index[tree.parents()]
  main_branch[:tree.num_leaves()] = True
  #print(main_branch)

  saddle_nodes = hg.propagate_sequential(tree, np.arange(tree.num_vertices())[tree.parents()], main_branch)
  base_nodes = hg.propagate_sequential(tree, np.arange(tree.num_vertices()), main_branch)
  return saddle_nodes, base_nodes