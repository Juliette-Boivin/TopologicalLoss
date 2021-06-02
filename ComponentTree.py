# -*- coding: utf-8 -*-
import numpy as np
import torch as tc
import higra as hg
from torch.nn import Module
from torch.autograd import Function


class ComponentTreeFunction(Function):
  @staticmethod
  def forward(ctx, graph, vertex_weights, thresh, tree_type="max", plateau_derivative="full"):
    """
    Construct a component tree of the given vertex weighted graph.

    tree_type must be in ("min", "max", "tos")

    plateau_derivative can be "full" or "single". In the first case, the gradient of an altitude component
    is back-propagated to the vertex weights of the whole plateau (to all proper vertices of the component).
    In the second case, an arbitrary vertex of the plateau is selected and will receive the gradient.

    return: the altitudes of the tree (torch tensor), the tree itself is stored as an attribute of the tensor
    """
    if tree_type == "max":
      # compute the maxtree a first time
      tree, altitudes = hg.component_tree_max_tree(graph, vertex_weights.detach().numpy())

      # identification of nodes whose altitude difference with their parent is below the threshold
      arr = (altitudes[np.arange(tree.num_vertices())] - altitudes[tree.parents()[np.arange(tree.num_vertices())]] < thresh)

      new_tree, node_map = hg.simplify_tree(tree, arr, process_leaves=False)
      new_altitudes = np.array([altitudes[nm] for nm in node_map])

    elif tree_type == "min":
      tree, altitudes = hg.component_tree_min_tree(graph, vertex_weights.detach().numpy())
      arr = (np.abs(altitudes[np.arange(tree.num_vertices())] - altitudes[tree.parents()[np.arange(tree.num_vertices())]]) < thresh)
      new_tree, node_map = hg.simplify_tree(tree, arr, process_leaves=False)
      new_altitudes = np.array([altitudes[nm] for nm in node_map])
      
    elif tree_type == "tos":
      tree, altitudes = hg.component_tree_tree_of_shapes_image2d(vertex_weights.detach().numpy())
    else:
      raise ValueError("Unknown tree type " + str(tree_type))

    if plateau_derivative == "full":
      plateau_derivative = True
    elif plateau_derivative == "single":
      plateau_derivative = False
    else:
      raise ValueError("Unknown plateau derivative type " + str(plateau_derivative))
    ctx.saved = (new_tree, graph, plateau_derivative)
    altitudes = tc.from_numpy(new_altitudes).clone().requires_grad_(True)
    # torch function can only return tensors, so we hide the tree as a an attribute of altitudes
    altitudes.tree = new_tree
    return altitudes

  @staticmethod
  def backward(ctx, grad_output):
    tree, graph, plateau_derivative = ctx.saved
    if plateau_derivative:
      grad_in = grad_output[tree.parents()[:tree.num_leaves()]]
    else:
      leaf_parents = tree.parents()[:tree.num_leaves()]
      _, indices = np.unique(leaf_parents, return_index=True)
      grad_in = tc.zeros((tree.num_leaves(),), dtype=grad_output.dtype)
      grad_in[indices] = grad_output[leaf_parents[indices]]
    return None, hg.delinearize_vertex_weights(grad_in, graph), None, None

class ComponentTree(Module):
    def __init__(self, tree_type):
        super().__init__()
        tree_types = ("max", "min", "tos")
        if tree_type not in tree_types:
          raise ValueError("Unknown tree type " + str(tree_type) + " possible values are " + " ".join(tree_types))

        self.tree_type = tree_type

    def forward(self, graph, vertex_weights, thresh):
        altitudes = ComponentTreeFunction.apply(graph, vertex_weights, thresh, self.tree_type)
        return altitudes.tree, altitudes

max_tree = ComponentTree("max")
min_tree = ComponentTree("min")
tos_tree = ComponentTree("tos")