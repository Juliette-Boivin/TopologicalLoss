# -*- coding: utf-8 -*-
import numpy as np
import higra as hg
import torch as tc
import math
from LossTopo.Attribute import attribute_depth, attribute_saddle_nodes
#from edist import ted
from LossTopo.Diagrams import NbMaximas2PersDiag
#from geomloss import SamplesLoss
#from edist import ted
from LossTopo import ComponentTree
from gudhi import wasserstein
from LossTopo.ComponentTree import max_tree


def loss_geomloss(graph, image, num_target_maxima, distance, blur):
  """
  Loss that favors the presence of num_target_maxima in the given image. 
  
  :param graph: adjacency pixel graph
  :param image: torch tensor 1d, vertex values of the input graph
  :param saliency_measure: string, how the saliency of maxima is measured, can be "altitude" or "dynamics"
  :param importance_measure: string, how the importance of maxima is measured, can be "altitude", "dynamics", "area", or "volume"
  :param num_target_maxima: int >=0, number of maxima that should be present in the result
  :param margin: float >=0, target altitude fo preserved maxima
  :param p: float >=0, power (see parameter p in loss_ranked_selection)
  :return: a torch scalar
  """

  tree, altitudes = ComponentTree.max_tree(graph, image)
  altitudes_np = altitudes.detach().numpy()

  extrema = hg.attribute_extrema(tree, altitudes_np)
  extrema_indices = np.arange(tree.num_vertices())[extrema]
  extrema_altitudes = altitudes[tc.from_numpy(extrema_indices)]

  depth = attribute_depth(tree, altitudes_np)
  saddle_nodes = tc.from_numpy(attribute_saddle_nodes(tree, depth)[0])
  birth = extrema_altitudes
  death = altitudes[saddle_nodes[extrema_indices]]

  perf_dgm = NbMaximas2PersDiag(num_target_maxima, 0, 1)

  dgm = tc.stack((birth, death), 1) 
  loss = SamplesLoss(loss=distance, p=2, blur=blur)
  L = loss(dgm, perf_dgm)

  return L, birth, death



def img2tree(grph, img, thresh, type):
  if type == 'maxtree':
    tree, altitudes = ComponentTree.max_tree(grph, tc.tensor(img), thresh)
  elif type == 'mintree':
    tree, altitudes = ComponentTree.min_tree(grph, tc.tensor(img), thresh)
    altitudes = 1 - altitudes
  else:
    raise ValueError("Unknown tree type " + str(type) + " possible values are 'maxtree' or 'mintree'")

  return tree, altitudes



def tree2diag(tree, altitudes):
  
  altitudes_np = altitudes.detach().numpy()

  extrema = hg.attribute_extrema(tree, altitudes_np)
  extrema_indices = np.arange(tree.num_vertices())[extrema]
  extrema_altitudes = altitudes[tc.from_numpy(extrema_indices).type(tc.LongTensor)]

  depth = attribute_depth(tree, altitudes_np)
  saddle_nodes = tc.from_numpy(attribute_saddle_nodes(tree, depth)[0]).type(tc.LongTensor)

  # persistence diagram's points built on the maxtree's extremas
  birth = extrema_altitudes
  death = altitudes[saddle_nodes[extrema_indices]]

  return birth, death



def loss_Hu(graph, image, type, num_max=1, thresh=1e10-5, perfect_diagram='None', image_graph_GT='None', optimize_maxtree=False):

  tree, altitudes = img2tree(graph, tc.tensor(image), thresh, type)
  #hg.print_partition_tree(tree)
  
  if optimize_maxtree:
    tree_nodes = [nd for nd in tree.root_to_leaves_iterator()]
    adj = []
    for nd in tree.root_to_leaves_iterator():
      ch = []
      for children in tree.children(nd):
        ch.append(children)
      adj.append(ch)
    loss = ted.standard_ted((tree_nodes, adj), ([0, 1, 2, 3, 4, 5], [[1, 2], [], [3, 4, 5], [], [], []]))
    birth = None
    death = None
    idx_holes_to_fix_or_perfect = None

    return loss, birth, death, idx_holes_to_fix_or_perfect
    

  # gt_pers: get persistence list from the target diagram
  elif image_graph_GT != 'None':

    birth, death = tree2diag(tree, altitudes)
    diag = tc.stack((birth, death), 1)
    lh_pers = birth - death

    (graph_GT, image_GT) = image_graph_GT
    tree_GT, altitudes_GT = img2tree(image_graph_GT[0], image_graph_GT[1], thresh, type)
    birth_GT, death_GT = tree2diag(tree_GT, altitudes_GT)
    diagram_GT = tc.stack((birth_GT, death_GT), 1)

    cost, matchings = wasserstein.wasserstein_distance(diag.detach(), diagram_GT, matching=True, order=30, internal_p=2)
    matchings = matchings[matchings[:, 1] != -1]
    idx_holes_to_fix_or_perfect = list(matchings[:, 0])
    loss = 0
    for i in range(matchings.shape[0]):
       loss += tc.cdist(tc.unsqueeze(diag[matchings[i][0]], 0), tc.unsqueeze(diagram_GT[matchings[i][1]], 0))
    
    idx_holes_to_remove = list(matchings[(matchings[:, 1] == -1)][:, 0])
    
    force_list = tc.zeros((lh_pers.shape[0], 2))
    for ind in idx_holes_to_remove:
      force_list[ind, 0] = lh_pers[ind] / math.sqrt(2.0)
      force_list[ind, 1] = -lh_pers[ind] / math.sqrt(2.0)
    for idx in idx_holes_to_remove:
      loss += force_list[idx, 0] ** 2 + force_list[idx, 1] ** 2
      
    return loss, birth, death, idx_holes_to_fix_or_perfect


  elif perfect_diagram != 'None':
    birth, death = tree2diag(tree, altitudes)
    diag = tc.stack((birth, death), 1)
    lh_pers = birth - death
    
    cost, matchings = wasserstein.wasserstein_distance(diag.detach(), np.array(perfect_diagram), matching=True, order=30, internal_p=2)
    matchings = matchings[matchings[:, 1] != -1]
    idx_holes_to_fix_or_perfect = list(matchings[:, 0])
    loss = 0
    for i in range(matchings.shape[0]):
       loss += tc.cdist(tc.unsqueeze(diag[matchings[i][0]], 0), tc.unsqueeze(tc.tensor(perfect_diagram)[matchings[i][1]], 0))
    
    idx_holes_to_remove = list(matchings[(matchings[:, 1] == -1)][:, 0])
    
    force_list = tc.zeros((lh_pers.shape[0], 2))
    for ind in idx_holes_to_remove:
      force_list[ind, 0] = lh_pers[ind] / math.sqrt(2.0)
      force_list[ind, 1] = -lh_pers[ind] / math.sqrt(2.0)
    for idx in idx_holes_to_remove:
      loss += force_list[idx, 0] ** 2 + force_list[idx, 1] ** 2
      
    return loss, birth, death, idx_holes_to_fix_or_perfect
        
    
  else:
    birth, death = tree2diag(tree, altitudes)
    lh_pers = birth - death
    
    gt_pers = NbMaximas2PersDiag(num_max, 0, 1)[:, 1] - NbMaximas2PersDiag(num_max, 0, 1)[:, 0]
  
    gt_n_holes = gt_pers.shape[0]  # number of holes in gt
    idx_holes_perfect = np.where(lh_pers == lh_pers.max())[0]


    # find top gt_n_holes indices
    if type == 'mintree': # we have to remove the connected component related to the background
      idx_holes_to_fix_or_perfect = np.argpartition(lh_pers.detach().numpy(), -(gt_n_holes + 1))[-(gt_n_holes + 1):(-1)] 
    else:
      idx_holes_to_fix_or_perfect = np.argpartition(lh_pers.detach().numpy(), -gt_n_holes)[-gt_n_holes:]
    
    # the difference is holes to be fixed to perfect
    idx_holes_to_fix = list(set(idx_holes_to_fix_or_perfect) - set(idx_holes_perfect))
    
    # remaining holes are all to be removed
    idx_holes_to_remove = list(set(range(lh_pers.shape[0])) - set(idx_holes_to_fix_or_perfect))
    
    # only select the ones whose persistence is large enough
    # the others are not taken into consideration
    pers_thd = 0.
    idx_valid = np.where(lh_pers > pers_thd)[0]
    idx_holes_to_remove = list(set(idx_holes_to_remove).intersection(set(idx_valid)))

    force_list = tc.zeros((lh_pers.shape[0], 2))
    # push each hole-to-fix to (0,1)
    for ind in idx_holes_to_fix:
      force_list[ind, 0] = 1 - birth[ind]
      force_list[ind, 1] = 0 - death[ind]

    # push each hole-to-remove to the diagram's diagonal
    for ind in idx_holes_to_remove:
      force_list[ind, 0] = lh_pers[ind] / math.sqrt(2.0)
      force_list[ind, 1] = -lh_pers[ind] / math.sqrt(2.0)

    loss = 0.0
    for idx in idx_holes_to_fix:
      loss = loss + force_list[idx, 0] ** 2 + force_list[idx, 1] ** 2
    for idx in idx_holes_to_remove:
      loss = loss + force_list[idx, 0] ** 2 + force_list[idx, 1] ** 2
      
    return loss, birth, death, idx_holes_to_fix_or_perfect


