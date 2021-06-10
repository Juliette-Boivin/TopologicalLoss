# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from Loss import loss_Hu
from Optimizer import Optimizer
import torch as tc
import os
import imageio


def experiment(graph, img, nb_iter, loss='minmax', nb_holes=2, nb_cc=4, thresh_min=1e10-5, thresh_max=1e10-5, perf_dgm='None', img_grph_GT='None', optimize_maxtree=False,
               color_min='blue', color_max='red', dataloss = False, plot_loss=True, result=True, diag_beg=True, 
               diag_end=True, nb_leaves=True, save_fig=False, save_diagrams=False, save_results=False):
  """
  Experiments with loss_Hu

  graph: the graph associated to the input image
  img: the image to optimize
  nb_iter: number of loops in the Optimizer
  loss: 'min', 'max' or 'minmax'. Use of the mintree, maxtree or a combination of the two.
  nb_holes: number of holes to enforce in the image
  nb_cc: number of connected components to enforce in the image
  thresh_min: threshold of altitudes difference used to cut nodes in the mintree
  thresh_max: threshold of altitudes difference used to cut nodes in the maxtree
  color_min: color of the diagram's points mapped to the mintree's leaves
  color_max: color of the diagram's points mapped to the maxtree's leaves
  plot_loss: if True, the loss function is displayed
  result: if True, the optimized image is displayed
  diag_beg: if True, the diagram at iteration zero is displayed
  diag_end: if True, the diagram at the last iteration is displayed
  nb_leaves: if True, the evolution of the number of nodes is displayed
  save_fig: if True, the figure is saved in the path filled out.
  save_diagrams: if True, slices of the diagram optimized are saved and a GIF is created.
  save_results: if True, slices of the image optimized are saved and a GIF is created.

  """

  L = [plot_loss, result, diag_beg, diag_end, nb_leaves]
  nb_plot = L.count(True)

  plt.figure(figsize=(25, 6))

  if loss == 'min':
    Loss = lambda image: loss_Hu(graph, image, thresh=thresh_min, num_max=nb_holes, type='mintree', perfect_diagram=perf_dgm, image_graph_GT=img_grph_GT)
  elif loss == 'max':
    Loss = lambda image: loss_Hu(graph, image, thresh=thresh_max, num_max=nb_cc, type='maxtree', perfect_diagram=perf_dgm, image_graph_GT=img_grph_GT)
  elif loss == 'minmax':
    lossmax = lambda image: loss_Hu(graph, image, thresh=thresh_max, num_max=nb_cc, type='maxtree', perfect_diagram=perf_dgm, image_graph_GT=img_grph_GT)
    lossmin = lambda image: loss_Hu(graph, image, thresh=thresh_min, num_max=nb_holes, type='mintree', perfect_diagram=perf_dgm, image_graph_GT=img_grph_GT)
    Loss = (lossmax, lossmin)
  else:
    raise ValueError("Unknown type " + str(loss) + ", possible values are 'min', 'max' or 'minmax'")

  opt = Optimizer(Loss, lr=0.001)
  opt.fit(tc.from_numpy(img.copy()), iter=nb_iter, min_lr=None)

  k = 1
  if plot_loss:
    # plot the loss evolution
    plt.subplot(1, nb_plot, k)
    opt.show_history()
    plt.title('loss')
    k += 1

  if result:
    # plot the optimized image
    plt.subplot(1, nb_plot, k)
    res = opt.best.detach().numpy()
    plt.imshow(res, cmap='gray')
    plt.title('image result')
    k += 1

  if diag_beg:
    # plot the first diagram
    plt.subplot(1, nb_plot, k)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    if loss == 'min':
      opt.show_dgm(0, color_min)
    elif loss == 'max':
      opt.show_dgm(0, color_max)
    else:
      color = (color_min, color_max)
      opt.show_dgm(0, color)
    plt.title('persistence diagram at iteration 0')
    k += 1

  if diag_end:
    # plot the last diagram
    plt.subplot(1, nb_plot, k)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    if loss == 'min':
      opt.show_dgm(-1, color_min)
    elif loss == 'max':
      opt.show_dgm(-1, color_max)
    else:
      color = (color_min, color_max)
      opt.show_dgm(-1, color)
    plt.title('persistence diagram at the last iteration')
    k += 1

  if nb_leaves:
    #plot the number of leaves' evolution
    plt.subplot(1, nb_plot, k)
    Nb_leaves = []
    for i in range(len(opt.b)):
      Nb_leaves.append(len(opt.b[i]))
    plt.plot(Nb_leaves)
    plt.title('number of leaves')
    
  plt.show()

  if save_fig:
    # save the figure in the path filled out
    Path = input('path')
    plt.savefig(Path)

  if save_diagrams:
    # save slices of diagram evolution to make a GIF
    filenames = []
    if loss == 'max':
      gif_dgm = input('Enter path to store diagram slices: ') + '/' + str(loss) + '/' + str(nb_cc) 
      if not os.path.exists(gif_dgm):
          os.makedirs(gif_dgm)
      for epoch in range(0, nb_iter, 10):
        opt.show_dgm_for_gif(epoch, filenames, gif_dgm, perf_dgm=perf_dgm, img_grph_GT=img_grph_GT, color='red', i=nb_cc)
    elif loss == 'min':
      gif_dgm = input('Enter path to store diagram slices: ') + '/' + str(loss) + '/' + str(nb_holes)
      if not os.path.exists(gif_dgm):
          os.makedirs(gif_dgm)
      for epoch in range(0, nb_iter, 10):
        opt.show_dgm_for_gif(epoch, filenames, gif_dgm, perf_dgm=perf_dgm, img_grph_GT=img_grph_GT, color='red', i=nb_holes)
    else:
      gif_dgm = input('Enter path to store diagram slices: ') + '/' + str(loss) + '/' + str(nb_cc) + '-' + str(nb_holes)
      if not os.path.exists(gif_dgm):
          os.makedirs(gif_dgm)
      for epoch in range(0, nb_iter, 10):
        opt.show_dgm_for_gif(epoch, filenames, gif_dgm, perf_dgm=perf_dgm, img_grph_GT=img_grph_GT, color='black', color2='red', i=nb_cc, j=nb_holes)
    
    with imageio.get_writer(gif_dgm + '/gif.gif', mode='I', duration=0.4) as writer:
      for filename in filenames:
        im = imageio.imread(gif_dgm + '/' + filename)
        writer.append_data(im)


  if save_results:
    filenames = []
    if loss == 'max':
      gif_res = input('Enter path to store result slices: ') + '/' + str(loss) + '/' + str(nb_cc)
      if not os.path.exists(gif_res):
          os.makedirs(gif_res)
      for epoch in range(0, len(opt.history_best), 10):
        opt.show_res_for_gif(epoch, filenames, gif_res)
    elif loss == 'min':
      gif_res = input('Enter path to store result slices: ') + '/' + str(loss) + '/' + str(nb_holes)
      if not os.path.exists(gif_res):
          os.makedirs(gif_res)
      for epoch in range(0, len(opt.history_best), 10):
        opt.show_res_for_gif(epoch, filenames, gif_res)
    else:
      gif_res = input('Enter path to store result slices: ') + '/' + str(loss) + '/' + str(nb_cc) + '-' + str(nb_holes)
      if not os.path.exists(gif_res):
          os.makedirs(gif_res)
      for epoch in range(0, len(opt.history_best), 10):
        opt.show_res_for_gif2(epoch, filenames, gif_res)

    with imageio.get_writer(gif_res + '/gif.gif', mode='I', duration=0.4) as writer:
      for filename in filenames:
        im = imageio.imread(gif_res + '/' + filename)
        writer.append_data(im)