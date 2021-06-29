# -*- coding: utf-8 -*-
import torch as tc
import matplotlib.pyplot as plt
import numpy as np
import ot
from Diagrams import NbMaximas2PersDiag

class Optimizer:
  def __init__(self, loss, lr, optimizer="SGD"):
    """
    Create an Optimizer utility object

    loss: function that takes a single torch tensor which support requires_grad = True and returns a torch scalar
    lr: learning rate
    optimizer: "adam" or "sgd"
    """

    if type(loss) == tuple:
      self.loss_function = None
      self.loss_function1 = loss[0]
      self.loss_function2 = loss[1]
    else:
      self.loss_function = loss

    self.history = []
    self.optimizer = optimizer
    self.lr = lr
    self.best = None
    self.best_loss = float("inf")

    #birth and death points of loss_function1
    self.b = []
    self.d = []

    #birth and death points of loss_function2
    self.b2 = []
    self.d2 = []

    #indices of points to fix 
    self.id = []
    self.id2 = []

    self.history_best = []

    #points where gradient > zero
    self.moving_points = []

    #points where gradient < zero
    self.moving_points2 = []
    self.datashape = None

  def fit(self, data, iter=1000, debug=False, min_lr=1e-6, coeff=2):
    """
    Fit the given data

    data: torch tensor, input data
    iter: int, maximum number of iterations
    debug: int, if > 0, print current loss value and learning rate every debug iterations
    min_lr: float, minimum learning rate (an LR scheduler is used), if None, no LR scheduler is used 
    """
    data = data.clone().requires_grad_(True)
    self.datashape = data.shape

    if self.optimizer == "adam":
      optimizer = tc.optim.Adam([data], lr=self.lr, amsgrad=True)
    else:
      optimizer = tc.optim.SGD([data], lr=self.lr)

    if min_lr:
      lr_scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=100)

    for t in range(int(iter)):

      #discretization of the data at each iteration
      data_ = (data * 255)
      data_.type(tc.ByteTensor)
      data_.type(tc.DoubleTensor)
      data_ = data_ / 255
      data_.detach()

      optimizer.zero_grad()
      if self.loss_function == None:
        loss1, birth, death, idx = self.loss_function1(tc.relu(data_))
        loss2, birth2, death2, idx2 = self.loss_function2(tc.relu(data_))

        # backpropagate only on the highest of the two losses
        if loss1 > loss2:
          loss = loss1
        else:
          loss = loss2

      else:
        loss, birth, death, idx = self.loss_function(tc.relu(data_))

      #Maybe add a coeff on one of the two losses?
      #loss = loss1 + coeff*loss2

      loss.backward()
      optimizer.step() 
      loss_value = loss.item()
      
      self.history.append(loss_value) 

      self.b.append(birth.detach().numpy())
      self.d.append(death.detach().numpy())
      self.id.append(idx)

      if self.loss_function == None:
        self.b2.append(birth2.detach().numpy())
        self.d2.append(death2.detach().numpy())
        self.id2.append(idx2)
      
      
      if loss_value < self.best_loss:
        self.best_loss = loss_value
        self.best = tc.relu(data_).clone()

      self.history_best.append(self.best)
      #self.moving_points.append(np.where(data_.grad > 0))
      #self.moving_points2.append(np.where(data_.grad < 0))
        
      if min_lr:
        lr_scheduler.step(loss_value)
        if optimizer.param_groups[0]['lr'] <= min_lr:
          break

      if debug and t % debug == 0:
        print("Iteration {}: Loss: {:.4f}, LR: {}".format(t, loss_value, optimizer.param_groups[0]['lr']))
    return self.best

  def show_dgm(self, nb, color):
    """Plot the persistence diagram at the iteration nb"""

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    if self.loss_function == None:
      plot1 = plt.scatter(self.b[nb], self.d[nb], color=color[0])
      plot2 = plt.scatter(self.b2[nb], self.d2[nb], color=color[1])
      plt.legend([color[0], color[1]], [str(self.loss_function1), str(self.loss_function2)])
    else:
      plt.scatter(self.b[nb], self.d[nb], color=color[0])

  def show_res_for_gif(self, nb, lst, path, grad_color=False):
    """Save slices in 'path/nb.png' to make a GIF"""
    """if grad_color = True, a green/red point is plotted on the pixels with positive/negative gradient"""

    plt.figure(figsize=(6, 6))
    plt.imshow(self.history_best[nb].detach().numpy(), cmap='gray', interpolation="bicubic")
    if grad_color:
      plt.scatter(self.moving_points[nb][0], self.moving_points[nb][1], color="green")

    filename = str(nb) + '.png'
    lst.append(filename)
    plt.savefig(path + '/' + filename)
    plt.close()


  def show_dgm_for_gif(self, nb, lst, path, i, j=None, color='black', color2='lemon'):

    plt.figure(figsize=(6, 6))
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    #create cost matrix to draw lines between the points of the two diagrams
    C = np.zeros((len(self.b[0]), i))
    z = 0
    for idx in self.id[nb]:
      C[idx, z] = 1
      z += 1

    #Points corresponding to the image, target points and links between them for the first loss
    plt.scatter(self.b[nb], self.d[nb], color=color, s=9)
    ot.plot.plot2D_samples_mat(np.stack((self.b[nb], self.d[nb]), 1), NbMaximas2PersDiag(i, 1, 0), C, color=color)
    plt.scatter(NbMaximas2PersDiag(i, 0, 1)[:, 1], NbMaximas2PersDiag(i, 0, 1)[:, 0], color='red', s=9)


    #Points corresponding to the image, target points and links between them for the second loss
    if self.loss_function == None:

      C2 = np.zeros((len(self.b2[0]), j))
      z = 0
      for idx in self.id2[nb]:
        C2[idx, z] = 1
        z += 1

      plt.scatter(self.b2[nb], self.d2[nb], color=color2, s=9)
      ot.plot.plot2D_samples_mat(np.stack((self.b2[nb], self.d2[nb]), 1), NbMaximas2PersDiag(j, 1, 0), C2, color=color2)
      plt.scatter(NbMaximas2PersDiag(j, 0, 1)[:, 1], NbMaximas2PersDiag(j, 0, 1)[:, 0], color='red', s=9)

    filename = str(nb) + '.png'
    lst.append(filename)
    plt.savefig(path + '/' + filename)
    plt.close()
  
  def show_history(self):
    """Plot loss history"""
    plt.plot(self.history)