# -*- coding: utf-8 -*-
import numpy as np
import imageio
import matplotlib.pyplot as plt
import higra as hg
import os

testimage = {}
np.random.seed(42)

def draw_Gaussian_2d(im, x0, y0, sigma_X, sigma_Y, theta, A):
  """
  Draw a 2d gaussian in the buffer image im at position (x0, y0), with 
  standard deviation sigma_X and sigma_Y, a rotation of theta, and a central
  brightness equal to A.
  """

  X, Y = np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]))

  a = np.cos(theta)**2/(2*sigma_X**2) + np.sin(theta)**2/(2*sigma_Y**2)
  b = -np.sin(2*theta)/(4*sigma_X**2) + np.sin(2*theta)/(4*sigma_Y**2)
  c = np.sin(theta)**2/(2*sigma_X**2) + np.cos(theta)**2/(2*sigma_Y**2)

  Z = A*np.exp( - (a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))
  
  im += Z


def gen_multi_max(noise):
  """
  Create a test image with 4 maxima of various sizes and brightnesses
  """
  im = np.zeros((100, 100))
  draw_Gaussian_2d(im, 30, 40, 18, 25, -0.3, 1.7)
  draw_Gaussian_2d(im, 20, 90, 5, 5, 0, 2)
  draw_Gaussian_2d(im, 80, 30, 12, 5, 1.8, 1.9)
  draw_Gaussian_2d(im, 80, 80, 15, 15, 0, 1.6)

  im = im / im.max()
  if noise > 0:
    im = im + np.random.randn(*im.shape) * noise
    im = im / im.max()
  return im

def gen_multi_max2(noise):
  """
  Create a test image with 4 maxima of various sizes and brightnesses
  """
  im = np.zeros((100, 100))
  draw_Gaussian_2d(im, 30, 40, 18, 25, -0.3, 1.7)
  draw_Gaussian_2d(im, 30, 40, 6, 6, -0.1, -1)
  draw_Gaussian_2d(im, 20, 90, 5, 5, 0, 2)
  draw_Gaussian_2d(im, 20, 90, 3, 3, 0, -2)
  draw_Gaussian_2d(im, 80, 30, 12, 5, 1.8, 1.9)
  draw_Gaussian_2d(im, 80, 80, 15, 15, 0, 1.6)

  im = im / im.max()
  if noise > 0:
    im = im + np.random.randn(*im.shape) * noise
    im = im / im.max()
  return im

testimage["multi_max"] = gen_multi_max(noise=0)
testimage["multi_max_noisy"] = gen_multi_max(noise=0.05)
testimage["multi_max_noisy_and_hole"] = gen_multi_max2(noise=0.05)

testimage["neurite"] = imageio.imread("http://perso.esiee.fr/~perretb/neurite.png")[:,:,0]
testimage["neurite"] = 1 - testimage["neurite"]/testimage["neurite"].max()
testimage["neurite_crop"] = testimage["neurite"][0:150,50:200]


figsize = 4
fig = plt.figure(figsize=(figsize * figsize, len(testimage)))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
def plot_im(rows, columns, y, x, im, title):
    plt.subplot(rows, columns, y * columns + x + 1); 
    plt.imshow(im, interpolation="bicubic", cmap="gray"); 
    plt.xticks([]); plt.yticks([])
    plt.title(title)

testdata = {}
for i, (k, im) in enumerate(testimage.items()):
  testdata[k] = (hg.get_8_adjacency_implicit_graph(im.shape), im)
  #plot_im(1, len(testimage), 0, i, im, k)
  

#CHASEDB1 images and ground truth segmentation

chase_images_name = input("path of the train data: ")
chase_1stHO_name = input("path of the GT data: ")

testimage["chase_images"] = {}
chase_images = []

testimage["chase_1stHO"] = {}
chase_1stHO = []

for test_name in os.listdir(chase_1stHO_name):
  chase_1stHO.extend(test_name.split('_')[1][:3])
  testimage["chase_1stHO"][test_name.split('_')[1][:3]] = imageio.imread(chase_1stHO_name + '/' + test_name)

for train_name in os.listdir(str(chase_images_name)):
  chase_images.extend(train_name.split('_')[1][:3])
  testimage["chase_images"][train_name.split('_')[1][:3]] = imageio.imread(chase_images_name + '/' + train_name)[:, :, 0]


  
  
  
  
  
  
  
  
  
  
  
  
