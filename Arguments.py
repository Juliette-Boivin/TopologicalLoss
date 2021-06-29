#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import TestData



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-graph', default=TestData.testdata["multi_max_noisy"][0], help='the graph associated to the input image')
    parser.add_argument('-image', default=TestData.testdata["multi_max_noisy"][1], help='the image to optimize')
    parser.add_argument('-nb_iter', default=10, help='number of loops in the Optimizer')
    parser.add_argument('-loss', default='minmax', choices=['min', 'max', 'minmax'])
    parser.add_argument('-nb_holes', default=2, type=int, 
                        help='number of holes to enforce in the image')
    parser.add_argument('-nb_cc', default=2, type=int, 
                        help='number of connected components to enforce in the image')
    parser.add_argument('-thresh_min', default=1e-5, type=float, 
                        help='threshold of altitudes difference used to cut nodes in the mintree')
    parser.add_argument('-thresh_max', default=1e-10, type=float, 
                        help='threshold of altitudes difference used to cut nodes in the maxtree')
    parser.add_argument('-perf_dgm', default='None', help='target diagram')
    parser.add_argument('-img_grph_GT', default='None', help='target graph and target image')
    parser.add_argument('-optimize_maxtree', default=False, help='target maxtree')
    parser.add_argument('-color_min', default='blue', type=str,
                        help="color of the diagram's points mapped to the mintree's leaves")
    parser.add_argument('-color_max', default='red', type=str,
                        help="color of the diagram's points mapped to the maxtree's leaves")
    parser.add_argument('-plot_loss', default=True, type=bool,
                        help='if True, the loss function is displayed')
    parser.add_argument('-result', default=True, type=bool,
                        help='if True, the optimized image is displayed')
    parser.add_argument('-diag_beg', default=True, type=bool,
                        help='if True, the diagram at iteration zero is displayed')
    parser.add_argument('-diag_end', default=True, type=bool,
                        help='if True, the diagram at the last iteration is displayed')
    parser.add_argument('-nb_leaves', default=True, type=bool,
                        help='if True, the evolution of the number of nodes is displayed')
    parser.add_argument('-save_fig', default=False, type=bool,
                        help='if True, the figure is saved in the path filled out.')
    parser.add_argument('-save_diagrams', default=False, type=bool,
                        help='if True, slices of the diagram optimized are saved and a GIF is created')
    parser.add_argument('-save_results', default=False, type=bool,
                        help='if True, slices of the image optimized are saved and a GIF is created')
    
    args = parser.parse_args()
    
    return args
    