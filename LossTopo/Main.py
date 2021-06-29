# -*- coding: utf-8 -*-
from Experiment import experiment
from Arguments import get_args
import TestData

args = get_args()
grapp, imagg = TestData.testdata["chase_images"]["08R"]
graph_chase_GT, image_chase_GT = TestData.testdata["chase_1stHO"]["08R"]
experiment(grapp, imagg, args.nb_iter, args.loss, args.nb_holes, args.nb_cc, args.thresh_min, args.thresh_max, 
           args.perf_dgm, (graph_chase_GT, image_chase_GT.astype(float)/255), args.optimize_maxtree, args.color_min, args.color_max, args.plot_loss, 
           args.result, args.diag_beg, args.diag_end, args.nb_leaves, args.save_fig, args.save_diagrams, args.save_results)


#import edist
#pip install edist
#pip install edist==1.2.0

