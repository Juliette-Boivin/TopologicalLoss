# -*- coding: utf-8 -*-
from Experiment import experiment
import TestData


graph, image = TestData.testdata["multi_max_noisy"]
experiment(graph, image, nb_iter=10, loss='minmax', nb_cc=2, thresh_max=1e-5, thresh_min=1e-10)



#import edist
#pip install edist
#pip install edist==1.2.0

