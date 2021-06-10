# TopologicalLoss
Optimize images thanks to persistence diagrams and/or maxtrees

Arguments.py: Parameters to choose

Attribute.py: Auxiliary functions to compute depth and saddle nodes of a tree (and its associated altitudes)

ComponentTree.py: Class of component trees with a custom backward layer (based on https://perso.esiee.fr/~perretb/tmp/Component_Tree_Loss.ipynb)

Diagrams.py: Compute synthetic diagrams from a wished number of maximas 

Experiment.py: Optimize the image and display differents results

Loss.py: Loss function (use some parts of https://github.com/HuXiaoling/TopoLoss and the geomloss library)

Optimizer.py: Optimization function (and some functions to save results to be displayed)

TestData.py: Load images and associated graphs to test the optimization

Main.py: Launch an experiment 
