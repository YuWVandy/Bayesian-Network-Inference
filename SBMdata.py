# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:51:55 2020

@author: 10624
"""
import numpy as np

##Generate vertice failure sequence data
num = 4 #100 group of failure sceneria
fail_prob_graph = 10 ##Specify one graph to simulate the failure propagation
candidate_num = 10000 #100 graph candidates
fail_node_num = 5 #the number of the initial failure nodes
seed = 1


block_num = 3
node_num = [30, 20, 10]
edge_prob = [[0.08, 0.05, 0.02], [0.05, 0.08, 0.03], [0.02, 0.03, 0.08]]
fail_prob = np.array([[0.2, 0.01, 0.02], [0.02, 0.3, 0.04], [0.01, 0.03, 0.1]])
color = ['maroon', 'navy', 'darkgreen']
