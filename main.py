# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:36:42 2020

@author: 10624
"""

import networkx as nx

G = nx.generators.random_graphs.erdos_renyi_graph(5, 0.3)
nx.draw(G, nx.random_layout(G),with_labels = True, node_color = 'mediumorchid', node_size = 600, font_color='white', font_size = 15)


