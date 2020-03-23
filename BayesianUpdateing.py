# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:12:10 2020

@author: 10624
"""
import numpy as np
from matplotlib import pyplot as plt
from RandomGraph import Random_Graph

def normalize(data):
    """Normalize each entry of the data based on division by the sum of the data
    Input: data(an numpy array)
    Output: normalized data
    """
    data = data/np.sum(data)
    return data

def likelihood(num, itemnum, data, adj_matrix, fail_prob):
    """Calculate the likelihood value based on the data and the model
    Input: Data
    Output: likelihood
    """
    likelihood = np.zeros([num, itemnum])
    for i in range(num):
        for j in range(itemnum):
            likelihood[i, j] = model(data[i], adj_matrix[j], fail_prob)
            print(i, j)
    
    return likelihood

def model(data, adj_matrix, fail_prob):
    """Calculate the epidemic model value given the failure sequence and the adj_matrix
    """
    prob = 1
    
    for i in range(len(data) - 1):
        fail1 = data[i]
        fail2 = data[i + 1]
        for j in range(len(adj_matrix)):
            m = 0
            for k in range(len(adj_matrix)):
                m += adj_matrix[k, j]*np.log(1 - fail_prob)*fail1[k]
            temp = (((1 - np.exp(m))**fail2[j])*(np.exp(m)**(1 - fail2[j])))**(1 - fail1[j])
            
            prob = prob*temp
    
    return prob

def posterior(prior, likelihood, num, itemnum):
    """Update the posterior distribution based on prior and likelihood
    Input: prior(n*(n-1)/2+1, 1), likelihood(datanum, n*(n-1)/2+1)
    """
    posterior = np.zeros([num + 1, itemnum])
    posterior[0, :] = prior
    for i in range(num):
        for j in range(itemnum):
            posterior[i + 1, j] = posterior[i, j]*likelihood[i, j]
        posterior[i + 1, :] = normalize(posterior[i + 1, :])
    
    return posterior

def adj_graph(adj_matrix, c):
    """set up the graph object from adj_matrix based using networkx package
    """
    import networkx as nx
        
    G = nx.convert_matrix.from_numpy_matrix(adj_matrix)
    nx.draw(G, nx.random_layout(G), with_labels = True, node_color = c, node_size = 600, font_color='white', font_size = 15)

    
##Generate vertice failure sequence data
fail_seq_data = []
num = 100
color1 = ['red', 'orange', 'orange', 'tomato', 'purple']
color2 = ['green', 'blue', 'purple', 'teal', 'royalblue']

node_num = 50
edge_prob = 0.08
#initial_fail_num = 10
#fail_prob = 0.5
seed = 1

itemnum = int(node_num*(node_num - 1)/2 + 1)

##Generate n*(n-1)/2 graphs
adj_matrix = [np.zeros([node_num, node_num])]
for i in range(node_num):
    for j in range(i + 1, node_num):
        temp = np.copy(adj_matrix[-1])
        temp[i, j] = 1
        temp[j, i] = 1
        adj_matrix.append(temp)

for i in range(num):
    random_graph = Random_Graph(node_num, edge_prob)
    initial_fail_num = np.random.randint(10)
#    fail_prob = 0.5*np.random.rand()
    random_graph.generate_initial_failure(initial_fail_num, seed)
    random_graph.adj_matrix = adj_matrix[300]
    random_graph.failure_simulation(fail_prob)
    
    fail_seq_data.append(random_graph.node_fail_sequence)
#    random_graph.visual_failure_process(1, color1[i], color2[i])

##Prior probability
prior = normalize(np.ones(len(adj_matrix)))
like = likelihood(num, itemnum, fail_seq_data, adj_matrix, fail_prob)
post_prob = posterior(prior, like, num, itemnum)