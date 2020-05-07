# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 23:07:08 2020

@author: 10624
"""

import numpy as np
from matplotlib import pyplot as plt
from StochasticBlockModel import SBM
import SBMdata as dt

def normalize(data):
    """Normalize each entry of the data based on division by the sum of the data
    Input: data(an numpy array)
    Output: normalized data
    """
    data = data/np.sum(data)
    return data

def likelihood(num, itemnum, data, adj_matrix, fail_prob, community):
    """Calculate the likelihood value based on the data and the model
    Input: Data
    Output: likelihood
    """
    likelihood = np.zeros([num, itemnum])
    for i in range(num):
        for j in range(itemnum):
            likelihood[i, j] = model(data[i], adj_matrix[j], fail_prob, community)
    return likelihood

def model(data, adj_matrix, fail_prob, community):
    """Calculate the epidemic model value given the failure sequence and the adj_matrix
    """
    prob = 1
    
    for i in range(len(data) - 1):
        fail1 = data[i]
        fail2 = data[i + 1]
        for j in range(len(adj_matrix)):
            m = 0
            for k in range(len(adj_matrix)):
                m += adj_matrix[k, j]*np.log(1 - fail_prob[community[k], community[j]])*fail1[k]

            temp = (((1 - np.exp(m))**fail2[j])*(np.exp(m)**(1 - fail2[j])))**(1 - fail1[j])
            
            prob = prob*temp
    
    return prob

def posterior(prior, likelihood, num, itemnum):
    """Update the posterior distribution based on prior and likelihood
    """
    posterior = np.zeros([num + 1, itemnum])
    posterior[0, :] = prior
    for i in range(num):
        for j in range(itemnum):
            posterior[i + 1, j] = posterior[i, j]*likelihood[i, j]
        posterior[i + 1, :] = normalize(posterior[i + 1, :])
    
    return posterior

def normalize_prior(prior):
    """Normalize the prior probability with log form
    """
    prior = np.array(prior)
    temp = -np.min(prior)
    
    prior = np.exp(prior + temp)
    prior = prior/np.sum(prior)
    
    return prior

##Generate candidate graphs
graphs = []
fail_seq_data = []
prior = []
adj_matrix = []
for i in range(dt.candidate_num):
    graph = SBM(dt.block_num, dt.node_num, dt.edge_prob, dt.fail_prob, dt.color)
    graphs.append(graph)
    adj_matrix.append(graph.adj_matrix)
    prior.append(graph.graph_log_prob)
    
    if(i == dt.fail_prob_graph):
        for j in range(dt.num):
            initial_fail_num = np.random.randint(10)
            graph.generate_initial_failure(initial_fail_num, dt.seed)
            graph.failure_simulation()
        
            fail_seq_data.append(graph.node_fail_sequence)
            
community = graph.community
        
##Prior probability
prior = normalize_prior(prior)
like = likelihood(dt.num, dt.candidate_num, fail_seq_data, adj_matrix, dt.fail_prob, community)
post_prob = posterior(prior, like, dt.num, dt.candidate_num)
    
    
    
    
    