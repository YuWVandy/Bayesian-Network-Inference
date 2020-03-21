# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 00:15:24 2020

@author: 10624
"""
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

class Random_Graph(object):
    """Define random graph class
    Input: node number, edge probability
    Output: an undirected random graph
    """
    
    def __init__(self, node_num = None, edge_prob = None):
        self.n = node_num
        self.p = edge_prob
        
        self.generate_random_graph(self.n, self.p)
        
    def generate_random_graph(self, node_num, edge_prob):
        """Generate the undirected random graph based on the number of nodes and edge probability
        Monte Carlo Simulation on each node pair
        Input: the number of nodes and the probability of edges
        Output: the adjacent list and the adjacent matrix
        """
        from collections import defaultdict
        
        self.adj_matrix = np.zeros([node_num, node_num])
        self.adj_list = dict()
        
        for i in range(node_num):
            self.adj_list["{}".format(i)] = []
            
        for i in range(node_num):
            for j in range(i + 1, node_num):
                temp = np.random.rand()
                if(temp <= edge_prob):
                    self.adj_matrix[i, j] = 1
                    self.adj_matrix[j, i] = 1
                    self.adj_list["{}".format(i)].append(j)
                    self.adj_list["{}".format(j)].append(i)
    
    def edge_failure_matrix(self, probability):
        """Calculate the failure matrix of the whole graph based on given instructions
        Input:
        Output: failure matrix of dimension n*n, fail_matrix[i, j] represents the failure probability of edge (i, j)
        """
        
        return(probability*np.ones([self.n, self.n]))
        
    def failure_probability(self):
        """Calculate the node failure probability based on failure_matrix
        Input: failure probability of each edge (conditional failure probability)
               failure_sequence: node failure sequence: 1 - failed, 2 - normal
        Output: failure_probability of each node
        """
        
        node_fail_prob = np.zeros(self.n)
        node_fail_sequence = self.node_fail_sequence[-1]
        for i in range(self.n):
            temp = 0
            for j in range(self.n):
                temp += self.adj_matrix[i, j]*np.log(1 - self.edge_fail_matrix[i, j])*node_fail_sequence[j]
            node_fail_prob[i] = 1 - np.exp(temp)
            
        self.node_fail_prob.append(node_fail_prob)
        
    def failure_sequence(self):
        """Simulate one further node failure sceneria based on MC simulation
        Input: the node failure sceneria at the previous step, the node_fail_probability at previous step
        Output: the node failure sceneria at the current step
        """
        
        node_fail_sequence = np.zeros(self.n)
        node_fail_prob = self.node_fail_prob[-1]
        for i in range(self.n):
            if(self.node_fail_sequence[-1][i] == 0):
                temp = np.random.rand()
                if(temp < node_fail_prob[i]):
                    node_fail_sequence[i] = 1
#            else:
#                node_fail_sequence[i] = 1
        
        self.node_fail_sequence.append(node_fail_sequence)
    
    def generate_initial_failure(self, num):
        """Generate the initial node failure sceneria
        Input: the number of initial failure nodes
        output: the initial failure sequence
        """
        
        initial_node_failure = np.zeros(self.n)
        temp = np.random.randint(self.n, size = num)
        initial_node_failure[temp] = 1
        self.node_fail_sequence.append(initial_node_failure)
        self.node_fail_final.append(initial_node_failure)
        
        
    def failure_simulation(self, num, probability):
        """Simulate the node failure sequence along the time
        Input: Initial node failure sceneria
        Output: Node failure sequence, node failure probability
        """
        
        self.node_fail_prob = []
        self.node_fail_sequence = []
        
        self.node_fail_final = []
        
        
        self.edge_fail_matrix = self.edge_failure_matrix(probability)
        self.generate_initial_failure(num)
        
        while(1):
            self.failure_probability()
            self.failure_sequence()
            
            node_fail_final = np.zeros(self.n)
            for i in range(self.n):
                if(self.node_fail_final[-1][i] == 1 or self.node_fail_sequence[-1][i] == 1):
                    node_fail_final[i] =1
            self.node_fail_final.append(node_fail_final)
            
            if((self.node_fail_final[-1] == self.node_fail_final[-2]).all() or (np.sum(self.node_fail_final[-1]) == self.n)):
                break
        
        
        
        