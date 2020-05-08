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

def normalize2(data):
    data = np.exp(data)/np.sum(np.exp(data))
    return data
def likelihood(num, itemnum, data, adj_matrix, fail_prob):
    """Calculate the likelihood value based on the data and the model
    Input: Data
    Output: likelihood
    """
    likelihood = np.ones([num, itemnum])
    loglike = np.zeros([num, itemnum])
    
    update_like = np.array([[None]*itemnum]*num)
    min_update_like = np.zeros(num)
    for i in range(num):
        min_update = math.inf
        for j in range(itemnum):
            update_like[i, j] = model(data[i], adj_matrix[j], fail_prob)
            for k in range(len(update_like[i, j])):
                if(update_like[i, j][k] != 0 and update_like[i, j][k] < min_update):
                    min_update = update_like[i, j][k]
            print(i, j)
        min_update_like[i] = min_update
    
    for i in range(num):
        for j in range(itemnum):
            for k in range(len(update_like[i, j])):
                if(update_like[i, j][k] == 0):
                    likelihood[i, j] = likelihood[i, j]*min_update_like[i]
                    loglike[i, j] = loglike[i, j] + np.log(min_update_like[i])
                else:
                    likelihood[i, j] = likelihood[i, j]*update_like[i, j][k]
                    loglike[i, j] = loglike[i, j] + np.log(min_update_like[i, j][k])
            print(i, j)
    
#    likelihood = np.zeros([num, itemnum])
#    for i in range(num):
#        for j in range(itemnum):
#            likelihood[i, j] = model(data[i], adj_matrix[j], fail_prob)
#            print(i, j)
            
    return likelihood, loglike

def model(data, adj_matrix, fail_prob):
    """Calculate the epidemic model value given the failure sequence and the adj_matrix
    """
    Temp = []
    
    for i in range(len(data) - 1):
        fail1 = data[i]
        fail2 = data[i + 1]
        for j in range(len(adj_matrix)):
            m = 0
            for k in range(len(adj_matrix)):
                m += adj_matrix[k, j]*np.log(1 - fail_prob)*fail1[k]
            
            temp = (((1 - np.exp(m))**fail2[j])*(np.exp(m)**(1 - fail2[j])))**(1 - fail1[j])
            Temp.append(temp)
    
    return Temp
            
#    prob = 1
#    
#    for i in range(len(data) - 1):
#        fail1 = data[i]
#        fail2 = data[i + 1]
#        for j in range(len(adj_matrix)):
#            m = 0
#            for k in range(len(adj_matrix)):
#                m += adj_matrix[k, j]*np.log(1 - fail_prob)*fail1[k]
#            
#            temp = (((1 - np.exp(m))**fail2[j])*(np.exp(m)**(1 - fail2[j])))**(1 - fail1[j])
#            prob = prob*temp
    
#    return prob

def posterior(prior, likelihood, loglike, num, itemnum):
    """Update the posterior distribution based on prior and likelihood
    Input: prior(n*(n-1)/2+1, 1), likelihood(datanum, n*(n-1)/2+1)
    """
    posterior = np.zeros([num + 1, itemnum])
    posterior[0, :] = prior
    logposterior = np.zeros([num + 1, itemnum])
    logposterior[0, :] = np.log(prior)
    
    for i in range(num):
        for j in range(itemnum):
            posterior[i + 1, j] = posterior[i, j]*likelihood[i, j]
            logposterior[i + 1, j] = logposterior[i, j] + loglike[i, j]
        posterior[i + 1, :] = normalize(posterior[i + 1, :])
        logposterior[i + 1, j] = normalize2(logposterior[i + 1, :])
    
    return posterior, logposterior

def adj_graph(adj_matrix, c):
    """set up the graph object from adj_matrix based using networkx package
    """
    import networkx as nx
        
    G = nx.convert_matrix.from_numpy_matrix(adj_matrix)
    nx.draw(G, nx.random_layout(G), with_labels = True, node_color = c, node_size = 600, font_color='white', font_size = 15)

def posterior_similarity(obj_adj_matrix, adj_matrix, posterior):
    """Calculate the posterial similarity based on equation in that paper
    """
    
    post_similar = []
    for i in range(len(posterior)):
        temp = 0
        for j in range(len(obj_adj_matrix)):
            for k in range(len(obj_adj_matrix[j])):
                temp += obj_adj_matrix[j, k]
                for l in range(len(adj_matrix)):
                    temp += - posterior[i, l]*adj_matrix[l][j, k]
        
        post_similar.append(1 - np.abs(temp/len(obj_adj_matrix)**2))

    return post_similar  
    
##Generate vertice failure sequence data

num = 5
color1 = ['red', 'orange', 'orange', 'tomato', 'purple']
color2 = ['green', 'blue', 'purple', 'teal', 'royalblue']

node_num = 50
edge_prob = 0.2
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

adj_num = 500
likearray = np.zeros([adj_num, num, itemnum])
postarray = np.zeros([adj_num, num+1, itemnum])
post_similarityarray = np.zeros([adj_num, num + 1])

#for j in range(len(adj_num)):
fail_seq_data = []
#    print(adj_num[j])
for i in range(num):
    random_graph = Random_Graph(node_num, edge_prob)
    initial_fail_num = np.random.randint(20)
    fail_prob = 0.2
#    fail_prob = 0.2*np.random.rand()
    random_graph.generate_initial_failure(initial_fail_num, seed)
    random_graph.adj_matrix = adj_matrix[adj_num]
    random_graph.failure_simulation(fail_prob)
    
    fail_seq_data.append(random_graph.node_fail_sequence)
#    random_graph.visual_failure_process(1, color1[i], color2[i])

##Prior probability
prior = normalize(np.ones(len(adj_matrix)))
like, loglike = likelihood(num, itemnum, fail_seq_data, adj_matrix, fail_prob)
post_prob = posterior(prior, like, loglike, num, itemnum)
#post_similarity = posterior_similarity(adj_matrix[adj_num[j]], adj_matrix, post_prob)
#likearray[j, :, :] = like
#postarray[j, :, :] = post_prob
#post_similarityarray[j, :] = post_similarity
#    

plt.figure(figsize = (14,10))
plt.plot(np.arange(1, itemnum+1, 1), post_prob[0], label = '0 - fs')
plt.plot(np.arange(1, itemnum+1, 1), post_prob[1], label = '1 - fs')
plt.plot(np.arange(1, itemnum+1, 1), post_prob[2], label = '2 - fs')
plt.plot(np.arange(1, itemnum+1, 1), post_prob[3], label = '3 - fs')
plt.plot(np.arange(1, itemnum+1, 1), post_prob[4], label = '4 - fs')
plt.plot(np.arange(1, itemnum+1, 1), post_prob[5], label = '5 - fs')
plt.plot(np.arange(1, itemnum+1, 1), post_prob[6], label = '6 - fs')
plt.plot(np.arange(1, itemnum+1, 1), post_prob[7], label = '7 - fs')
plt.plot(np.arange(1, itemnum+1, 1), post_prob[8], label = '8 - fs')
plt.plot(np.arange(1, itemnum+1, 1), post_prob[9], label = '9 - fs')
plt.plot(np.arange(1, itemnum+1, 1), post_prob[10], label = '10 - fs')
plt.plot(np.arange(1, itemnum+1, 1), post_prob[11], label = '11 - fs')
plt.plot(np.arange(1, itemnum+1, 1), post_prob[12], label = '12 - fs')
plt.plot(np.arange(1, itemnum+1, 1), post_prob[13], label = '13 - fs')
plt.plot(np.arange(1, itemnum+1, 1), post_prob[14], label = '14 - fs')
plt.plot(np.arange(1, itemnum+1, 1), post_prob[15], label = '15 - fs')
plt.plot(np.arange(1, itemnum+1, 1), post_prob[16], label = '16 - fs')
plt.plot(np.arange(1, itemnum+1, 1), post_prob[17], label = '17 - fs')
plt.plot(np.arange(1, itemnum+1, 1), post_prob[18], label = '18 - fs')
plt.plot(np.arange(1, itemnum+1, 1), post_prob[19], label = '19 - fs')
plt.plot(np.arange(1, itemnum+1, 1), post_prob[20], label = '20 - fs')
plt.xlabel('Graph number')
plt.ylabel('Probability')
plt.grid(True)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1, frameon = 0)



lower = 0
upper = 250+1
plt.plot(np.arange(lower, upper, 1), post_prob[0][lower:upper], label = '0 - fs')
plt.plot(np.arange(lower, upper, 1), post_prob[1][lower:upper], label = '1 - fs')
plt.plot(np.arange(lower, upper, 1), post_prob[2][lower:upper], label = '2 - fs')
plt.plot(np.arange(lower, upper, 1), post_prob[3][lower:upper], label = '3 - fs')
plt.plot(np.arange(lower, upper, 1), post_prob[4][lower:upper], label = '4 - fs')
plt.plot(np.arange(lower, upper, 1), post_prob[5][lower:upper], label = '5 - fs')
plt.plot(np.arange(lower, upper, 1), post_prob[10][lower:upper], label = '10 - fs')
plt.plot(np.arange(lower, upper, 1), post_prob[30][lower:upper], label = '30 - fs')
plt.plot(np.arange(lower, upper, 1), post_prob[50][lower:upper], label = '50 - fs')
plt.xlabel('Graph number')
plt.ylabel('Probability')
plt.grid(True)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1, frameon = 0)

#Calculate the posterial similarity

plt.scatter(np.arange(0, num+1, 1), post_similarity)

for i in range(len(post_similarityarray)):
    for j in range(len(post_similarityarray[i])):
        if(post_similarityarray[i, j] > 1):
            post_similarityarray[i, j] = 2 - post_similarityarray[i, j]
            

plt.figure(figsize = (10, 7))
for i in np.arange(0, len(post_similarityarray), 10):
    plt.plot(np.arange(0, num+1, 1), post_similarityarray[i], label = '{} graph'.format(i*10))
plt.xlabel('The number of data set')
plt.ylabel('Posterior Similarity')
plt.grid(True)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1, frameon = 0)

#graph_num = 70
#plt.plot(np.arange(1, itemnum+1, 1), postarray[graph_num][0], label = '0 - fs')
#plt.plot(np.arange(1, itemnum+1, 1), postarray[graph_num][1], label = '1 - fs')
#plt.plot(np.arange(1, itemnum+1, 1), postarray[graph_num][2], label = '2 - fs')
#plt.plot(np.arange(1, itemnum+1, 1), postarray[graph_num][3], label = '3 - fs')
#plt.plot(np.arange(1, itemnum+1, 1), postarray[graph_num][4], label = '4 - fs')
#plt.plot(np.arange(1, itemnum+1, 1), postarray[graph_num][5], label = '5 - fs')
#plt.plot(np.arange(1, itemnum+1, 1), postarray[graph_num][10], label = '10 - fs')
#plt.plot(np.arange(1, itemnum+1, 1), postarray[graph_num][30], label = '30 - fs')
#plt.xlabel('Graph number')
#plt.ylabel('Probability')
#plt.grid(True)
#plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1, frameon = 0)
    