#libraries
import networkx as nx
import numpy as np
import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

############################################################################################
#data cleaning: removing Y chromosome and isolated nodes

def data_cleaning(data0):
    #removing Y crhomosome
    Y = list(range(2892,2952))
    data=np.delete(data0,Y, axis=0)
    data=np.delete(data,Y, axis=1)

    #removing isolated nodes
    data=data[~np.all(data == 0, axis=1)]
    data=data[:, ~np.all(data == 0, axis=0)]

    return data


############################################################################################
#strength of nodes ---> degree centrality

def stregth(G):
    #strength of a node = sum of the weights of all the node's links
    weights={node: sum(weight for _, _, weight in G.edges(node, data='weight')) for node in G.nodes} 
    str=np.fromiter(weights.values(), dtype=float)

    return str

############################################################################################
#clustering coefficient: local(single node) and average

#function to compute local clustering for a single node
def local_clustering(node,G):
    result = nx.clustering(G, nodes=node, weight='weight')
    return result


def parallel_clustering(G):

    #list of nodes
    nodes = list(G.nodes)

    if __name__ == "__main__":

        #use a pool to compute local clustering in parallel
        with Pool(processes=4) as pool:

            with tqdm(total=len(nodes)) as pbar:
                    clustering_results = []
                    for result in pool.imap_unordered(local_clustering, nodes,G):
                        clustering_results.append(result)
                        pbar.update(1)

        #compute the average clustering coefficient
        avg_clustering_parallel = sum(clustering_results) / len(clustering_results)
        print(avg_clustering_parallel)

    return clustering_results,avg_clustering_parallel


############################################################################################
#cleaning indexes to match the cleaned data

def clean_indexing(matrix):
    #add Y indexes corresponding to the range 2893-2952
    ind=list(range(2893,2953))
    #removing isolated nodes indexes
    for i in range(matrix.shape[0]):
        #NOTE: i+1 bc this index starts from 1 and i from 0
        if np.all(matrix[i] == 0) and i not in ind:
            ind.append(i+1)
    return ind


############################################################################################
#adajacency matrix

#function to plot the adjacency matrix
def plot_adjacency_matrix(data, cell, N):
    plt.imshow(data, cmap='plasma', interpolation='none') #plasma or gist_heat
    plt.colorbar()
    plt.title(f'Adjacency Matrix {cell}: with {N} values')
    plt.show()
    return


#function to compute the adjacency matrix using the top N eigenvalues and eigenvectors
def compute_essential_matrix(eigval, eigvec, N, cell):
    #ordering in an absolute descending order 
    idx = np.argsort(np.abs(eigval))[::-1]
    # selecting top N eigenvalues and eigenvectors
    eigval_topN = eigval[idx][:N]
    eigvec_topN = eigvec[idx][:N]
    
    A_ess = np.zeros((len(eigval), len(eigval)))
    
    #compute the essential matrix using the formula
    for n in range(N):
        k_n = eigval_topN[n]
        a_n = eigvec_topN[n]
        
        #compute the contribution of eigenvector n to the essential matrix
        A_ess += k_n * np.outer(a_n, a_n)  
    
    #creating corresponding plot
    plot_adjacency_matrix(A_ess, cell, N)

    return 
