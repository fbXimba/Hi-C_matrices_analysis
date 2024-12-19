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

#indexing clean

def dict_indexing(data0,chrom,start,end):
    #chrom:
    #GM12878 and KBM7:dfchr["chr"]
    #HMEC and NHEK: chromosomes_ HMEC_NHEK[:0]

    #start:
    #GM12878 and KBM7:dfchr["start"]
    #HMEC and NHEK: chromosomes_ HMEC_NHEK[:1]

    #end:
    #GM12878 and KBM7:dfchr["end"]
    #HMEC and NHEK: chromosomes_ HMEC_NHEK[:2]

    ind=clean_indexing(data0)

    ind_chr=[]

    #create a dictionary with the indexes of the chromosomes
    for i in range(len(chrom)):
        ind_chr.append(list(range(start[i],end[i]+start[0])))

    chro = dict(zip(chrom, ind_chr)) 

    #remove the indexes of the Y chromosome and the isolated nodes
    a=0
    for key, value in chro.items():
        up_ind = [x + 1 for x in ind]
        new_values=[item for item in value if item not in up_ind]
        #substitute old values with new ones starting from zero and increasin
        if len(new_values)!=0:
            new_values=list(range(a,a+len(new_values)))
            a+=len(new_values)  
            chro[key]=new_values
        else:
            chro[key]=[]

    return chro
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
    ind=list(range(2892,2952))
    #removing isolated nodes indexes
    for i in range(matrix.shape[0]):
        if np.all(matrix[i] == 0) and i not in ind:
            ind.append(i)
    return ind


############################################################################################
#adajacency matrix

#function to plot the adjacency matrix
def plot_adjacency_matrix(data, cell, N, color):
    plt.imshow(data, cmap= color, interpolation='none') #plasma or gist_heat
    plt.colorbar()
    plt.title(f'Adjacency Matrix {cell}: with {N}')
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

############################################################################################
#thresholding with mask

def thresholding(data, t):
    thr=np.where(data>t,1,0)
    
    return thr

############################################################################################
#cluster view

def cluster_view (community_list,cell):
    matrix = np.zeros((2888,2888))
    
    for cluster in community_list:
            for i in cluster:
                for j in cluster:
                    matrix[i,j] = len(community_list) - community_list.index(cluster) + 1
     
    plt.imshow(matrix, cmap='nipy_spectral', interpolation='none') 
    plt.colorbar()
    plt.title(f"Community Visualization Matrix {cell} with {len(community_list)} clusters")
    plt.show()
    
    return matrix

############################################################################################
#clusters scatter plot

def cluster_scatter(part,chro,chrom,cell ):

    #chrom:
    #GM12878 and KBM7:dfchr["chr"]
    #HMEC and NHEK: chromosomes_ HMEC_NHEK[:,0]

    cmap = plt.get_cmap('nipy_spectral')
    colors = [cmap(i) for i in np.linspace(0, 1, 23)]
    
    n=0
    for j in range(len(part)):
        cl=part[j]
        for i, color in enumerate(colors, start=0):
            r=chro[chrom[i]]
            cl_chr= [val for val in r if val in cl]
            n=(np.array(range(len(cl_chr)))/len(cl_chr))+j
            plt.scatter(n,cl_chr, s=3 ,label=f"chr {i+1}", color=color)
        plt.axvline(x=j, color="black", lw=0.1)
        plt.ylabel("indexes")
        plt.xlabel("clusters")
        plt.title(f"{cell} {j+1} clusters distribution")
    plt.show()

    return

