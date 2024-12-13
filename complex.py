#%%
#libaries imported
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from scipy.stats import pearsonr
import math
#import random as rnd
import networkx as nx
import time as tm
from multiprocessing import Pool
from tqdm import tqdm 
from numpy import linalg as LA

#NOTE: This code is currently a work in progress in its initial stages. NO BITCHING.

#%%
#directories and data import
t0=tm.time()

#get working ditectory and create Data , HMEC and NHRK directories
dir_home=os.getcwd()
dir_data=dir_home+"/Data/"
dir_HMEC=dir_data+"raw_HMEC_1Mb/"
dir_NHEK=dir_data+"raw_NHEK_1Mb/"

#read data from GM12878 and KBM7
#GM12878 healthy cells
dfGM=pd.read_csv(dir_data+"raw_GM12878_1Mb.csv",header=None)
dataGM0=dfGM.to_numpy()

#KBM7 aberrant cells
dfKBM=pd.read_csv(dir_data+"raw_KBM7_1Mb.csv",header=None)
dataKBM0=dfKBM.to_numpy()

#%%
#Data cleaning

#removing Y crhomosome from data
Y = list(range(2892,2951))

#GM12878
dataGM = np.delete(dataGM0,Y, axis=0)
dataGM = np.delete(dataGM,Y, axis=1)

#KBM7
dataKBM = np.delete(dataKBM0,Y, axis=0)
dataKBM = np.delete(dataKBM,Y, axis=1)

#removing isolated nodes
#GM12878
dataGM = dataGM[~np.all(dataGM == 0, axis=1)]
dataGM = dataGM[:, ~np.all(dataGM == 0, axis=0)]

#KBM7
dataKBM = dataKBM[~np.all(dataKBM == 0, axis=1)]
dataKBM = dataKBM[:, ~np.all(dataKBM == 0, axis=0)]

#log_scale for better visualization
#log1p does log10(1+x) in order to avoid problems where there's a 0
#GM12878
dataGM_normalized = np.log1p(dataGM)

#KBM7
dataKBM_normalized = np.log1p(dataKBM)

#%%
#Creation of graphs
#from array (faster than from dataframe)
t2=tm.time()

#GM12878
G_GM= nx.from_numpy_array(dataGM_normalized)

#KBM7
G_KBM= nx.from_numpy_array(dataKBM_normalized)

#%%
#Adjacency matrix visualization
#GM12878
plt.imshow(dataGM_normalized, cmap='plasma', interpolation='none')
plt.colorbar()
plt.title('Adjacency Matrix GM12878')
plt.show()

#KBM7
plt.imshow(dataKBM_normalized, cmap='gist_heat', interpolation='none')
plt.colorbar()
plt.title('Adjacency Matrix KBM7')
plt.show()
#%%
#Strength of nodes
t3=tm.time()

#GM12878
weights_GM={node: sum(weight for _, _, weight in G_GM.edges(node, data='weight')) for node in G_GM.nodes} 
str_GM=np.fromiter(weights_GM.values(), dtype=float)
print(np.min(str_GM), np.max(str_GM), np.mean(str_GM)) #check min and max values

#KBM7
weights_KBM={node: sum(weight for _, _, weight in G_KBM.edges(node, data='weight')) for node in G_KBM.nodes}
str_KBM=np.fromiter(weights_KBM.values(), dtype=float)
print(np.min(str_KBM), np.max(str_KBM), np.mean(str_KBM)) 

#%%
#Histograms of strength
t4=tm.time()

#GM12878
plt.hist(str_GM, bins=400)
plt.xlim(-1, np.max(str_GM))
plt.xlabel("log_10(strength)")
plt.title("strength histogram GM normalized")
plt.show()

#KBM7
plt.hist(str_KBM, bins=400)
plt.xlim(-1, np.max(str_KBM))
plt.xlabel("log_10(strength)")
plt.title("strength histogram KBM normalized")
plt.show()

#%%
#Average clustering coefficient
t5=tm.time()

# Function to compute local clustering for a single node
def local_clustering(node):
    result = nx.clustering(G_GM, nodes=node, weight='weight')
    return result

# List of nodes
nodes = list(G_GM.nodes)

if __name__ == "__main__":

    # Use a pool to compute local clustering in parallel
    with Pool(processes=4) as pool:

        with tqdm(total=len(nodes)) as pbar:
                clustering_resultsGM = []
                for result in pool.imap_unordered(local_clustering, nodes):
                    clustering_resultsGM.append(result)
                    pbar.update(1)
        
        
    # Compute the average clustering coefficient
    avg_clustering_parallelGM = sum(clustering_resultsGM) / len(clustering_resultsGM)
    print(avg_clustering_parallelGM)

#%%

# Function to compute local clustering for a single node
def local_clustering(node):
    result = nx.clustering(G_KBM, nodes=node, weight='weight')
    return result

# List of nodes
nodes = list(G_KBM.nodes)

if __name__ == "__main__":

    # Use a pool to compute local clustering in parallel
    with Pool(processes=4) as pool:

        with tqdm(total=len(nodes)) as pbar:
                clustering_resultsKBM = []
                for result in pool.imap_unordered(local_clustering, nodes):
                    clustering_resultsKBM.append(result)
                    pbar.update(1)
        
        
    # Compute the average clustering coefficient
    avg_clustering_parallelKBM = sum(clustering_resultsKBM) / len(clustering_resultsKBM)
    print(avg_clustering_parallelKBM)

# %%
# Save the results to a text file
file_clustering = open("clustering_resultsGM.txt", "w")
with open("clustering_resultsGM.txt", "w") as file:
    for value in clustering_resultsGM:
        file_clustering.write(f"{value}\n")  # Each float on a new line
file_clustering.close()


#%%
file_clustering = open("clustering_resultsKBM.txt", "w")
with open("clustering_resultsKBM.txt", "w") as file:
    for value in clustering_resultsKBM:
        file_clustering.write(f"{value}\n")  # Each float on a new line
file_clustering.close()

#with open("clustering_resultsGM.txt", "r") as file:
#    clustering_resultsGM= [float(line.strip()) for line in file]
#    #print(clustering_resultsGM)
#file.close()

#%%
# Save as a NumPy binary file
np.save("clustering_resultsGM.npy", clustering_resultsGM)


#%%
np.save("clustering_resultsKBM.npy", clustering_resultsKBM)

#%%
# Load the NumPy binary file
clustering_resultsGM= np.load("clustering_resultsGM.npy")
clustering_resultsKBM= np.load("clustering_resultsKBM.npy")

# %%
#Histogram of clustering coefficients
t6=tm.time()

#conditions for log10: remove zeros
cl_coeff_GM=np.array(clustering_resultsGM)
cl_coeff_GM_n0=cl_coeff_GM[cl_coeff_GM>0]

plt.hist(cl_coeff_GM_n0, bins=300)
plt.xlabel("clustering coefficient")
plt.title("histogram clustering coefficient GM normalized")
plt.show()

plt.hist(np.log10(cl_coeff_GM_n0), bins=300)
plt.xlabel("log_10(clustering coefficient)")
plt.title("histogram log clustering coefficient GM normalized")
plt.show()

#%%
#Spectral analysis of matrices
#GM12878
eigenvalues_GM_norm, eigenvectors_GM_norm = LA.eigh(dataGM_normalized)
print(np.max(eigenvalues_GM_norm))

eigenvalues_GM_norm=eigenvalues_GM_norm[::-1]
eigenvectors_GM_norm=np.transpose(eigenvectors_GM_norm)
eigenvectors_GM_norm=eigenvectors_GM_norm[::-1]

#KBM7
eigenvalues_KBM_norm, eigenvectors_KBM_norm = LA.eigh(dataKBM_normalized)
print(np.max(eigenvalues_KBM_norm))

eigenvals_KBM=np.sort(eigenvalues_KBM_norm)[::-1]
eigenvectors_KBM_norm=np.transpose(eigenvectors_KBM_norm)
eigenvectors_KBM_norm=eigenvectors_KBM_norm[::-1]
#%%
#Checking that the eigenvalues and eigenvenctors found return the initial matrix

# Construct the diagonal matrix from the eigenvalues
D = np.diag(eigenvalues_GM_norm)

# Reconstruct the original matrix using the eigen decomposition formula A = PDP^-1
P = eigenvectors_GM_norm
P_inv = np.linalg.inv(P)
A = P @ D @ P_inv

#%%
#Histogram of eigenvalues
#GM12878
eigvals_GM = np.delete(eigenvalues_GM_norm, 2888)

plt.hist(eigvals_GM, bins=100)
plt.xlabel("eigenvalues_GM")
plt.title("histogram eigenvalues GM")
plt.show()

#KBM7
eigvals_KBM = np.delete(eigenvalues_KBM_norm, 2888)

plt.hist(eigvals_KBM, bins=100)
plt.xlabel("eigenvalues_KBM")
plt.title("histogram eigenvalues KBM")
plt.show()

#%%
#Spectral density comparison

plt.hist(eigvals_GM, bins=800, density = True, histtype= 'step', label='GM12878')
plt.hist(eigvals_KBM, bins=800, density = True, histtype= 'step', label='KBM7')
plt.xlim(min(min(eigenvalues_GM_norm), min(eigenvalues_KBM_norm)),100)
plt.legend(loc="best")
plt.xlabel("eigenvalues")
plt.ylabel("Density")
plt.title("SPectral density comparison")
plt.show()
#%%
#Eigenvectors component distribution

#GM12878
plt.hist(eigenvectors_GM_norm[0], bins=80)
plt.xlabel("eigenvector 1")
plt.title("eigenvector 1 component distribution GM12878")
plt.show()


plt.hist(eigenvectors_GM_norm[1], bins=80)
plt.xlabel("eigenvector 2")
plt.title("eigenvector 2 component distribution GM12878")
plt.show()

plt.hist(eigenvectors_GM_norm[19], bins=80)
plt.xlabel("eigenvector 20")
plt.title("eigenvector 20 component distribution GM12878")
plt.show()

plt.hist(eigenvectors_GM_norm[99], bins=80)
plt.xlabel("eigenvector 100")
plt.title("eigenvector 100 component distribution GM12878")
plt.show()

#%%
plt.hist(eigenvectors_KBM_norm[0], bins=80)
plt.xlabel("eigenvector 1")
plt.title("eigenvector 1 component distribution KBM7")
plt.show()

plt.hist(eigenvectors_KBM_norm[1], bins=80)
plt.xlabel("eigenvector 2")
plt.title("eigenvector 2 component distribution KBM7")
plt.show()

plt.hist(eigenvectors_KBM_norm[19], bins=80)
plt.xlabel("eigenvector 20")
plt.title("eigenvector 20 component distribution KBM7")
plt.show()

plt.hist(eigenvectors_KBM_norm[99], bins=80)
plt.xlabel("eigenvector 100")
plt.title("eigenvector 100 component distribution KBM7")
plt.show()

#%%
#modify indexing to locate chromoseomes

def clean_indexing(matrix):
    #add Y indexes corresponding to the range 2893-2952
    ind=list(range(2893,2953))
    for i in range(matrix.shape[0]):
        if np.all(matrix[i] == 0) and i not in ind:
            ind.append(i)
    return ind

ind_GM=clean_indexing(dataGM0)
ind_KBM=clean_indexing(dataKBM0)

#%%
#modify indexing to locate chromoseomes

ind_chr=[]
dfchr=pd.read_csv(dir_data+"metadata_GM12878_KBM7.csv",header=0)

#create a dictionary with the indexes of the chromosomes
for i in range(len(dfchr)):
    ind_chr.append(list(range(dfchr["start"][i],dfchr["end"][i]+1)))
    
chro = dict(zip(dfchr["chr"], ind_chr))  

#remove the indexes of the Y chromosome and the isolated nodes
for key, value_list in chro.items():
    chro[key] = [item for item in value_list if item not in ind_GM]

#substitute old values with new ones starting from zero and increasing
a=0
for key, value in chro.items():
    if len(value)!=0:
        value=list(range(a,a+len(value)))
        a+=len(value)  
        chro[key]=value

# %%
#Eigenvectors analysis
#plotting the chromosomes for some eigenvalues

#first eigenvector and component graph
#GM12878
#lambda1_GM = eigenvectors_GM_norm[0]

plt.plot(eigenvectors_GM_norm[0])
plt.show()

plt.plot(eigenvectors_GM_norm[0])
plt.fill_between(x=range(2889) ,y1=eigenvectors_GM_norm[0])
plt.xlabel("eigenvector 1")
plt.title("filled eigenvector 1 component distribution GM12878")
plt.show()

plt.plot(eigenvectors_GM_norm[8])
plt.fill_between(x=range(2889) ,y1=eigenvectors_GM_norm[8])
plt.xlabel("eigenvector 9")
plt.title("filled eigenvector 9 component distribution GM12878")
plt.show()

plt.plot(eigenvectors_GM_norm[14])
plt.fill_between(x=range(2889) ,y1=eigenvectors_GM_norm[14])
plt.xlabel("eigenvector 15")
plt.title("filled eigenvector 15 component distribution GM12878")
plt.show()

# %%
#KBM7
#lambda1_KBM = eigenvectors_KBM_norm[0]

plt.plot(eigenvectors_KBM_norm[0])
plt.show()

plt.plot(eigenvectors_KBM_norm[0])
plt.fill_between(x=range(2889) ,y1=eigenvectors_KBM_norm[0])
plt.xlabel("eigenvector 1")
plt.title("filled eigenvector 1 component distribution GM12878")
plt.show()

plt.plot(eigenvectors_KBM_norm[8])
plt.fill_between(x=range(2889) ,y1=eigenvectors_KBM_norm[8])
plt.xlabel("eigenvector 9")
plt.title("filled eigenvector 9 component distribution GM12878")
plt.show()

plt.plot(eigenvectors_KBM_norm[14])
plt.fill_between(x=range(2889) ,y1=eigenvectors_KBM_norm[14])
plt.xlabel("eigenvector 15")
plt.title("filled eigenvector 15 component distribution GM12878")
plt.show()

# %%
#plotting the chromosomes for some eigenvalues with sections for each chromosome

cmap = plt.get_cmap('nipy_spectral')
colors = [cmap(i) for i in np.linspace(0, 1, 23)]

#chr_names=list(chro.keys())
#GM12878
for i, color in enumerate(colors, start=0):
    r=chro[dfchr["chr"][i]]
    #r=chro[chr_names[i]]
    vals=eigenvectors_GM_norm[0][r]
    plt.plot(r ,vals, color=color)  #all chromosomes starts at 0
    plt.fill_between(x=r ,y1=vals, color=color)
    plt.xlabel("eigenvector 1")
    plt.title("filled eigenvector 1 component distribution GM12878")
plt.show()
#NOTE: missing last chromosome

#%%
#IPR

#GM12878
IPR_GM = np.sum(eigenvectors_GM_norm**4, axis=1)

#KBM7
IPR_KBM = np.sum(eigenvectors_KBM_norm**4, axis=1)

#%%
#IPR comparison

#IPR values of the first 25 eigenvectors of GM12878 and KBM7
plt.scatter(list(range(1,26)),IPR_GM[:25], s=5, color="r", label="GM12878")
plt.plot(list(range(1,26)),IPR_GM[:25],lw=0.1, color="r")
plt.scatter(list(range(1,26)),IPR_KBM[:25], s=5, color="b", label="KBM7")
plt.plot(list(range(1,26)),IPR_KBM[:25],lw=0.1, color="b")
plt.xlabel("eigenvectors")
plt.ylabel("IPR")
plt.title("IPR comparison GM12878 and KBM7 - first 25 eigenvectors")
plt.legend(loc="best")
plt.show()

# IPR values of all eigenvectors and cuts in 3 ranges for better visualization
l_ranges=[[0,2889, "all"],[0,199, "1-200"],[899,1249, "900-1250"],[2749,2889, "2750-2890"]]
for i in l_ranges:
    plt.semilogy(list(range(i[0]+1,i[1]+1)), IPR_GM[i[0]:i[1]],lw=0.6, ls='-', color="r", label="GM12878")
    plt.semilogy(list(range(i[0]+1,i[1]+1)), IPR_KBM[i[0]:i[1]],lw=0.6, ls='-', color="b", label="KBM7")
    plt.xlabel("eigenvectors")
    plt.ylabel("log10(IPR)")
    plt.title(f"IPR comparison GM12878 and KBM7 - {i[2]} eigenvectors")
    plt.legend(loc="best")
    plt.show()
    
#%%
# Computing essential matrix for a different number of eigenvectors and eigenvalues

def compute_essential_matrix(eigval, eigvec, N):
    # ordering in an absolute descending order 
    idx = np.argsort(np.abs(eigval))[::-1]
    # selecting top N eigenvalues and eigenvectors
    eigval_topN = eigval[idx][:N]
    eigvec_topN = eigvec[idx][:N]
    
    A_ess = np.zeros((len(eigval), len(eigval)))
    
    # compute the essential matrix using the formula
    for n in range(N):
        k_n = eigval_topN[n]
        a_n = eigvec_topN[n]
        
        # compute the contribution of eigenvector n to the essential matrix
        A_ess += k_n * np.outer(a_n, a_n)  
    
    #creating corresponding plot
    
    plt.imshow(A_ess, cmap='plasma', interpolation='none')
    plt.colorbar()
    plt.title('Adjacency Matrix KBM7')
    plt.show()
    return 0

compute_essential_matrix(eigenvalues_GM_norm, eigenvectors_GM_norm, 10)
compute_essential_matrix(eigenvalues_GM_norm, eigenvectors_GM_norm, 15)
compute_essential_matrix(eigenvalues_GM_norm, eigenvectors_GM_norm, 20)
compute_essential_matrix(eigenvalues_GM_norm, eigenvectors_GM_norm, 25)
