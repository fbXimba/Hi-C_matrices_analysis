#%%
#libaries imported
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from scipy.stats import pearsonr
#import math
#import random as rnd
import networkx as nx
import time as tm
#from multiprocessing import Pool
#from tqdm import tqdm 
from numpy import linalg as LA
import utils as utils

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
#Data cleaning:
t1=tm.time()

#removing Y crhomosome + isolated nodes from data
#Y = list(range(2892,2951))

#GM12878
dataGM=utils.data_cleaning(dataGM0)

#KBM7
dataKBM=utils.data_cleaning(dataKBM0)

#normalization = log_scale for better visualization
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
#Strength of nodes
t3=tm.time()

#GM12878
str_GM=utils.stregth(G_GM)
print(np.min(str_GM), np.max(str_GM), np.mean(str_GM)) #check min and max values

#KBM7
str_KBM=utils.stregth(G_KBM)
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

#GM12878
t5=tm.time()

clustering_results_GM,avg_clustering_parallel_GM=utils.parallel_clustering(G_GM)

# Save the results to a text file
file_clustering = open("clustering_resultsGM.txt", "w")
with open("clustering_resultsGM.txt", "w") as file:
    for value in clustering_results_GM:
        file_clustering.write(f"{value}\n")  # Each float on a new line
file_clustering.close()

# Save as a NumPy binary file
np.save("clustering_resultsGM.npy", clustering_results_GM)

#%%
#KBM7
t5p=tm.time()

clustering_results_KBM,avg_clustering_parallel_KBM=utils.parallel_clustering(G_KBM)

file_clustering = open("clustering_resultsKBM.txt", "w")
with open("clustering_resultsKBM.txt", "w") as file:
    for value in clustering_results_KBM:
        file_clustering.write(f"{value}\n")  # Each float on a new line
file_clustering.close()

np.save("clustering_resultsKBM.npy", clustering_results_KBM)

# %%

# Load the results from the text file

#with open("clustering_resultsGM.txt", "r") as file:
#    clustering_resultsGM= [float(line.strip()) for line in file]
#    #print(clustering_resultsGM)
#file.close()

#with open("clustering_resultsKBM.txt", "r") as file:
#    clustering_resultsKBM= [float(line.strip()) for line in file]
#    #print(clustering_resultsKBM)
#file.close()

#%%

# Load the NumPy binary file
clustering_results_GM= np.load("clustering_resultsGM.npy")
clustering_results_KBM= np.load("clustering_resultsKBM.npy")

# %%
#Histogram of clustering coefficients

#conditions for log10: remove zeros
cl_coeff_GM=np.array(clustering_results_GM)
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
t6=tm.time()

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
t7=tm.time()

plt.hist(eigvals_GM[eigvals_GM<100], bins=150, density = True, histtype= 'step', label='GM12878')
plt.hist(eigvals_KBM[eigvals_KBM<100], bins=150, density = True, histtype= 'step', label='KBM7')
plt.legend(loc="best")
plt.xlabel("eigenvalues")
plt.ylabel("Density")
plt.title("Spectral density comparison")
plt.show()
#%%
#Eigenvectors component distribution
#GM12878 and KBM7, eigenvectors 1, 2, 20 and 100

for eigenvectors in [[eigenvectors_GM_norm,"GM12878"], [eigenvectors_KBM_norm,"KBM7"]]:
    #GM12878 and KBM7
    for n in [0,1,19,99]:
        #eigenvectors 1, 2, 20 and 100
        plt.hist(eigenvectors[0][n], bins=80)
        plt.title(f"{eigenvectors[1]} eigenvector {n+1} distribution")
        plt.show()

#%%
#modify indexing to locate chromosomes for eigenvectors analysis
t8=tm.time()

ind_GM=utils.clean_indexing(dataGM0)
ind_KBM=utils.clean_indexing(dataKBM0)

ind_chr=[]
dfchr=pd.read_csv(dir_data+"metadata_GM12878_KBM7.csv",header=0)

#create a dictionary with the indexes of the chromosomes
for i in range(len(dfchr)):
    ind_chr.append(list(range(dfchr["start"][i],dfchr["end"][i]+1)))
    
chro = dict(zip(dfchr["chr"], ind_chr)) 

#remove the indexes of the Y chromosome and the isolated nodes
a=0
for key, value in chro.items():
    chro[key] = [item for item in value if item not in ind_GM]
    #substitute old values with new ones starting from zero and increasing
    if len(value)!=0:
        value=list(range(a,a+len(value)))
        a+=len(value)  
        chro[key]=value    

# %%
#Eigenvectors analysis
#GM12878 and KBM7, eigenvectors 1, 9 and 15
#plotting the chromosomes for some eigenvalues with sections for each chromosome

cmap = plt.get_cmap('nipy_spectral')
colors = [cmap(i) for i in np.linspace(0, 1, 23)]

for eigenvectors in [[eigenvectors_GM_norm,"GM12878"], [eigenvectors_KBM_norm,"KBM7"]]:
    #GM12878 and KBM7
    for n in [0,8,14]:
        #eigenvectors 1, 9 and 15
        for i, color in enumerate(colors, start=0):
            #each chromosome
            r=chro[dfchr["chr"][i]]
            vals=(eigenvectors[0])[n][r]
            plt.plot(r ,vals, color=color)  #all chromosomes starts at 0
            plt.fill_between(x=r ,y1=vals, color=color)
            plt.xlabel("eigenvector component")
            plt.title(f"{eigenvectors[1]} eigenvector {n+1} components")
        plt.show()

#%%
#IPR
t9=tm.time()

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

#IPR values of all eigenvectors and cuts in 3 ranges for better visualization
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
#Adjacency matrix
t10=tm.time()

#Adjacency matrix visualization
utils.plot_adjacency_matrix(dataGM_normalized, "GM12878", "all")
utils.plot_adjacency_matrix(dataKBM_normalized, "KBM7", "all")

#%%
#Computing essential matrix for a different number of eigenvectors and eigenvalues

#GM12878
utils.compute_essential_matrix(eigenvalues_GM_norm, eigenvectors_GM_norm, 10, "GM12878")
utils.compute_essential_matrix(eigenvalues_GM_norm, eigenvectors_GM_norm, 15, "GM12878")
utils.compute_essential_matrix(eigenvalues_GM_norm, eigenvectors_GM_norm, 20, "GM12878")
utils.compute_essential_matrix(eigenvalues_GM_norm, eigenvectors_GM_norm, 25, "GM12878")

# %%
