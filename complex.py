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
import construction_site as cs
import leidenalg
import igraph as ig

#NOTE: This code is currently a work in progress in its initial stages. NO BITCHING.

#%%
#directories and data import/creation
t0=tm.time()

#get current ditectory and create Data , HMEC and NHRK directories
dir_home=os.getcwd()
dir_data=dir_home+"/Data/"
dir_HMEC=dir_data+"raw_HMEC_1Mb/"
dir_NHEK=dir_data+"raw_NHEK_1Mb/"

#read data GM12878 and KBM7 and create matrices
#GM12878 healthy cells
dfGM=pd.read_csv(dir_data+"raw_GM12878_1Mb.csv",header=None)
dataGM0=dfGM.to_numpy()

#KBM7 aberrant cells
dfKBM=pd.read_csv(dir_data+"raw_KBM7_1Mb.csv",header=None)
dataKBM0=dfKBM.to_numpy()

#read data of HMEC and NHEK and create matrices
chromosomes_HMEC_NHEK = cs.chromosomes()

#HMEC healthy cells
file_list_HMEC = [f for f in os.listdir(dir_HMEC)]

dataHMEC0=cs.matrix_construction(file_list_HMEC, chromosomes_HMEC_NHEK, dir_HMEC)

#NHEK healthy cells
file_list_NHEK = [f for f in os.listdir(dir_NHEK)]

dataNHEK0=cs.matrix_construction(file_list_NHEK, chromosomes_HMEC_NHEK, dir_NHEK)

#%%
#Data cleaning:
t1=tm.time()

#removing Y crhomosome + isolated nodes from data
#Y = list(range(2892,2951))

#GM12878
dataGM=utils.data_cleaning(dataGM0)

#KBM7
dataKBM=utils.data_cleaning(dataKBM0)

#HMEC
dataHMEC=utils.data_cleaning(dataHMEC0)

#NHEK
dataNHEK=utils.data_cleaning(dataNHEK0)

#normalization = log_scale for better visualization
#log1p does log10(1+x) in order to avoid problems where there's a 0

#GM12878
dataGM_normalized = np.log1p(dataGM)

#KBM7
dataKBM_normalized = np.log1p(dataKBM)

#HMEC
dataHMEC_normalized = np.log1p(dataHMEC)

#NHEK
dataNHEK_normalized = np.log1p(dataNHEK)

#%%
#Creation of graphs
#from array (faster than from dataframe)
t2=tm.time()

#GM12878
G_GM= nx.from_numpy_array(dataGM_normalized)

#KBM7
G_KBM= nx.from_numpy_array(dataKBM_normalized)

#HMEC
G_HMEC= nx.from_numpy_array(dataHMEC_normalized)

#NHEK
G_NHEK= nx.from_numpy_array(dataNHEK_normalized)

#%%
#Strength of nodes
t3=tm.time()

#GM12878
str_GM=utils.stregth(G_GM)
print(np.min(str_GM), np.max(str_GM), np.mean(str_GM)) #check min and max values

#KBM7
str_KBM=utils.stregth(G_KBM)
print(np.min(str_KBM), np.max(str_KBM), np.mean(str_KBM)) 

#HMEC
str_HMEC=utils.stregth(G_HMEC)
print(np.min(str_HMEC), np.max(str_HMEC), np.mean(str_HMEC))

#NHEK
str_NHEK=utils.stregth(G_NHEK)
print(np.min(str_NHEK), np.max(str_NHEK), np.mean(str_NHEK))

#%%
#Histograms of strength
t4=tm.time()

#GM12878
plt.hist(str_GM, bins=400)
plt.xlabel("log_10(strength)")
plt.title("strength histogram GM normalized")
plt.show()

#KBM7
plt.hist(str_KBM, bins=400)
plt.xlabel("log_10(strength)")
plt.title("strength histogram KBM normalized")
plt.show()

#HMEC
plt.hist(str_HMEC, bins=400)
plt.xlabel("log_10(strength)")
plt.title("strength histogram HMEC normalized")
plt.show()

#NHEK
plt.hist(str_NHEK, bins=400)
plt.xlabel("log_10(strength)")
plt.title("strength histogram NHEK normalized")
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
#HMEC
t5s=tm.time()

clustering_results_HMEC,avg_clustering_parallel_HMEC=utils.parallel_clustering(G_HMEC)

file_clustering = open("clustering_resultsHMEC.txt", "w")
with open("clustering_resultsHMEC.txt", "w") as file:
    for value in clustering_results_HMEC:
        file_clustering.write(f"{value}\n")  # Each float on a new line
file_clustering.close()

np.save("clustering_resultsHMEC.npy", clustering_results_HMEC)

#%%
#NHEK
t5t=tm.time()

clustering_results_NHEK,avg_clustering_parallel_NHEK=utils.parallel_clustering(G_NHEK)

file_clustering = open("clustering_resultsNHEK.txt", "w")
with open("clustering_resultsNHEK.txt", "w") as file:
    for value in clustering_results_NHEK:
        file_clustering.write(f"{value}\n")  # Each float on a new line
file_clustering.close()

np.save("clustering_resultsNHEK.npy", clustering_results_NHEK)

#%%

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

#%%
#NOTE: NON ANCORA OTTENUTI
clustering_results_HMEC= np.load("clustering_resultsHMEC.npy")
clustering_results_NHEK= np.load("clustering_resultsNHEK.npy")

# %%
#Histogram of clustering coefficients

#GM12878
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

eigenvalues_KBM_norm=np.sort(eigenvalues_KBM_norm)[::-1]
eigenvectors_KBM_norm=np.transpose(eigenvectors_KBM_norm)
eigenvectors_KBM_norm=eigenvectors_KBM_norm[::-1]

#HMEC
eigenvalues_HMEC_norm, eigenvectors_HMEC_norm = LA.eigh(dataHMEC_normalized)
print(np.max(eigenvalues_HMEC_norm))

eigenvalues_HMEC_norm=np.sort(eigenvalues_HMEC_norm)[::-1]
eigenvectors_HMEC_norm=np.transpose(eigenvectors_HMEC_norm)
eigenvectors_HMEC_norm=eigenvectors_HMEC_norm[::-1]

#NHEK
eigenvalues_NHEK_norm, eigenvectors_NHEK_norm = LA.eigh(dataNHEK_normalized)
print(np.max(eigenvalues_NHEK_norm))

eigenvalues_NHEK_norm=np.sort(eigenvalues_NHEK_norm)[::-1]
eigenvectors_NHEK_norm=np.transpose(eigenvectors_NHEK_norm)
eigenvectors_NHEK_norm=eigenvectors_NHEK_norm[::-1]

#%%
#Histogram of eigenvalues
#GM12878
eigvals_GM = np.delete(eigenvalues_GM_norm, 0)

plt.hist(eigvals_GM[eigvals_GM<500], bins=100)
plt.xlabel("eigenvalues_GM")
plt.title("histogram eigenvalues GM")
plt.show()

#KBM7
eigvals_KBM = np.delete(eigenvalues_KBM_norm, 0)

plt.hist(eigvals_KBM[eigvals_KBM<500], bins=100)
plt.xlabel("eigenvalues_KBM")
plt.title("histogram eigenvalues KBM")
plt.show()

#HMEC
eigvals_HMEC = np.delete(eigenvalues_HMEC_norm, 0)

plt.hist(eigvals_HMEC[eigvals_HMEC<500], bins=100)
plt.xlabel("eigenvalues_HMEC")
plt.title("histogram eigenvalues HMEC")
plt.show()

#NHEK
eigvals_NHEK = np.delete(eigenvalues_NHEK_norm, 0)

plt.hist(eigvals_NHEK[eigvals_NHEK<500], bins=100)
plt.xlabel("eigenvalues_NHEK")
plt.title("histogram eigenvalues NHEK")
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


plt.hist(eigvals_GM[eigvals_GM<100], bins=150, density = True, histtype= 'step', color="r", label='GM12878')
plt.hist(eigvals_KBM[eigvals_KBM<100], bins=150, density = True, histtype= 'step', color="b",label='KBM7')
plt.hist(eigvals_HMEC[eigvals_HMEC<100], bins=150, density = True, histtype= 'step', color="g",label='HMEC')
plt.hist(eigvals_NHEK[eigvals_NHEK<100], bins=150, density = True, histtype= 'step', color="y",label='NHEK')
plt.legend(loc="best")
plt.xlabel("eigenvalues")
plt.ylabel("Density")
plt.title("Spectral density comparison")
plt.show()

#%%
#Eigenvectors component distribution
#GM12878 and KBM7, eigenvectors 1, 2, 20 and 100

for eigenvectors in [[eigenvectors_GM_norm,"GM12878"], [eigenvectors_KBM_norm,"KBM7"], [eigenvectors_HMEC_norm,"HMEC"], [eigenvectors_NHEK_norm,"NHEK"]]:
    #GM12878 and KBM7
    for n in [0,1,19,99]:
        #eigenvectors 1, 2, 20 and 100
        plt.hist(eigenvectors[0][n], bins=80)
        plt.title(f"{eigenvectors[1]} eigenvector {n+1} distribution")
        plt.show()

#%%
#modify indexing to locate chromosomes for eigenvectors analysis
t8=tm.time()
#NOTE: FINIRE DI IMPLEMENTARE PER HMEC E NHEK
#GM12878 and KBM7 Y=2893-2952 (from 1)

dfchr=pd.read_csv(dir_data+"metadata_GM12878_KBM7.csv",header=0)

chro1=utils.dict_indexing(dataGM0,dfchr["chr"],dfchr["start"],dfchr["end"])

chro2=utils.dict_indexing(dataHMEC0,chromosomes_HMEC_NHEK[:,0],chromosomes_HMEC_NHEK[:,1],chromosomes_HMEC_NHEK[:,2])
   
     
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
            r=chro1[dfchr["chr"][i]]
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

#HMEC
IPR_HMEC = np.sum(eigenvectors_HMEC_norm**4, axis=1)

#NHEK
IPR_NHEK = np.sum(eigenvectors_NHEK_norm**4, axis=1)

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

#IPR values of the first 25 eigenvectors of GM12878, KBM7, HMEC and NHEK
plt.scatter(list(range(1,26)),IPR_GM[:25], s=5, color="r", label="GM12878")
plt.plot(list(range(1,26)),IPR_GM[:25],lw=0.1, color="r")
plt.scatter(list(range(1,26)),IPR_KBM[:25], s=5, color="b", label="KBM7")
plt.plot(list(range(1,26)),IPR_KBM[:25],lw=0.1, color="b")
plt.scatter(list(range(1,26)),IPR_HMEC[:25], s=5, color="g", label="HMEC")
plt.plot(list(range(1,26)),IPR_HMEC[:25],lw=0.1, color="g")
plt.scatter(list(range(1,26)),IPR_NHEK[:25], s=5, color="y", label="NHEK")
plt.plot(list(range(1,26)),IPR_NHEK[:25],lw=0.1, color="y")
plt.xlabel("eigenvectors")
plt.ylabel("IPR")
plt.title("IPR comparison GM12878, KBM7, HMEC and NHEK - first 25 eigenvectors")
plt.legend(loc="best")
plt.show()
    
#IPR values of all eigenvectors and cuts in 3 ranges for better visualization
l_ranges=[[0,2888, "all"],[0,199, "1-200"],[899,1249, "900-1250"],[2749,2888, "2750-2890"]]
for i in l_ranges:
    plt.semilogy(list(range(i[0]+1,i[1]+1)), IPR_GM[i[0]:i[1]],lw=0.6, ls='-', color="r", label="GM12878")
    plt.semilogy(list(range(i[0]+1,i[1]+1)), IPR_KBM[i[0]:i[1]],lw=0.6, ls='-', color="b", label="KBM7")
    plt.semilogy(list(range(i[0]+1,i[1]+1)), IPR_HMEC[i[0]:i[1]],lw=0.6, ls='-', color="g", label="HMEC")
    plt.semilogy(list(range(i[0]+1,i[1]+1)), IPR_NHEK[i[0]:i[1]],lw=0.6, ls='-', color="y", label="NHEK")
    plt.xlabel("eigenvectors")
    plt.ylabel("log10(IPR)")
    plt.title(f"IPR comparison GM12878 and KBM7 - {i[2]} eigenvectors")
    plt.legend(loc="best")
    plt.show()

#%%
#Adjacency matrix
t10=tm.time()

#Adjacency matrix visualization
utils.plot_adjacency_matrix(dataGM_normalized, "GM12878", "all", 'plasma')
utils.plot_adjacency_matrix(dataKBM_normalized, "KBM7", "all", 'plasma')
utils.plot_adjacency_matrix(dataHMEC_normalized, "HMEC", "all", 'plasma')
utils.plot_adjacency_matrix(dataNHEK_normalized, "NHEK", "all", 'plasma')

#%%
#Computing essential matrix for a different number of eigenvectors and eigenvalues

#GM12878
utils.compute_essential_matrix(eigenvalues_GM_norm, eigenvectors_GM_norm, 10, "GM12878")
utils.compute_essential_matrix(eigenvalues_GM_norm, eigenvectors_GM_norm, 15, "GM12878")
utils.compute_essential_matrix(eigenvalues_GM_norm, eigenvectors_GM_norm, 20, "GM12878")
utils.compute_essential_matrix(eigenvalues_GM_norm, eigenvectors_GM_norm, 25, "GM12878")

#KBM7
utils.compute_essential_matrix(eigenvalues_KBM_norm, eigenvectors_KBM_norm, 10, "KBM7")
utils.compute_essential_matrix(eigenvalues_KBM_norm, eigenvectors_KBM_norm, 15, "KBM7")
utils.compute_essential_matrix(eigenvalues_KBM_norm, eigenvectors_KBM_norm, 20, "KBM7")
utils.compute_essential_matrix(eigenvalues_KBM_norm, eigenvectors_KBM_norm, 25, "KBM7")
utils.compute_essential_matrix(eigenvalues_KBM_norm, eigenvectors_KBM_norm, 30, "KBM7")

#HMEC
utils.compute_essential_matrix(eigenvalues_HMEC_norm, eigenvectors_HMEC_norm, 10, "HMEC")
utils.compute_essential_matrix(eigenvalues_HMEC_norm, eigenvectors_HMEC_norm, 15, "HMEC")
utils.compute_essential_matrix(eigenvalues_HMEC_norm, eigenvectors_HMEC_norm, 20, "HMEC")
utils.compute_essential_matrix(eigenvalues_HMEC_norm, eigenvectors_HMEC_norm, 25, "HMEC")
utils.compute_essential_matrix(eigenvalues_HMEC_norm, eigenvectors_HMEC_norm, 35, "HMEC")

#NHEK
utils.compute_essential_matrix(eigenvalues_NHEK_norm, eigenvectors_NHEK_norm, 10, "NHEK")
utils.compute_essential_matrix(eigenvalues_NHEK_norm, eigenvectors_NHEK_norm, 15, "NHEK")
utils.compute_essential_matrix(eigenvalues_NHEK_norm, eigenvectors_NHEK_norm, 20, "NHEK")
utils.compute_essential_matrix(eigenvalues_NHEK_norm, eigenvectors_NHEK_norm, 25, "NHEK")
utils.compute_essential_matrix(eigenvalues_NHEK_norm, eigenvectors_NHEK_norm, 37, "NHEK")

# %%
#Thresholding for binary matrix

#GM12878
dataGM_bin=utils.thresholding(dataGM_normalized, 4.7)
utils.plot_adjacency_matrix(dataGM_bin, "GM12878", "threshold 4.7", 'plasma')

#KBM7
dataKBM_bin=utils.thresholding(dataKBM_normalized, 4.7)
utils.plot_adjacency_matrix(dataKBM_bin, "KBM7", "threshold 4.7", 'plasma')

#HMEC
dataHMEC_bin=utils.thresholding(dataHMEC_normalized, 4.1)
utils.plot_adjacency_matrix(dataHMEC_bin, "HMEC", "threshold 4.1", 'plasma')

#NHEK
dataNHEK_bin=utils.thresholding(dataNHEK_normalized, 4.3)
utils.plot_adjacency_matrix(dataNHEK_bin, "NHEK", "threshold 4.3", 'plasma')

# %%
#binary graphs and clustering

#binary graphs creation
Gbin_GM= ig.Graph.Adjacency(dataGM_bin, mode="undirected")
Gbin_KBM7= ig.Graph.Adjacency(dataKBM_bin, mode="undirected")
Gbin_HMEC= ig.Graph.Adjacency(dataHMEC_bin, mode="undirected")
Gbin_NHEK= ig.Graph.Adjacency(dataNHEK_bin, mode="undirected")

#clustering with Leiden algorithm

#GM12878
part_GM = leidenalg.find_partition(Gbin_GM, leidenalg.ModularityVertexPartition)
mod_GM = part_GM.modularity

#KBM7
part_KBM = leidenalg.find_partition(Gbin_KBM7, leidenalg.ModularityVertexPartition)
mod_KBM = part_KBM.modularity

#HMEC
part_HMEC = leidenalg.find_partition(Gbin_HMEC, leidenalg.ModularityVertexPartition)
mod_HMEC = part_HMEC.modularity

#NHEK
part_NHEK = leidenalg.find_partition(Gbin_NHEK, leidenalg.ModularityVertexPartition)
mod_NHEK = part_NHEK.modularity

print(mod_GM, mod_KBM, mod_HMEC, mod_NHEK)
#%%
#Visualization of community adjacency matrix

community_GM = list(part_GM)
community_KBM = list(part_KBM)
community_HMEC = list(part_HMEC)
community_NHEK = list(part_NHEK)

#GM12878
A = utils.cluster_view(community_GM,"GM12878")
#possibility to confront the cluster matrix with the original adjacency matri
utils.plot_adjacency_matrix(dataGM_normalized, "GM12878", "all", 'plasma')

#KBM7
B = utils.cluster_view(community_KBM, "KBM7")
#utils.plot_adjacency_matrix(dataKBM_normalized, "KBM7", "all", 'plasma')

#HMEC
utils.cluster_view(community_HMEC, "HMEC")
#utils.plot_adjacency_matrix(dataHMEC_normalized, "HMEC", "all", 'plasma')

#NHEK
utils.cluster_view(community_NHEK, "NHEK")
#utils.plot_adjacency_matrix(dataNHEK_normalized, "NHEK", "all", 'plasma')

#Visualization of differences between clustering matrices of GM12878 and KBM7
diff_M = np.subtract(B, A)

utils.plot_adjacency_matrix(diff_M, "KBM7 - GM12878", "all", 'gnuplot2')

#%%
#scatter plot of clusters 
#clusters visually divided and different colors for each chromosome
#necesseary to create chromosomes dictionary cho1 and chro2: cell at line 362

utils.cluster_scatter(part_GM, chro1, dfchr["chr"], "GM12878")
utils.cluster_scatter(part_KBM, chro1, dfchr["chr"], "KBM7")
utils.cluster_scatter(part_HMEC, chro2, chromosomes_HMEC_NHEK[:,0], "HMEC")
utils.cluster_scatter(part_NHEK, chro2, chromosomes_HMEC_NHEK[:,0], "NHEK")