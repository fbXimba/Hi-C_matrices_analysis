#%%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
#NOTE: matrix creation work in progress. PROJECT FRANKENSTEIN

#%%
#directories and data import

#get current ditectory and create Data , HMEC and NHRK directories
dir_home=os.getcwd()

dir_data=dir_home+"/Data/"
dir_HMEC=dir_data+"raw_HMEC_1Mb/"
dir_NHEK=dir_data+"raw_NHEK_1Mb/"

def chromosomes():
    #
    chromosomes = pd.read_csv(dir_data + "chr_sizes_HMEC_NHEK.txt", sep='\t',header=None)
    chromosomes=chromosomes.to_numpy()  
    chromosomes[:,1] = (chromosomes[:,1] / 10**6).astype(int) + 1
    indexes = np.cumsum(chromosomes[:,1])
    #format equal to the one in the GM12878 and KBM7 metadata + chromosome size
    chromosomes= np.column_stack((chromosomes[:,0], indexes-chromosomes[:,1], indexes, chromosomes[:,1]))

    return chromosomes

def matrix_construction(file_list, chromosomes, file_paths):
    # List all files in the folder
    #file_list = [f for f in os.listdir(dir_HMEC)] 

    matrix = np.zeros((3113,3113))  

    for file in file_list:
        #df = pd.read_csv(file_path, delimiter='\t', header=None)
        #df.columns = ['init', 'end', 'value']
        #df['init'] = df['init'] / 1_000_000
        #df['end'] = df['end'] / 1_000_000
        file_path = file_paths + file
        data = np.loadtxt(file_path)
        data = data.astype(int)
        data[:,:2] = data[:,:2] / 1_000_000
        parts = file.split('_')

        # Extract the numbers
        cr1 = parts[0]
        cr2 = parts[1]
        if len(cr2) > 2 :
            cr2 = cr1
        else:
            cr2 = "chr" + cr2

        move_x = chromosomes[chromosomes[:,0] == cr1][0,1]
        move_y = chromosomes[chromosomes[:,0] == cr2][0,1]

        for row in data: 
            i = row[0] + move_x
            j = row[1]+ move_y
            if i == j:
                matrix[i, j] = 0
            else:
                matrix[i, j] = row[2]
                matrix[j, i] = row[2]

    return matrix