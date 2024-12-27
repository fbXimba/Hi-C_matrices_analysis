## Hi-C matrices analysis

In this project we applied some network analysis functions to Hi-C matrices in order to establish if network parameters can be good indicators of possible genome's aberrations or translocations.
We focused on some networks' measurements such as strength, clustering coefficients, eigenvalues and eigenvectors distribution, Inverse Participation Ratio and clustering. Inside the program there is also a function to reconstruct and visualize the adjacency matric starting from raw data.


## Install and run the code

From _terminal_ move into the desired folder and clone this repository using the following command:

```shell
git clone https://github.com/fbXimba/Complex_Networks.git
```

Once the github repository is cloned the user will have access to all files needed. 
The code can be run by the _complex.py_ file, we recommend to avoid running the _Average_clustering_coefficient_ section if not interested because it's the most computationally heavy section of the program and it could take the computer some time to run through it.
The results of this function applied to the analyzed matrices can be found in the _clustering_results_ text file and  can be accessed using the proper function present in the code.

## Required Python packages

[python](https://www.python.org)
[os](https://docs.python.org/3/library/os.html)
[pandas](https://pandas.pydata.org/)
[numpy](https://numpy.org)
[matplotlib.pyplot](https://matplotlib.org/stable/api/pyplot_summary.html)
[networkx](https://networkx.org/documentation/stable/install.html)
[time](https://docs.python.org/3/library/time.html)
[multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
[tqdm](https://tqdm.github.io/)
[leidenalg](https://github.com/vtraag/leidenalg)
[igraph](https://python.igraph.org/en/latest/install.html)

## Repository structure

The repository contains the following files:

- _complex.py_ : runs the program and saves the plots in the Plots folder
- _construction_site.py_ : definition of the function used for the adjacency matrix constrtuction
- _utils.py_ : definition of all function used in _complex.py_
- _Data_ : folder containing all data used in this work
- _Plots_ : folder containing obtained plots
 