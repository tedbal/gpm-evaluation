# utility functions for the analysis
# dependencies
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle as pkl
from IPython.display import HTML
from tqdm import tqdm

# set the data base path for the work
DATA_BASE_PATH = os.path.abspath("../data/")


# read data file
def read_data(filename):
    with open(os.path.join(DATA_BASE_PATH, filename), 'rb') as fi:
        data = np.array(pkl.load(fi))

    # add the concentrations from the five sources together to analyze as one
    data = np.add.reduce(data, axis=0)
    data = np.flip(data, axis=1) # flip it along the downstream axis is greater

    return data

# calculate coordinates

# calculate the meanfree (fluctuating) field
def calculate_meanfree(data)
    cbar = np.mean(data, axis=0) # mean field
    cf = np.zeros_like(data) # initialize the meanfree field
    for i in range(data.shape[0]):
        cf[i, :, :] = np.subtract(data[i, :, :], cbar)

    return cf, cbar


# check if everything is working
if __name__ == "__main__":
    # read in a data file
    read_data(
