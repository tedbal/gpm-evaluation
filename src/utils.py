# utility functions for the analysis
# dependencies
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle as pkl
from IPython.display import HTML
from tqdm import tqdm

from gpmeval import GaussianPlume2D

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

# calculate the meanfree (fluctuating) field
def calculate_meanfree(data):
    cbar = np.mean(data, axis=0) # mean field
    cf = np.zeros_like(data) # initialize the meanfree field
    for i in range(data.shape[0]):
        cf[i, :, :] = np.subtract(data[i, :, :], cbar)

    return cf, cbar

# find the source in the mean data
def find_source(cbar):
    return np.unravel_index(np.argmax(cbar, axis=None), cbar.shape)

# create a function to calculate the downwind variance from a meanfield
def calculate_downwind_variance(pts, cbar):
    # first, calculate the xi and eta
    isrc_x1, isrc_x2 = find_source(cbar)
    xi, eta = pts[0] - pts[0][isrc_x1, isrc_x2], pts[1] - pts[1][isrc_x1, isrc_x2]

    # now initialize the results and begin iteration
    sigmas = np.zeros(cbar.shape[0])
    for i in range(cbar.shape[0]):
        # get the tranverse cross section at point i
        C = cbar[i, :]

        # integrate the mass and apply some conditions for less than zero mass
        mass = np.trapezoid(C, eta[i,:])
        if mass <= 0:
            sigmas[i] = np.nan
            continue

        # calculate the center of the mass and the variance
        center = np.trapezoid(eta[i, :] * C, eta[i, :]) / mass
        variance = np.trapezoid((eta[i, :] - center)**2 * C, eta[i, :]) / mass
        sigmas[i] = variance

    return sigmas

# create a function to load the optimal GPM from data
def load_gpm(filename):
    # read the whole file
    with open(filename, 'r') as fi:
        lines = fi.readlines()[0]

    # if there is a trailing comma, don't read it
    if lines[-1] == ",":
        lines = lines[:-1]

    # separate out into the results
    sep = lines.strip().split(",")
    sepf = [float(i) for i in sep]
    J0, U0, I2, J2, K2 = sepf

    # build and return the gpm model
    gpm = GaussianPlume2D(J0, U0)
    gpm.set_parameters(I2, J2, K2)
    return gpm
    


# check if everything is working
if __name__ == "__main__":
    # read in a data file and calcualte its meanfree and mean component
    data = read_data(os.path.join(DATA_BASE_PATH, "2d_conc_wind205.pkl"))
    cf, cbar = calculate_meanfree(data)
