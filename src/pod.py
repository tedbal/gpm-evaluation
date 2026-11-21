import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# import functions from util
from utils import *
from gpmeval import GaussianPlume2D

# set the data base path for the work
DATA_BASE_PATH = os.path.abspath("../data/")


def calculate_exposure(gpm, c, pts):
    # takes in a fitted gaussian plume model instance
    # and a concentration (nt, nx1, nx2)
    # returns the exposure (nt, nx1, nx2)
    ctilde = gpm.evaluate(pts)
    diff = c - ctilde
    exposure = np.fmax(np.zeros_like(diff), diff)

    # calculate as meanfree
    exposure = exposure - np.mean(exposure, axis=0)

    return exposure


class ExposureROM2D:
    def __init__(self, exposure):
        self.exposure = exposure
        # call the fit function on the exposure
        self.fit(exposure)

    def fit(self, exposure):
        # calculate the autocovariance matrix
        epod = exposure.reshape(exposure.shape[0], -1).T
        Rc = np.matmul(np.transpose(epod), epod)

        # calculate the eigenvalues and eigenvectors of the autocovariance
        eigs, eigvs = np.linalg.eigh(Rc)

        # sort the eigenvalues and vectors
        inds = np.argsort(eigs)[::-1]
        eigs_sorted = np.maximum(0.0, eigs[inds])
        eigvs_sorted = eigvs[:,inds]

        # calculate the relative energy of each mode
        rel_e1 = eigs_sorted / np.sum(eigs_sorted)
        cum_e1 = np.cumsum(rel_e1)

        # calculate the modes (phi) and the time-dependent coefficients (a)
        nt, nx1, nx2 = exposure.shape
        phi = (epod @ eigvs_sorted) / np.sqrt(eigs_sorted)[None, :]
        a = phi.T @ epod
        modeshapes = phi.reshape(nx1, nx2, nt)

        # save the results as class parameters
        self.modeshapes = modeshapes
        self.a = a
        self.rel_e1 = rel_e1
        self.cum_e1 = cum_e1

    def plot_energy(self, nmodes=None, savefile=None):
        # if nmodes is none, use the total number of modes
        if nmodes is None:
            nmodes = self.rel_e1.shape[0]
        rel_e1_trim = self.rel_e1[:nmodes]
        cum_e1_trim = self.cum_e1[:nmodes]
            
        # plot the result
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(rel_e1_trim, color='b')
        ax2.plot(cum_e1_trim, color='b')

        # format the plot
        ax1.set_ylabel("Relative energy")
        ax1.set_xlabel("Mode")
        ax1.set_xlim(0, None)
        ax2.set_ylabel("Cumulative energy")
        ax2.set_xlabel("Mode")
        ax2.set_xlim(0, None)

        # save and show the plot
        if savefile is not None:
            plt.savefig(savefile, dpi=600, bbox_inches='tight')
        plt.show()

    def plot_modes(self, savefile=None):
        # plot the result
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
        im = ax0.contourf(self.modeshapes[..., 0].T, cmap='RdBu')
        im = ax1.contourf(self.modeshapes[..., 1].T, cmap='RdBu')
        im = ax2.contourf(self.modeshapes[..., 2].T, cmap='RdBu')

        # format the plot
        ax0.set_title(r"First POD mode $\phi_1$")
        ax1.set_ylabel(r"$x_2$")
        ax1.set_title(r"Second POD mode $\phi_2$")
        ax1.set_ylabel(r"$x_2$")
        ax2.set_title(r"Third POD mode $\phi_3$")
        ax2.set_ylabel(r"$x_2$")
        ax2.set_xlabel(r"$x_1$")
        fig.colorbar(im, ax=[ax0, ax1, ax2], location='bottom')

        # save and show the plot
        if savefile is not None:
            plt.savefig(savefile, dpi=600, bbox_inches='tight')
        plt.show()

    # plot the mean exposure
    def plot_mean_exposure(self, savefile=None):
        # calculate the mean exposure, and then plot it
        expmean = np.mean(self.exposure, axis=0)
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        im = ax.contourf(expmean.T, cmap='Reds')

        # format the plot
        ax.set_title(r"Exposure, $E$")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        fig.colorbar(im, ax=ax, location='bottom')

        # save and show the plot
        if savefile is not None:
            plt.savefig(savefile, dpi=600, bbox_inches='tight')
        plt.show()

        

if __name__ == "__main__":
    # define constants
    U0 = 6.16
    dx = 0.15

    # read in the data and the best-fit model from the results
    datafile = f"2d_conc_wind{100*U0:.0f}.pkl"
    data = read_data(os.path.join(DATA_BASE_PATH, datafile))
    cf, cbar = calculate_meanfree(data)
    gpm = load_gpm(f"../res/gpm{U0*100:.0f}_optim.txt")
    gpm.normalize()

    # calculate the x1 and x2 coordinates and the corresponding offset coordinates
    x1, x2 = np.arange(cbar.shape[0]) * dx, np.arange(-cbar.shape[1]/2, cbar.shape[1]/2) * dx
    x1, x2 = np.meshgrid(x1, x2, indexing='ij')
    isrc_x1, isrc_x2 = find_source(cbar)
    xi, eta = x1 - x1[isrc_x1, isrc_x2], x2 - x2[isrc_x1, isrc_x2]

    # calculate the exposure and generate the ROM
    exposure = calculate_exposure(gpm, cf, (x1, x2))
    rom = ExposureROM2D(exposure)

    # plot the relative and cumulative energies
    rom.plot_mean_exposure(savefile=f"../figs/exposure_{U0*100:.0f}.png")
    rom.plot_energy(nmodes=50, savefile=f"../figs/pod_{U0*100:.0f}_energy.png")
    rom.plot_modes(savefile=f"../figs/modes_{U0*100:.0f}.png")
