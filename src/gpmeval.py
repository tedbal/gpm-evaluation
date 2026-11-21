# evaluation of the gaussian plume model
# ted balabanski
# dependencies
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# import functions from util
from utils import *

# set the data base path for the work
DATA_BASE_PATH = os.path.abspath("../data/")

# declare the gaussian plume model
class GaussianPlume2D:
    # instantiation
    def __init__(self, J0, U0):
        # declare the mean windspeed and the mass flow rate as a class value
        self.J0 = J0
        self.U0 = U0

        # initialize the coefficients to fit the variance
        self.I2 = 0.0
        self.J2 = 0.0
        self.K2 = 0.0

    # for debugging, print out the plume
    def __str__(self):
        str1 = f"Gaussian Plume. J0 = {self.J0:.3f}; U0 = {self.U0:.3f}"
        str2 = f"Parameters: I2 = {self.I2:.3f}; J2 = {self.J2:.3f}; K2 = {self.K2:.3f}"
        return "\n".join([str1, str2])

    
    # set the parameters of the GPM, if passed
    def set_parameters(self, I2 = None, J2 = None, K2 = None):
        if (I2 is not None):
            self.I2 = I2
        if (J2 is not None):
            self.J2 = J2
        if (K2 is not None):
            self.K2 = K2

    # set the J0 and U0 such that their ratio is unity
    def normalize(self):
        self.J0 = 1
        self.U0 = 1
    
    # evaluate at a set of points    
    def evaluate(self, pts):
        # separate out the x1 and x2 points to be evaluated
        x1, x2 = pts[0], pts[1]
        x1f, x2f = x1.flatten(), x2.flatten()

        # get a mask where upwind of the source
        Cmask = x1f > 0
        x1m, x2m = x1f[Cmask], x2f[Cmask]
        
        # calculates the concentration at the effective plume height
        # calculate the variance in the x2 direction
        a = self.I2 + self.J2 * np.log(x1m) + self.K2 * np.square(np.log(x1m))
        sigma2 = np.square(np.exp(a))
        
        Cf = np.zeros_like(x1f)
        Cf[Cmask] = (self.J0 / self.U0) * (1/(np.sqrt(sigma2) * np.sqrt(2 * np.pi))) * np.exp(-np.square(x2m)/(2 * sigma2))
        Cf = np.nan_to_num(Cf, nan=0.0, posinf=0.0, neginf=0.0)
        C = Cf.reshape(x1.shape)
        
        return C

    
    # define a function to evaluate the GPM
    # with the parameters passed (i.e., I, J, K)
    def _parameterized_eval(self, pts, I_eval, J_eval, K_eval):
        # save the old parameters, and set the new ones
        I_reset, J_reset, K_reset = self.I2, self.J2, self.K2
        self.set_parameters(I_eval, J_eval, K_eval)
        
        # calculate the evaluation
        C_eval = self.evaluate(pts)

        # reset the parameters and return the evaluated value
        self.set_parameters(I_reset, J_reset, K_reset)
        return C_eval

    # for use in curve_fit
    def _flat_parameterized_eval(self, pts, I_eval, J_eval, K_eval):
        return self._parameterized_eval(pts, I_eval, J_eval, K_eval).ravel()

    # for use in curve fit
    def _parameterized_sigma2(self, xival, I2, J2, K2):
        a = I2 + J2 * np.log(xival) + K2 * np.square(np.log(xival))
        return np.square(np.exp(a))
    
    # calculate the mean squared error
    def mse(self, pts, meanfield):
        ctilde = self.evaluate(pts)
        err = meanfield - ctilde
        return np.sqrt(np.mean(np.square(err)))
    
    # fit the gaussian plume to data
    def fit(self, meanfield, pts):
        # takes the mean field vector
        # and the source coordinate (isrc_x1, isrc_x2)
        # returns nothing, fits the gpm to the data
        isrc_x1, isrc_x2 = find_source(meanfield)
        xi, eta = pts[0] - pts[0][isrc_x1, isrc_x2], pts[1] - pts[1][isrc_x1, isrc_x2] 
        
        # curve fit the results and set it to the parameterized values
        popt, pcov = curve_fit(self._flat_parameterized_eval, (xi, eta), meanfield.flatten())
        I_fit, J_fit, K_fit = popt
        self.set_parameters(I_fit, J_fit, K_fit)


    # fit just to variances
    def fitvar(self, meanfield, x1v, observed_variances):
        # pass the x1v as a vector of x1values, not the full result
        isrc_x1, isrc_x2 = find_source(meanfield)
        xi = x1v - x1v[isrc_x1]

        # only fit the points where xi > 0
        mask = xi > 0
        xim = xi[mask]
        obsvarm = observed_variances[mask]

        # use the curve fit on the variances
        popt, pcov = curve_fit(self._parameterized_sigma2, xim, obsvarm)
        I_fit, J_fit, K_fit = popt
        self.set_parameters(I_fit, J_fit, K_fit)


    # write the results to a file
    def save(self, filename, acc=12):
        # check if the file exists
        if not os.path.exists(filename):
            os.mknod(filename)

        with open(filename, 'w') as fi:
            fi.write('{:.{acc}f},'.format(self.J0, acc=acc))
            fi.write('{:.{acc}f},'.format(self.U0, acc=acc))
            fi.write('{:.{acc}f},'.format(self.I2, acc=acc))
            fi.write('{:.{acc}f},'.format(self.J2, acc=acc))
            fi.write('{:.{acc}f},'.format(self.K2, acc=acc))           


        
if __name__ == "__main__":
    # set parameters
    U0 = 4.11 # mean windspeed ratio to rms speed
    Urms = 1.46
    U = U0 * Urms
    dt = 5e-4
    G = 1000 / (10 * dt) # 1000 emitted every 10 seconds
    G = G/3.2
    dx = 0.15
    J0 = G * (dx ** 3)

    # read in a data file and calcualte its meanfree and mean component
    datafile = f"2d_conc_wind{100*U0:.0f}.pkl"
    data = read_data(os.path.join(DATA_BASE_PATH, datafile))
    cf, cbar = calculate_meanfree(data)
    
    # calculate the x1 and x2 coordinates and the corresponding offset coordinates
    x1, x2 = np.arange(cbar.shape[0]) * dx, np.arange(-cbar.shape[1]/2, cbar.shape[1]/2) * dx
    x1, x2 = np.meshgrid(x1, x2, indexing='ij')
    isrc_x1, isrc_x2 = find_source(cbar)
    xi, eta = x1 - x1[isrc_x1, isrc_x2], x2 - x2[isrc_x1, isrc_x2]

    # calculate the mean U0/J0 ratio at the midpoint downwind
    imid = x1.shape[0] // 2
    data_loading = np.trapezoid(cbar[imid, :], eta[imid, :])
    print(f"observed linear loading: {data_loading:.1f}")
    print(f"theoretical linear loading: {J0/U:.1f}")
    print("\n")
    J0 = data_loading * U
    
    print(f"read {datafile}")
    print(f"data shape {cf.shape}")
    print(f"mean shape {cbar.shape}")
    print(f"shape of coordinate points {x2.shape}")
    print("")

    # define the gaussian plume model
    gpm = GaussianPlume2D(J0, U)

    # fit and evaluate the model
    # calculate the observed downwind variance
    observed_variances = calculate_downwind_variance((x1, x2), cbar)
    gpm.fitvar(cbar, x1[:,0], observed_variances)
    print(gpm)
    mse = gpm.mse((x1, x2), cbar)
    print(f"mse of fit {mse:.4f}")


    # calculate the model predictions and plot it against the 
    modeleval = gpm.evaluate((x1, x2))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8))
    ax1.contourf(x1, x2, np.log10(cbar), cmap='Reds')
    ax2.contourf(x1, x2, np.log10(modeleval), cmap='Reds')
    plt.show()

    # do a new plot of a transverse profile
    fig, ax = plt.subplots(1, 1, figsize=(8,4))
    i = x1.shape[0] // 2 # choose the midpoint downwind
    ax.plot(x2[i,:], modeleval[i,:], label="gpm")
    ax.plot(x2[i,:], cbar[i,:], label="data")
    ax.legend()
    plt.show()

    # save the result
    gpm.save(f"../res/gpm{U0*100:.0f}_optim.txt")
    print("wrote the results")
