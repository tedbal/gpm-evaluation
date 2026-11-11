# evaluation of the gaussian plume model
# ted balabanski
# dependencies
import numpy as np
import os
import matplotlib.pyplot as plt

# import functions from util
from utils import read_data, calculate_meanfree

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

    def evaluate(self, pts):
        # separate out the x1 and x2 points to be evaluated
        x1, x2 = pts[0], pts[1]
        
        # calculates the concentration at the effective plume height
        # calculate the variance in the x2 direction
        a = self.I2 + self.J2 * np.log(x1) + self.K2 * np.square(np.log(x2))
        sigma2 = np.square(np.exp(a))
        C = (self.J0 / self.U0) * (1/(sigma2 * np.sqrt(2 * np.pi))) * np.exp(np.square(x2))

        return C
        
        
        
    # fit the gaussian plume to data
    def fit(meanfield):
        # TODO: implement
        pass
