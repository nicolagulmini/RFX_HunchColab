from numpy import linspace
from numpy import array
from numpy.random import normal
from scipy.stats import norm
from matplotlib import pyplot as plt

def generate_gaussian_mixture_curve(n_p=2000, n_g=2, mean_and_vars=[(0,1), (4,2)], start=-3, sample_period=.005, awgn_dev=.001, plot=False):
    x = linspace(start, start+n_p*sample_period, n_p)
    y = [sum([norm.pdf(el, mean_and_vars[i][0], mean_and_vars[i][1])+normal(0, awgn_dev) for i in range(n_g)]) for el in x]
    if plot:
        plt.plot(x, y)
        plt.show()
    return array(x+y)
