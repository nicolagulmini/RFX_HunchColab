from numpy import linspace
from numpy import array
from numpy.random import normal
from scipy.stats import norm
from matplotlib import pyplot as plt

def generate_gaussian_mixture_curve_1(n_p=2000, n_g=2, mean_and_vars=[(0,1), (4,2)], start=-3, sample_period=.005, awgn_dev=.001, plot=False):
    x = linspace(start, start+n_p*sample_period, n_p)
    y = [sum([norm.pdf(el, mean_and_vars[i][0], mean_and_vars[i][1])+normal(0, awgn_dev) for i in range(n_g)]) for el in x]
    if plot:
        plt.plot(x, y)
        plt.show()
    return array(x.tolist()+y)

def generate_gaussian_mixture_curve_2(n_p=100, n_g=2, mean_and_vars=[(0,0.01), (0.5,0.01)], awgn_dev=.001, plot=False):
    x = linspace(0, 1, n_p)
    y = [sum([norm.pdf(el, mean_and_vars[i][0], mean_and_vars[i][1])+normal(0, awgn_dev) for i in range(n_g)]) for el in x]
    y = y/max(y)
    if plot:
        plt.plot(x, y)
        plt.show()
    return x, y

''' test
x, y = generate_gaussian_mixture_curve_2()
print(type(x))
print(type(y))
print(x.shape)
print(y.shape)
bidim_curve = []
for i in range(x.shape[0]):
    point = array([x[i], y[i]])
    bidim_curve.append(point)
bidim_curve = array(bidim_curve)
print(bidim_curve.shape)
'''
