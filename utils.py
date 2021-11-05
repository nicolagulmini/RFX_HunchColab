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

def generate_gaussian_mixture_curve_2(n_p=100, mean_and_vars=[(0,0.01), (0.5,0.01)], awgn_dev=.001, plot=False, bidim_points=False):
    x = linspace(0, 1, n_p)
    y = [sum([norm.pdf(el, mean_and_vars[i][0], mean_and_vars[i][1])+normal(0, awgn_dev) for i in range(len(mean_and_vars))]) for el in x]
    y = y/max(y)
    if plot:
        plt.plot(x, y)
        plt.show()
    if bidim_points:
        bidim_curve = []
        for i in range(x.shape[0]):
            point = array([x[i], y[i]])
            bidim_curve.append(point)
        return array(bidim_curve)
    return x, y

def plot_latent_space(vae, sub_len=20, step=.2, savefig=False):
    fig, axs = plt.subplots(sub_len, sub_len, sharex='all', sharey='all', figsize=(10, 10))
    row_index = 0
    for first_dim in range(-int(sub_len/2), int(sub_len/2)):
        col_index = 0
        for second_dim in range(-int(sub_len/2), int(sub_len/2)):
            point = array([[first_dim*step, second_dim*step]])
            generated = vae.decoder.predict(point).tolist()[0]
            x_axis = [el[0] for el in generated]
            y_axis = [el[1] for el in generated]
            axs[row_index][col_index].plot(x_axis, y_axis, linewidth=1)
            axs[row_index][col_index].get_xaxis().set_visible(False)
            axs[row_index][col_index].get_yaxis().set_visible(False)
            col_index += 1
        row_index += 1
    if savefig == True:
        plt.savefig('latent_space.png')
