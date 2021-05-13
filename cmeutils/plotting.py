import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def plot_distribution(data, label, fit_line=True):
    bin_heights, bin_borders = np.histogram(data, bins='auto')
    bin_widths = np.diff(bin_borders)
    bin_centers = bin_borders[:-1] + bin_widths / 2
    plt.bar(bin_centers, bin_heights, width=bin_widths, label=label)
    plt.legend()

    if fit_line:
        popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.])
        x_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
        y_fit = gaussian(x_fit, *popt)
        plt.plot(x_fit, y_fit, "red")
        return bin_centers, bin_heights, x_fit, y_fit

    return bin_centers, bin_heights

def gaussian(x, mean, amplitude, standard_deviation):
        return amplitude * np.exp( - (x - mean)**2 / (2*standard_deviation ** 2))
