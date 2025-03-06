import csv
import figure_handler as fh
import config as cfg
import numpy as np

"""
This script imports the lists which contain the measure values for differents mus and plots the graphs
"""

random_gauss_path = "Results/png/mu_dependence/random_gauss"
checkerboard_gauss_path = "Results/png/mu_dependence/checkerboard_gauss"
gauss_gauss_path = "Results/png/mu_dependence/gauss_gauss"

path_distance_phi = f"{gauss_gauss_path}/distance_phi_list"
path_distance_n = f"{gauss_gauss_path}/distance_n_list"
path_kullback_n = f"{gauss_gauss_path}/kullback_n_list"

mu_list = np.linspace(-5, 5, 50)

config = cfg.Config(0)
figure_handler = fh.FigureHandler(config)

figure_handler.mu_dependence_plots(mu_list, path_distance_phi, path_distance_n, path_kullback_n)

