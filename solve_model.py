import cahn_allen_2d as CA
import config as cfg
import figure_handler as fh

"""
This script calls the solve function from the cahn_allen_2d script to solve the model.
"""

config = cfg.Config(-0.5)

# available phi ic options: constant, tanh, cosine, gaussian, sine_checkerboard, random
phi_init_option = "random"

# available n ic options: constant, gaussian, half_domain, two_drop
n_init_option = "gaussian"

cahn_allen = CA.CahnAllen2D(config, 0.2, 1, phi_init_option, n_init_option, True)
iterates_phi, iterates_n, distance_measure_n, distance_measure_phi, kullback_measure_n = cahn_allen.solve()
phi_0, n_0 = iterates_phi[0], iterates_n[0]

first_timestamp = 5
second_timestamp = 15
phi_solutions = [phi_0, iterates_phi[first_timestamp], iterates_phi[second_timestamp], iterates_phi[len(iterates_phi) - 1]]
n_solutions = [n_0, iterates_n[first_timestamp], iterates_n[second_timestamp], iterates_n[len(iterates_n) - 1]]
timestamps = [0, first_timestamp*config.dt, second_timestamp*config.dt, config.T]

figure_handler = fh.FigureHandler(config)
save_slice = False
save_individual = False
save_free_energy = False
#figure_handler.individual_plots(phi_solutions, n_solutions, timestamps, save_individual)
#figure_handler.two_by_two_plot(phi, phi_0, n, n_0)
#figure_handler.horizont_slice_n_plot(n_solutions, timestamps, 12.5, 100, save_slice)

#print(f"Distance measure = {distance_n}")
#print(f"Kullback measure = {kullback_n}")
