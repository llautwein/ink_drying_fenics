import cahn_allen_2d as CA
import figure_handler as fh
import config as cfg
import csv
import numpy as np

"""
Solve this model for various values of eps and saves the different measures in a csv file.
"""

def write_csv(list, name):
    with open(f"{name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for value in list:
            writer.writerow([value])

# available phi ic options: constant, tanh, cosine, gaussian, sine_checkerboard, random
phi_init_option = "gaussian"

# available n ic options: constant, gaussian, half_domain, two_drop
n_init_option = "gaussian"

mu_list = np.linspace(-5, 5, 50)
distance_phi_list = []
distance_n_list = []
kullback_n_list = []
for mu in mu_list:
    print(f"mu = {mu}")
    config = cfg.Config(mu)
    cahn_allen = CA.CahnAllen2D(config, 0.2, 1,
                                phi_init_option, n_init_option, False)
    _, _, distance_measure_n, distance_measure_phi, kullback_measure_n = cahn_allen.solve()
    distance_phi_list.append(distance_measure_phi)
    distance_n_list.append(distance_measure_n)
    kullback_n_list.append(kullback_measure_n)
    print("#############################################################################")

print(distance_phi_list)
print(distance_n_list)
print(kullback_n_list)
write_csv(distance_phi_list, "Results/distance_phi_list")
write_csv(distance_n_list, "Results/distance_n_list")
write_csv(kullback_n_list, "Results/kullback_n_list")

