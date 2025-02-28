import fenics
from dolfin import *
import ufl
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
import figure_handler as fh

# available phi ic options: constant, tanh, cosine, gaussian, sine_checkerboard, random
phi_init_option = "random"

# available n ic options: constant, gaussian, half_domain
n_init_option = "gaussian"

# mobility function
#D = lambda p: 0.05*(1+ufl.tanh(20*phi))
n_thr_below = 0.5
n_thr_above = 1
D = lambda p, n: 0.05*(1+ufl.tanh(10*phi))*(1-ufl.tanh(10*(n - n_thr_below)*(n_thr_below - n)))

config = cfg.Config()

phi_k, n_k = config.set_ics(phi_init_option, n_init_option)
phi_0 = phi_k.copy(deepcopy=True)
n_0 = n_k.copy(deepcopy=True)

# weak formulation
phi = Function(config.V)
v = TestFunction(config.V)
F_phi = (phi - phi_k) / config.dt * v * dx + inner(grad(phi), grad(v)) * dx + (phi ** 3 - phi + config.eps * n_k - config.mu) * v * dx
J_phi = derivative(F_phi, phi)

n = Function(config.V)
F_n = (n - n_k) / config.dt * v * dx + (1 / config.M_sig) * D(phi, n) * inner(grad(n) + config.eps * n * grad(phi), grad(v)) * dx
J_n = derivative(F_n, n)

free_energy = (-(phi**2)/2 + (phi**4)/4 + config.sigma/2*inner(grad(phi), grad(phi)) -
               config.mu*phi + n*(ufl.ln(n)-1) + config.eps*n*phi)*dx

# set up of the solvers
problem_phi = NonlinearVariationalProblem(F_phi, phi, J=J_phi)
solver_phi = NonlinearVariationalSolver(problem_phi)

problem_n = NonlinearVariationalProblem(F_n, n, J=J_n)
solver_n = NonlinearVariationalSolver(problem_n)

phi_xdmf = XDMFFile("phi_solution.xdmf")
n_xdmf = XDMFFile("n_solution.xdmf")

phi_xdmf.parameters["flush_output"] = True
phi_xdmf.parameters["functions_share_mesh"] = True
n_xdmf.parameters["flush_output"] = True
n_xdmf.parameters["functions_share_mesh"] = True

phi_xdmf.write(phi_k, 0)
n_xdmf.write(n_k, 0)

iterates_phi = [phi_0]
iterates_n = [n_0]
free_energy_vals = []
for i in range(config.num_steps):
    t = (i+1)*config.dt
    solver_phi.solve()
    phi_k.assign(phi)
    iterates_phi.append(phi_k.copy(deepcopy=True))

    solver_n.solve()
    n_k.assign(n)
    iterates_n.append(n_k.copy(deepcopy=True))
    phi_xdmf.write(phi_k, t)
    n_xdmf.write(n_k, t)
    free_energy_vals.append(assemble(free_energy))



figure_handler = fh.FigureHandler(config)
#figure_handler.individual_plots([phi_0, phi], [n_0, n], [0, config.T])
#figure_handler.two_by_two_plot(phi, phi_0, n, n_0)
#figure_handler.free_energy_plot(free_energy_vals)
figure_handler.horizont_slice_n_plot(n, 0.5, 13, 100)
