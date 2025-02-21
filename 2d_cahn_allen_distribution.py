import fenics
from dolfin import *
import ufl
import numpy as np
import matplotlib.pyplot as plt
import config as cfg

# available phi ic options: constant, tanh, cosine, gaussian, sine_checkerboard, random
phi_init_option = "gaussian"

# available n ic options: constant, gaussian, half_domain
n_init_option = "gaussian"

# mobility function
def D(p):
    return 0.05*(1+ufl.tanh(20*phi))

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
F_n = (n - n_k) / config.dt * v * dx + (1 / config.M_sig) * D(phi) * inner(grad(n) + config.eps * n * grad(phi), grad(v)) * dx
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

plt.rcParams.update({'font.size': 18})

fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
cmap = "coolwarm"

plt.sca(axes[0, 0])
c0 = plot(phi_0, mode="color")
c0.set_clim(-1, 1)
c0.set_cmap(cmap)
axes[0, 0].set_title("Initial condition (order parameter)")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")

plt.sca(axes[0, 1])
c1 = plot(phi, mode="color")
c1.set_clim(-1, 1)
c1.set_cmap(cmap)
axes[0, 1].set_title(f"Solution at t={config.T} (order parameter)")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("y")

cbar1 = fig.colorbar(c1, ax=[axes[0, 0], axes[0, 1]], orientation='vertical', shrink=0.8)
cbar1.set_label("Order parameter")

#########################################
cmap = "binary"
nmin = n_0.vector().min()
nmax = n_0.vector().max()

plt.sca(axes[1, 0])
c2 = plot(n_0, mode="color")
c2.set_clim(nmin, nmax)
c2.set_cmap(cmap)
axes[1, 0].set_title("Initial condition (distribution)")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y")

plt.sca(axes[1, 1])
c3 = plot(n, mode="color")
c3.set_clim(nmin, nmax)
c3.set_cmap(cmap)
axes[1, 1].set_title(f"Solution at t={config.T} (distribution)")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")

cbar2 = fig.colorbar(c3, ax=[axes[1, 0], axes[1, 1]], orientation='vertical', shrink=0.8)
cbar2.set_label("Distribution")

plt.show()
plt.close(fig)
#################################
plt.figure(figsize=(12, 6))
plt.plot([config.dt*k for k in range(config.num_steps)], free_energy_vals)
plt.xlabel("Time")
plt.ylabel("Free energy")
plt.title("Evolution of free energy over time")

plt.show()