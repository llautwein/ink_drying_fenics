import fenics
from dolfin import *
import ufl
import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 10  # Time interval length
num_steps = 50  # number of time steps
dt = T / num_steps
mu = 0.0  # moisture parameter
L = 10  # domain length
eps = -1e-1  # how much the ink particles like to be in the solvent
sigma = 0.01
M = 0.1
M_sig = M * sigma  # product of evaporation coefficient and surface tension
# appears through non-dimensionalisation

# mesh and function space (linear Lagrange elements and 1d interval)
nx = 50
ny = 50
mesh = RectangleMesh(Point(0, 0), Point(L, L), nx, ny)
V = FunctionSpace(mesh, "Lagrange", 1)

# mobility function
def D(p):
    return 0.5*ufl.tanh(4*phi)+0.5

# initial conditions
#phi_k = interpolate(Expression("0.2*tanh(x[0]-5)", degree=2), V)
#phi_k = interpolate(Expression("0.2*cos(pi*(x[0]+x[1])/5)", degree=2), V)
phi_k = Function(V)
vec = phi_k.vector()
vec[:] = np.random.uniform(-0.1, 0.1, phi_k.vector()[:].size)
phi_0 = phi_k.copy(deepcopy=True)

n_k = interpolate(Constant(1/L), V)
# n_k = interpolate(Expression("x[0] < L/2 ? 0.2 : 0", L=L, degree=1), V)
#n_k = interpolate(Expression("0.1*exp(-pow(x[0]-L/2, 2)/0.01)", L=L, degree=2), V)
n_0 = n_k.copy(deepcopy=True)

# weak formulation
phi = Function(V)
v = TestFunction(V)
F_phi = (phi - phi_k) / dt * v * dx + inner(grad(phi), grad(v)) * dx - (-phi ** 3 + phi - eps * n_k + mu) * v * dx
J_phi = derivative(F_phi, phi)

n = Function(V)
F_n = (n - n_k) / dt * v * dx + (1 / M_sig) * D(phi) * inner(grad(n) + eps * n * grad(phi), grad(v)) * dx
J_n = derivative(F_n, n)

free_energy = (-(phi**2)/2 + (phi**4)/4 + sigma/2*inner(grad(phi), grad(phi)) - mu*phi)*dx

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
for i in range(num_steps):
    t = (i+1)*dt
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

vmin = phi.vector().min()
vmax = phi.vector().max()

plt.sca(axes[0, 0])
c0 = plot(phi_0, mode="color")
c0.set_clim(vmin, vmax)
c0.set_cmap(cmap)
axes[0, 0].set_title("Initial condition (order parameter)")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")

plt.sca(axes[0, 1])
c1 = plot(phi, mode="color")
c1.set_clim(vmin, vmax)
c1.set_cmap(cmap)
axes[0, 1].set_title(f"Solution at t={T} (order parameter)")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("y")

cbar1 = fig.colorbar(c1, ax=[axes[0, 0], axes[0, 1]], orientation='vertical', shrink=0.8)
cbar1.set_label("Order parameter")

#########################################
cmap = "binary"
nmin = n.vector().min()
nmax = n.vector().max()

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
axes[1, 1].set_title(f"Solution at t={T} (distribution)")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")

cbar2 = fig.colorbar(c3, ax=[axes[1, 0], axes[1, 1]], orientation='vertical', shrink=0.8)
cbar2.set_label("Distribution")

plt.show()
plt.close(fig)
#################################
plt.figure(figsize=(12, 6))
plt.plot([dt*k for k in range(num_steps)], free_energy_vals)
plt.xlabel("Time")
plt.ylabel("Free energy")
plt.title("Evolution of free energy over time")

plt.show()