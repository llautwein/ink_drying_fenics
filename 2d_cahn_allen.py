import numpy as np 
import fenics
from dolfin import *
import matplotlib.pyplot as plt

# Parameters
T = 10
num_steps = 50
dt = T / num_steps  
mu = 0.0

# mesh and function space (linear Lagrange elements and 1d interval)
nx = 50
ny = 50
L = 10
mesh = RectangleMesh(Point(0, 0), Point(L, L), nx, ny)
V = FunctionSpace(mesh, "Lagrange", 1)

# initial condition
#phi_k = interpolate(Constant(-0.5), V)
#phi_k = interpolate(Expression("x[0] < 5.0 ? 0.1 : -0.1", degree=0), V)
phi_k = interpolate(Expression("0.2*cos(pi*(x[0]+x[1])/5)", degree=2), V)
#phi_k = interpolate(Expression("0.2*tanh(x[0]-5)", degree=2), V)
phi_0 = phi_k.copy(deepcopy=True)

# weak formulation of the problem
phi = Function(V)
v = TestFunction(V)
F = (phi - phi_k)/dt*v*dx + inner(grad(phi), grad(v))*dx - (-phi**3+phi+mu)*v*dx


J = derivative(F, phi)  # Ableitung für Newton-Verfahren
problem = NonlinearVariationalProblem(F, phi, J=J)
solver = NonlinearVariationalSolver(problem)

iterates = [phi_k]
for n in range(num_steps):
    solver.solve()       
    phi_k.assign(phi)
    phi_solution = phi.copy(deepcopy=True)
    iterates.append(phi_solution)

plt.rcParams.update({'font.size': 18})

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
cmap = "coolwarm"

vmin = phi.vector().min()
vmax = phi.vector().max()

plt.sca(axes[0])  
c0 = plot(phi_0, mode="color")
c0.set_clim(vmin, vmax)  
c0.set_cmap(cmap)
axes[0].set_title("Initial condition")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

# Plot final solution
plt.sca(axes[1])  
c1 = plot(phi, mode="color")
c1.set_clim(vmin, vmax) 
c1.set_cmap(cmap)
axes[1].set_title(f"Solution at t={T}")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")

# Gemeinsame Colorbar
fig.colorbar(c1, ax=axes.ravel().tolist(), orientation='vertical')
fig.subplots_adjust(right=0.85)
plt.show()
"""
plt.figure()
initial_condition = plot(phi_0)
plt.colorbar(initial_condition)
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Initial condition")

plt.figure()
final_result = plot(phi)
plt.colorbar(final_result)
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Solution of 2D Allen-Cahn equation at t={T}")
plt.show()
"""