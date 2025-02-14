import fenics
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 10                  # Time interval length
num_steps = 50          # number of time steps
dt = T / num_steps      
mu = 0.0                # moisture parameter
L = 10                  # domain length
eps = 1e-1              # how much the ink particles like to be in the solvent
M_sig = 0.2*0.03        # product of evaporation coefficient and surface tension
                        # appears through non-dimensionalisation

# mesh and function space (linear Lagrange elements and 1d interval)
nx = 1000
mesh = IntervalMesh(nx, 0, L)
V = FunctionSpace(mesh, "Lagrange", 1)

# initial conditions
#phi_k = interpolate(Expression("0.2*tanh(x[0]-5)", degree=2), V)
phi_k = interpolate(Expression("0.2*cos(pi*x[0]/5)", degree=2), V)
phi_0 = phi_k.copy(deepcopy=True)

#n_k = interpolate(Constant(1/L), V)
#n_k = interpolate(Expression("x[0] < L/2 ? 0.2 : 0", L=L, degree=1), V)
n_k = interpolate(Expression("0.1*exp(-pow(x[0]-L/2, 2)/0.01)", L=L, degree=2), V)
n_0 = n_k.copy(deepcopy=True)

# mobility function
def D(distr):
    return 1

# weak formulation
phi = Function(V)
v = TestFunction(V)
F_phi = (phi - phi_k)/dt*v*dx + inner(grad(phi), grad(v))*dx - (-phi**3 + phi - eps*n_k + mu)*v*dx
J_phi = derivative(F_phi, phi)

n = Function(V)
F_n = (n - n_k)/dt*v*dx + (1/M_sig)*D(n)*inner(grad(n) + eps*n*grad(phi), grad(v))*dx
J_n = derivative(F_n, n)

# set up of the solvers
problem_phi = NonlinearVariationalProblem(F_phi, phi, J=J_phi)
solver_phi = NonlinearVariationalSolver(problem_phi)

problem_n = NonlinearVariationalProblem(F_n, n, J=J_n)
solver_n = NonlinearVariationalSolver(problem_n)

iterates_phi = [phi_0]
iterates_n = [n_0]
for i in range(num_steps):
    solver_phi.solve()       
    phi_k.assign(phi)
    iterates_phi.append(phi_k.copy(deepcopy=True))
    
    solver_n.solve()
    n_k.assign(n)
    iterates_n.append(n_k.copy(deepcopy=True))

plt.rcParams.update({'font.size': 18})
x = mesh.coordinates().flatten()  

plotted_timesteps = [1, 50]

plt.subplots(1, 2, figsize=(15, 9))

plt.subplot(1, 2, 1)
plt.plot(x, phi_0.compute_vertex_values(), label="Initial condition")
for j in plotted_timesteps:
    plt.plot(x, iterates_phi[j].compute_vertex_values(), label=f"t={j*dt}")
    
plt.xlim(0, 10)
plt.ylim(-1.1, 1.1)
plt.xlabel("x")
plt.ylabel("phi")
plt.title("Solution of 1D Allen-Cahn\n equation (order parameter)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, n_0.compute_vertex_values(), label="Initial condition")
for j in plotted_timesteps:
    plt.plot(x, iterates_n[j].compute_vertex_values(), label=f"t={j*dt}")
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("n")
plt.title("Solution of 1D Allen-Cahn\n equation (distribution)")
plt.legend()

plt.show()