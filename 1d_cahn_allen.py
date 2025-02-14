import numpy as np 
from dolfin import *
import fenics
import matplotlib.pyplot as plt

# Parameters
T = 10
num_steps = 50
dt = T / num_steps  
mu = 0.0

# mesh and function space (linear Lagrange elements and 1d interval)
nx = 1000
mesh = IntervalMesh(nx, 0, 10)
V = FunctionSpace(mesh, "Lagrange", 1)

# initial condition
#phi_k = interpolate(Constant(-0.5), V)
#phi_k = interpolate(Expression("x[0] < 5.0 ? 0.5 : -0.5", degree=0), V)
#phi_k = interpolate(Expression("0.2*cos(pi*x[0]/5)", degree=2), V)
phi_k = interpolate(Expression("0.2*tanh(x[0]-5)", degree=2), V)


phi_0 = phi_k.copy(deepcopy=True)
# weak formulation of the problem
phi = Function(V)
v = TestFunction(V)
F = (phi - phi_k)/dt*v*dx + inner(grad(phi), grad(v))*dx - (-phi**3+phi+mu)*v*dx


J = derivative(F, phi)  # Gateaux derivative of F for Newton's method
problem = NonlinearVariationalProblem(F, phi, J=J)
solver = NonlinearVariationalSolver(problem)

iterates = [phi_k]
for n in range(num_steps):
    solver.solve()       
    phi_k.assign(phi)
    phi_solution = phi.copy(deepcopy=True)
    iterates.append(phi_solution)
    

plt.rcParams.update({'font.size': 18})
plt.figure()
x = mesh.coordinates().flatten()  # Gitterpunkte holen
plt.plot(x, phi_0.compute_vertex_values(), label="Initial condition")
plt.plot(x, phi.compute_vertex_values(), label=f"Final state at t={T}")
plt.xlim(0, 10)
plt.ylim(-1.1, 1.1)
plt.xlabel("x")
plt.ylabel("phi")
plt.legend()
plt.title("Solution of 1D Allen-Cahn equation")
plt.show()