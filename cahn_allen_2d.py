import fenics
from dolfin import *
import ufl
import numpy as np
import matplotlib.pyplot as plt
import config as cfg



class CahnAllen2D:

    def __init__(self, config, n_thr_below, n_thr_above, phi_init_option, n_init_option, write_xdmf=False):
        """
        Initialises the problem
        :param config: config script which sets the parameters
        :param n_thr_below: lower threshold of the ink distribution in the mobility function,
                            below this value ink cannot move
        :param n_thr_above: upper threshold of the ink distribution in the mobility function,
                            above this value ink cannot move
        :param phi_init_option: initial condition of phi
        :param n_init_option: initial condition of n
        :param write_xdmf: writes xdmf file of phi and n if true
        """
        self.config = config
        self.n_thr_below = n_thr_below
        self.n_thr_above = n_thr_above
        self.phi_init_option = phi_init_option
        self.n_init_option = n_init_option
        self.write_xdmf = write_xdmf

        # mobility function
        D = lambda p, n: 0.05*(1+ufl.tanh(10*p))*(1+ufl.tanh(10*(n - self.n_thr_below)*(self.n_thr_above - n)))


        self.phi_k, self.n_k = self.config.set_ics(phi_init_option, n_init_option)
        self.phi_0 = self.phi_k.copy(deepcopy=True)
        self.n_0 = self.n_k.copy(deepcopy=True)

        # weak formulation
        self.phi = Function(self.config.V)
        self.v = TestFunction(self.config.V)
        F_phi = ((self.phi - self.phi_k) / self.config.dt * self.v * dx
                 + inner(grad(self.phi), grad(self.v)) * dx +
                 (self.phi ** 3 - self.phi + self.config.eps * self.n_k - self.config.mu) * self.v * dx)
        J_phi = derivative(F_phi, self.phi)

        self.n = Function(self.config.V)
        F_n = ((self.n - self.n_k) / self.config.dt * self.v * dx
               + (1 / self.config.M_sig) * D(self.phi, self.n) * inner(grad(self.n) + self.config.eps * self.n * grad(self.phi), grad(self.v)) * dx)
        J_n = derivative(F_n, self.n)

        #free_energy = (-(phi**2)/2 + (phi**4)/4 + config.sigma/2*inner(grad(phi), grad(phi)) -
        #               config.mu*phi + n*(ufl.ln(n)-1) + config.eps*n*phi)*dx

        # set up of the solvers
        problem_phi = NonlinearVariationalProblem(F_phi, self.phi, J=J_phi)
        self.solver_phi = NonlinearVariationalSolver(problem_phi)

        problem_n = NonlinearVariationalProblem(F_n, self.n, J=J_n)
        self.solver_n = NonlinearVariationalSolver(problem_n)

        self.solver_phi.parameters["newton_solver"]["maximum_iterations"] = 10
        self.solver_n.parameters["newton_solver"]["maximum_iterations"] = 10

        if self.write_xdmf:
            self.init_xdmf_file()

    def init_xdmf_file(self):
        self.phi_xdmf = XDMFFile("phi_solution.xdmf")
        self.n_xdmf = XDMFFile("n_solution.xdmf")

        self.phi_xdmf.parameters["flush_output"] = True
        self.phi_xdmf.parameters["functions_share_mesh"] = True
        self.n_xdmf.parameters["flush_output"] = True
        self.n_xdmf.parameters["functions_share_mesh"] = True

        self.write_xdmf_file(self.phi_k, self.n_k, 0)

    def write_xdmf_file(self, phi, n, t):
        self.phi_xdmf.write(phi, t)
        self.n_xdmf.write(n, t)

    @staticmethod
    def kullback_measure(f, f_0):
        f_normalised = f / (assemble(f * dx))
        f_0_normalised = f_0 / (assemble(f_0 * dx))
        return assemble(f_normalised*ufl.ln(f_normalised/f_0_normalised)*dx)

    @staticmethod
    def distance_measure(f, f_0):
        return assemble((f-f_0)**2*dx)/(assemble(f_0*dx) * assemble(f*dx))

    def solve(self):
        iterates_phi = [self.phi_0]
        iterates_n = [self.n_0]
        grad_phi_vals = []

        for i in range(self.config.num_steps):
            t = (i+1)*self.config.dt
            self.solver_phi.solve()
            self.phi_k.assign(self.phi)
            iterates_phi.append(self.phi_k.copy(deepcopy=True))

            self.solver_n.solve()
            self.n_k.assign(self.n)
            iterates_n.append(self.n_k.copy(deepcopy=True))
            if self.write_xdmf:
                self.write_xdmf_file(self.phi_k, self.n_k, t)
            grad_phi_vals.append(assemble(inner(grad(self.phi), grad(self.phi)) * dx))

        distance_measure_n = self.distance_measure(self.n_0, self.n)
        distance_measure_phi = self.distance_measure(self.phi_0, self.phi)
        kullback_measure_n = self.kullback_measure(self.n_0, self.n)


        return iterates_phi, iterates_n, distance_measure_n, distance_measure_phi, kullback_measure_n


