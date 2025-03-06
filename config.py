from dolfin import *
import fenics
import numpy as np


class Config:

    def __init__(self, mu):
        self.T = 5  # Time interval length
        self.num_steps = 20  # number of time steps
        self.dt = self.T / self.num_steps
        self.mu = mu  # moisture parameter
        self.L = 25  # domain length
        self.eps = -1  # how much the ink particles like to be in the solvent
        self.sigma = 0.03 # surface tension
        self.M = 0.2
        self.M_sig = self.M * self.sigma

        self.nx, self.ny = 100, 100
        self.mesh = RectangleMesh(Point(0, 0), Point(self.L, self.L), self.nx, self.ny)
        self.V = FunctionSpace(self.mesh, "Lagrange", 1)

        self.phi_options = {
            "constant": Constant(-1),
            "tanh":  Expression("0.2*tanh(x[0]-5)", degree=2),
            "cosine": Expression("0.2*cos(pi*(x[0]+x[1])/5)", degree=2),
            "gaussian": Expression("2 * exp((-pow(x[0]-L/2,2) - pow(x[1]-L/2,2))/100)-1", L=self.L, degree=2),
            "sine_checkerboard": Expression("sin(x[0])*sin(x[1])", degree=2),
            "random": self.random_phi()
        }

        self.n_options = {
            "constant": Constant(0),
            "gaussian": Expression("1*exp((-pow(x[0]-L/2,2) - pow(x[1]-L/2,2))/100)", L=self.L, degree=2),
            "half_domain": Expression("x[0] < L/2 ? 0.2 : 0", L=self.L, degree=1),
            "two_drop": Expression("0.7*(exp(-pow(x[0]-L/4,2)/100) * exp(-pow(x[1]-L/2,2)/100) + exp(-pow(x[0]-(3*L)/4,2)/100) * exp(-pow(x[1]-L/2,2)/100))",
                        degree=2, L=self.L)

        }

    def set_ics(self, phi_option, n_option):
        if phi_option == "random":
            phi_init = self.phi_options[phi_option]
        else:
            phi_init = interpolate(self.phi_options[phi_option], self.V)
        n_init = interpolate(self.n_options[n_option], self.V)
        return phi_init, n_init

    def random_phi(self):
        phi_k = Function(self.V)
        vec = phi_k.vector()
        vec[:] = np.random.uniform(-1, 1, phi_k.vector()[:].size)
        return phi_k
