import matplotlib.pyplot as plt
import numpy as np
from dolfin import *
import csv

class FigureHandler():

    def __init__(self, config):
        """
        Initialises object
        :param cfg: config data
        """
        self.config = config
        plt.rcParams.update({
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        })

    def individual_plots(self, phi_solutions, n_solutions, timestamps, savefig=False):
        """
        Shows one heatmap plot for each phi and n solution.
        :param phi_solutions: List of phi_solutions
        :param n_solutions: List of n_solutions
        :param timestamps: corresponding times of the solutions
        :param savefig: whether to save the figure
        """
        path = "Results/png/individual"
        cmap_phi = "coolwarm_r"
        cmap_n = "binary"
        for k in range(len(phi_solutions)):
            plt.figure()
            fig_phi = plot(phi_solutions[k], mode="color")
            fig_phi.set_cmap(cmap_phi)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"t={round(timestamps[k], 2)}")
            plt.colorbar(fig_phi)
            if savefig:
                plt.savefig(f"{path}/phi_t={round(timestamps[k], 2)}.png")
            else:
                plt.show()

            plt.figure()
            fig_n = plot(n_solutions[k], mode="color")
            fig_n.set_cmap(cmap_n)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"t={round(timestamps[k], 2)}")
            plt.colorbar(fig_n)
            if savefig:
                plt.savefig(f"{path}/n_t={round(timestamps[k], 2)}.png")
            else:
                plt.show()


    def two_by_two_plot(self, phi, phi_0, n, n_0):
        """
        Shows the 2 by 2 plot of the initial and final phi and n. The colorbar is not rescaled,
        it is the same for both the initial and final heatmap
        :param phi: final phi
        :param phi_0: initial phi
        :param n: final n
        :param n_0: initial n
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
        cmap = "coolwarm_r"

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
        axes[0, 1].set_title(f"Solution at t={self.config.T} (order parameter)")
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
        axes[1, 1].set_title(f"Solution at t={self.config.T} (distribution)")
        axes[1, 1].set_xlabel("x")
        axes[1, 1].set_ylabel("y")

        cbar2 = fig.colorbar(c3, ax=[axes[1, 0], axes[1, 1]], orientation='vertical', shrink=0.8)
        cbar2.set_label("Distribution")

        plt.show()
        plt.close(fig)


    def free_energy_plot(self, free_energy_vals, savefig=False):
        """
        Plot the time evolution of the free energy function
        :param free_energy_vals: Values of the free energy
        :param savefig: whether to save the figure
        """
        path = "Results/png/free_energy"
        plt.figure(figsize=(12, 6))
        plt.plot([self.config.dt * k for k in range(self.config.num_steps)], free_energy_vals)
        plt.xlabel("Time")
        plt.ylabel("Free energy")
        plt.title("Evolution of free energy over time")
        if savefig:
            plt.savefig(f"{path}/free_energy.png")
        else:
            plt.show()


    def horizont_slice_n_plot(self, n_solutions, timestamps, ycoord, num_x_points, savefig=False):
        """
        Takes a horizontal slice of the 2d solution and plots the distribution along this line
        :param n_solutions: ink distribution solutions
        :param timestamps: corresponding time point of the n solutions
        :param ycoord: y-coord where the horizontal slice starts
        :param num_x_points: number of points along the horizontal slice
        :param savefig: whether to save the figure
        """
        path = "Results/png/slice"
        xvals = np.linspace(0, self.config.L, num_x_points)
        n_vals = []
        plt.figure()
        for i in range(len(n_solutions)):
            n_vals.append([n_solutions[i](xvals[j], ycoord) for j in range(len(xvals))])
            plt.plot(xvals, n_vals[i], label=f"t={round(timestamps[i], 2)}")
        plt.xlabel("x")
        plt.ylabel("n")
        plt.legend()
        if savefig:
            plt.savefig(f"{path}/slice_plot.png")
        else:
            plt.show()

    @staticmethod
    def read_csv(name):
        with open(f"{name}.csv", "r") as f:
            reader = csv.reader(f)
            return [float(row[0]) for row in reader]


    def mu_dependence_plots(self, mu_list, path_distance_phi, path_distance_n, path_kullback_n):
        distance_phi_list = self.read_csv(path_distance_phi)
        distance_n_list = self.read_csv(path_distance_n)
        kullback_n_list = self.read_csv(path_kullback_n)

        plt.figure()
        plt.plot(mu_list, distance_phi_list, label="distance\nmeasure phi")
        plt.legend(loc="lower right")
        plt.show()

        plt.figure()
        plt.plot(mu_list, distance_n_list, label="distance\nmeasure n")
        plt.legend(loc="upper left")
        plt.show()

        plt.figure()
        plt.plot(mu_list, kullback_n_list, label="kullback\nmeasure n")
        plt.legend(loc="upper left")
        plt.show()




