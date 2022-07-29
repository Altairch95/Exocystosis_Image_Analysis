#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python functions to run outlier rejection of measured distances
"""
import sys
import options as opt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rnd import pdf, cdf
from lle import bootstrap, LL


def outlier_rejection():
    """
    Method to run outlier rejection of the distances'
    distribution, based on maximizing the likelihood
    estimate for "mu" and "sigma" to follow a distribution
    as described in Eq. 4 (Churchman, 2004).
    """
    print("\n\n####################################\n"
          "Initializing Outlier rejection \n"
          "########################################\n\n")
    # Read data from KDE to select distances
    df = pd.read_csv(opt.results_dir + "kde/W1_kde_sel.csv",
                     usecols=["ID", "x", "y", "img", "distances"])
    if opt.Picco:
        df_andrea = pd.read_csv(opt.results_dir + "picco_after_gaussian.csv",
                                usecols=["dist"])
    distances_alta = np.array(df.distances.tolist())
    if opt.Picco:
        distances_andrea = np.array(df_andrea.dist.tolist()) * 64.5

    # INPUT PARAMETERS
    np.random.seed(5)  # important when creating random datasets to ensure the results is always the same
    mu0 = 20  # true mu for testing the outlier rejection
    sigma0 = 10  # true sigma for testing the outlier rejection
    x0 = [20, 20]  # mu / sigma --> initial guess for "mu" and "sigma"

    # Create function to fit, based on Churchman eq. 4
    c = cdf(mu0, sigma0)

    # compute the MLE on my data
    mu_alta, sigma_alta, sh_alta, i_min_alta, i_max_alta, n_alta = bootstrap(LL, x0=x0, d=distances_alta)

    if opt.Picco:
        # compute the MLE on Picco's data
        mu_picco, sigma_picco, sh_picco, i_min_picco, i_max_picco, n_apicco = bootstrap(LL, x0=x0, d=distances_andrea)

    # output a summary plot
    f = plt.figure()
    plt.hist(distances_alta, density=True, color='blue', alpha=0.5, label='My Data')
    if opt.Picco:
        plt.hist(distances_andrea, density=True, color='orange', alpha=0.5, label="Andrea's Data")
    # Plot Alta's
    plt.plot(c[0], pdf(c[0], mu_alta[i_max_alta][0], sigma_alta[i_max_alta][0]), color='blue',
             label='Bootstrap to Altair data:\n'
                   '$\mu=$' + str(round(mu_alta[i_max_alta][0], 2)) + '$\pm$' + str(round(mu_alta[i_max_alta][1], 2))
                   + 'nm, $\sigma=$' +
                   str(round(sigma_alta[i_max_alta][0], 2)) + '$\pm$' + str(round(sigma_alta[i_max_alta][1], 2)) + 'nm'
                   + ", n=" + str(n_alta))
    if opt.Picco:
        # Plot Picco
        plt.plot(c[0], pdf(c[0], mu_picco[i_max_picco][0], sigma_picco[i_max_picco][0]), color='orange',
                 label='Bootstrap to Andrea data:\n'
                       '$\mu=$' + str(round(mu_picco[i_max_picco][0], 2)) + '$\pm$'
                       + str(round(mu_picco[i_max_picco][1], 2)) + 'nm, $\sigma=$' +
                       str(round(sigma_picco[i_max_picco][0], 2)) + '$\pm$'
                       + str(round(sigma_picco[i_max_picco][1], 2)) + 'nm'
                       + ", n=" + str(n_apicco))
    plt.xlabel('$\mu$ (nm)')
    plt.ylabel('Density')
    plt.legend()

    # Plot scores based on Shannon Entropy
    if opt.Picco:
        inset_alta = f.add_axes([0.5, 0.225, 0.25, 0.10])
    else:
        inset_alta = f.add_axes([0.5, 0.205, 0.35, 0.20])
    inset_alta.plot([mu_alta[i][0] for i in range(len(sh_alta))], sh_alta)
    inset_alta.set_xlabel('$\mu$ (nm)')
    inset_alta.set_ylabel('S Alta')

    if opt.Picco:
        inset_andrea = f.add_axes([0.5, 0.45, 0.25, 0.10])
        inset_andrea.plot([mu_picco[i][0] for i in range(len(sh_picco))], sh_picco)
        inset_andrea.set_xlabel('$\mu$ (nm)')
        inset_andrea.set_ylabel('S Andrea')

    # plt.legend()
    plt.savefig(opt.figures_dir + "distances_{}.pdf".format(opt.dataset))


if __name__ == "__main__":
    print("Functions for outlier rejection :)\n")
    sys.exit(0)
