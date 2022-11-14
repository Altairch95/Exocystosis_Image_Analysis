#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python functions to run outlier rejection of measured distances
"""
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rnd import pdf, cdf
from lle import bootstrap, LL


def read_csv_2(file):
	"""
    Function to read multiple csv (input data)
    """
	cols = ["ID", "x", "y", "img", "distances"]
	df = pd.read_csv(file, usecols=cols, sep="\t")
	return df


def outlier_rejection(results_dir, figures_dir, test=False, mu_ini=20, sigma_ini=15):
	"""
    Method to run outlier rejection of the distances'
    distribution, based on maximizing the likelihood
    estimate for "mu" and "sigma" to follow a distribution
    as described in Eq. 4 (Churchman, 2004).

    mu_ini: initial guess of mu value for starting the MLE
    sigma_ini: initial guess for sigma value for starting the MLE
    """
	print("\n\n####################################\n"
		  "Initializing Outlier rejection \n"
		  "########################################\n\n")
	# Read data from KDE to select distances
	if test:
		data = pd.read_csv(results_dir + "spot_detection/distances_trackpy.csv", sep="\t")
	else:
		data = pd.concat(map(read_csv_2, sorted(glob.glob(f"{results_dir}kde/W1_kde_sel.csv"))),
						 ignore_index=True)
	distances_alta = data.distances.to_numpy()
	dataset_name = figures_dir.split("/")[-4]
	# INPUT PARAMETERS
	np.random.seed(5)  # important when creating random datasets to ensure the results is always the same
	mu0 = np.median(distances_alta)  # initial guess for mu opt parameter
	sigma0 = np.std(distances_alta)  # initial guess for sigma opt parameter
	x0 = [mu0, sigma0]  # mu / sigma --> initial guess for "mu" and "sigma"
	print("\nChoosing distance distribution median and stdev as initial values to start fitting..\n\n"
		  "\tInitial mu: {}\n"
		  "\tInitial sigma: {}\n\n"
		  "\tStarting optimization...\n".format(mu0, sigma0))

	# Create function to fit, based on Churchman eq. 4
	c = cdf(mu0, sigma0)

	# compute the MLE on my data
	mu, sigma, sh_scores, i_min, i_max, n, sel_distribution = bootstrap(results_dir, LL,
																		x0=x0, d=distances_alta,
																		reject_lower=10)

	# PLOT DISTANCE DISTRIBUTION AFTER KDE SELECTION
	# MEASURE DISTANCE DISTRIBUTION AFTER GAUSSIAN
	initial_distances = np.loadtxt(results_dir + "distances_after_warping.csv")
	np.savetxt(results_dir + "final_distances.csv", sel_distribution, delimiter="\t")
	# PLOT NEW DISTANCE DISTRIBUTION
	fig, ax = plt.subplots(figsize=(25, 15))
	sns.set(font_scale=3)
	ax.set_title("Distances after OUTLIER rejection\n\n"
				 "mean initial = {} nm; stdev initial = {} nm; n = {}\n"
				 "mean final = {} nm; stdev final = {} nm; "
				 "n = {} \n".format(np.around(np.mean(initial_distances), 2),
									np.around(np.std(initial_distances), 2),
									len(initial_distances),
									np.around(np.mean(sel_distribution), 2),
									np.around(np.std(sel_distribution), 2),
									len(sel_distribution)),
				 fontweight="bold", size=25)
	sns.histplot(data=initial_distances, kde=False, color="sandybrown", ax=ax, fill=True, stat="density")
	sns.histplot(data=sel_distribution, kde=False, ax=ax, color="tab:red", fill=True, stat="density")
	ax.set_xlabel("$Distances \ (nm) $", fontsize=45, labelpad=30)
	ax.set_ylabel("$Count $", fontsize=45, labelpad=30)
	ax.axvline(x=np.mean(initial_distances), color='sandybrown', ls='--', lw=2.5, alpha=0.8)
	ax.axvline(x=np.mean(sel_distribution), color='cornflowerblue', ls='--', lw=2.5, alpha=0.8)
	ax.plot(c[0], pdf(c[0], mu[i_max][0], sigma[i_max][0]), color='tab:blue', linewidth=6,
			label='$\mu=$' + str(round(mu[i_max][0], 2)) + '$\pm$' + str(round(mu[i_max][1], 2))
				  + 'nm, $\sigma=$' +
				  str(round(sigma[i_max][0], 2)) + '$\pm$' + str(round(sigma[i_max][1], 2)) + 'nm'
				  + ", n=" + str(n))
	plt.savefig(figures_dir + "final_distance_distribution.png")
	plt.clf()

	# OUTPUT SUMMARY: PLOT FINAL DISTANCE DISTRIBUTION WITH FIT AND SCORES
	# ax1 Distribution with fit
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40, 15))
	sns.histplot(x=sel_distribution, kde=False, ax=ax1, stat="density", legend=True, color='tab:red')
	# sns.histplot(data=data, x="distances", kde=True, ax=ax1, stat="density", legend=True, color='tab:orange')
	ax1.plot(c[0], pdf(c[0], mu[i_max][0], sigma[i_max][0]), color='tab:blue', linewidth=6,
			 label=
			 '$\mu=$' + str(round(mu[i_max][0], 2)) + '$\pm$' + str(round(mu[i_max][1], 2))
			 + 'nm, $\sigma=$' +
			 str(round(sigma[i_max][0], 2)) + '$\pm$' + str(round(sigma[i_max][1], 2)) + 'nm'
			 + ", n=" + str(n))
	ax1.axvline(x=mu[i_max][0], color='black', ls='--', lw=3, alpha=0.3)
	# ax1.set_title("Fitting Distance Distribution", fontweight="bold", size=20)
	ax1.set_xlabel('distance (nm)', fontsize=30)
	ax1.set_ylabel('Density', fontsize=30)
	ax1.set_xlim(0, max(sel_distribution))  # max(distances_alta)
	legend = ax1.legend(loc='upper right', shadow=True, fontsize=30)
	legend.get_frame().set_facecolor('C0')
	ax1.tick_params(axis='x', labelsize=25)
	ax1.tick_params(axis='y', labelsize=25)

	# Shannon Scoring Plot
	shannon_data = list(zip([mu[i][0] for i in range(len(sh_scores))], sh_scores))
	dist_list, shannon_list = list(zip(*shannon_data))

	scores_df = pd.DataFrame({"mu": dist_list,
							  "sh_score": shannon_list})
	scores_df["selected"] = np.where(scores_df.sh_score == scores_df.sh_score[i_max], "sel", "non-sel")
	hue_order = ['sel', 'non-sel']
	sns.scatterplot(data=scores_df, x="mu", y="sh_score", hue="selected", palette=["red", "black"], alpha=0.8,
					size="selected", sizes=(500, 200), ax=ax2, hue_order=hue_order)
	ax2.axhline(y=max(shannon_list), color='red', ls='--', lw=3, alpha=0.3)
	ax2.set_xlabel('distance (nm)', fontsize=30)
	ax2.set_ylabel('Scores', fontsize=30)
	sns.move_legend(
		ax1, "lower center",
		bbox_to_anchor=(.5, 1), ncol=2, title="{}\nFinal Distribution".format(dataset_name), frameon=False,
	)
	sns.move_legend(
		ax2, "lower center",
		bbox_to_anchor=(.5, 1), ncol=2, title="Bootstrap Scores", frameon=False,
	)

	# plt.legend()
	plt.tight_layout()
	plt.xticks(fontsize=25)
	plt.yticks(fontsize=25)
	plt.savefig(figures_dir + f"{dataset_name}.pdf")


if __name__ == "__main__":
	print("Functions for outlier rejection :)\n")
	sys.exit(0)
