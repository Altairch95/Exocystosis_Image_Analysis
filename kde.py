#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python script for spot selection using 2D-Kernel Density Estimate
"""

import os
import sys
import glob
import options as opt
import logging
import time

from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from math import sqrt
from skimage.io import imread, imsave

from IPython.display import display_html
from itertools import chain, cycle


# ==========
# KDE FUNCTIONS
# ==========

def display_side_by_side(*args, titles=cycle([''])):
    html_str = ''
    for df, title in zip(args, chain(titles, cycle(['</br>']))):
        html_str += '<th style="text-align:center"><td style="vertical-align:top">'
        html_str += f'<center><h2>{title}</h2></center>'
        html_str += df.to_html().replace('table', 'table style="display:inline"')
        html_str += '</td></th>'
    display_html(html_str, raw=True)


def read_csv_2(file):
    """
    Function to read multiple csv (input data)
    """
    if opt.Picco:
        cols = ["img", "x", "y", "m0", "m2", "m11", "m20", "m02"]
    elif opt.Altair:
        cols = ["img", "x", "y", "mass", "size", "ecc"]
    df = pd.read_csv(file, usecols=cols)
    return df


def calculate_distances(df_1, df_2):
    """
    Calculate distances (in nm) between coloc. spots
    """
    return np.sqrt(
        (df_1["x"].to_numpy() - df_2["x"].to_numpy()) ** 2 + (df_1["y"].to_numpy() - df_2["y"].to_numpy()) ** 2) * 64.5


def get_data_from_grid(x, y, dx, dy, indexes):
    """
    Get data from a 2d-grid with a list of indexes.

    x: vector with values in x axis.
    y: vector with values in y axis.
    dx, dy: pixel size of the grid for x and y axis.
    indexes: desired indexes to select the data from.
    """
    selected_data = list()
    for i in indexes:
        xmin = x.min()
        ymin = y.min()
        ix = i[0]
        iy = i[1]
        # Do a mask for x and y values
        x_values = x[
            (x > xmin + (ix * dx)) & (x < xmin + (ix + 1) * dx) & ((y > ymin + (iy * dy)) & (y < ymin + (iy + 1) * dy))]
        y_values = y[
            (x > xmin + (ix * dx)) & (x < xmin + (ix + 1) * dx) & ((y > ymin + (iy * dy)) & (y < ymin + (iy + 1) * dy))]
        selected_data += list(zip(x_values.tolist(), y_values.tolist()))
    return selected_data


def load_data_for_kde():
    """
    Method to load data and prepared it for running the KDE.

    Returns
    -------
    dataframe W1, dataframe W2, dataframe for processing and plotting
    """
    # ====================
    # Gathering the data
    # =====================
    # Load data ensuring that W1 & W2 are paired
    df_W1 = pd.concat(map(read_csv_2, sorted(glob.glob(opt.results_dir + "gaussian_fit/detected_*W1*"))),
                      ignore_index=True)
    df_W2 = pd.concat(map(read_csv_2, sorted(glob.glob(opt.results_dir + "gaussian_fit/detected_*W2*"))),
                      ignore_index=True)

    # Add ID to each data point (spot) so the paired spots are paired
    df_W1["ID"] = list(range(1, df_W1.shape[0] + 1))
    df_W2["ID"] = list(range(1, df_W2.shape[0] + 1))

    # Calculate eccentricity
    if opt.Picco:
        df_W1["ecc"] = df_W1.apply(lambda row: (row.mu20 + row.mu02 +
                                                sqrt((row.mu20 - row.mu02) ** 2 + (4 * row.mu11 ** 2))) / (
                                                       row.mu20 + row.mu02 - sqrt(
                                                   (row.mu20 - row.mu02) ** 2 + (4 * row.mu11 ** 2))), axis=1)
        df_W2["ecc"] = df_W2.apply(
            lambda row: (row.mu20 + row.mu02 + sqrt((row.mu20 - row.mu02) ** 2 + (4 * row.mu11 ** 2))) / (
                    row.mu20 + row.mu02 - sqrt((row.mu20 - row.mu02) ** 2 + (4 * row.mu11 ** 2))), axis=1)

    # Calculate distances
    df_W1["distances"] = calculate_distances(df_W1, df_W2)
    df_W2["distances"] = calculate_distances(df_W1, df_W2)

    # Reduce the table to columns of interest
    if opt.Picco:
        df_W1 = df_W1.loc[:, ['ID', 'x', 'y', 'm2', 'ecc', 'img', 'distances']]
        df_W2 = df_W2.loc[:, ['ID', 'x', 'y', 'm2', 'ecc', 'img', 'distances']]
    elif opt.Altair:
        df_W1 = df_W1.loc[:, ['ID', 'x', 'y', 'size', 'ecc',
                              'img', 'distances']].rename(columns={"size": "m2"})
        df_W2 = df_W2.loc[:, ['ID', 'x', 'y', 'size', 'ecc',
                              'img', 'distances']].rename(columns={"size": "m2"})

    # Create df with m2, ecc values for each channel
    df_data = pd.concat([df_W1.ID, df_W1.m2.rename("m2_W1"), df_W1.ecc.rename("ecc_W1"),
                         df_W2.m2.rename("m2_W2"), df_W2.ecc.rename("ecc_W2")], axis=1)
    print("\nData collected from Gaussian selected spots!\n"
          "Total spots: {}\n".format(df_data.shape[0]))

    return df_W1, df_W2, df_data


def kde(df_W1, df_W2, df_data):
    """
    Method to run KDE from already gathered data.
    Parameters
    ----------
    df_W1: loaded data for W1
    df_W2: loaded data for W2
    df_data: dataframe with data of interest for the analysis

    Returns
    -------
    dataframes after processing
    """
    print("\nKDE using Silverman's method for clustering...\n")
    # ==========================
    # Kernel Density Estimation
    # ==========================
    # Data as x and y vectors
    W1x, W1y = df_data.m2_W1, df_data.ecc_W1
    W2x, W2y = df_data.m2_W2, df_data.ecc_W2
    data_xmin, data_xmax = min(W1x.min(), W2x.min()), max(W1x.max(), W2x.max())
    data_ymin, data_ymax = min(W1y.min(), W2y.min()), max(W1y.max(), W2y.max())

    # create a dense multi-dim mesh grid (of 100 x 100)
    X, Y = np.mgrid[data_xmin:data_xmax:50j, data_ymin:data_ymax:50j]
    positions = np.vstack([X.ravel(), Y.ravel()])  # stack the 2D in a 1D array
    dx = X[1][0] - X[0][0]
    dy = Y[0][1] - Y[0][0]

    # KDE using silverman's method
    values_W1 = np.vstack([df_W1.m2, df_W1.ecc])  # W1 values (m2 & ecc)
    values_W2 = np.vstack([df_W2.m2, df_W2.ecc])  # W2 values (m2 & ecc)

    kernel_W1 = stats.gaussian_kde(values_W1, bw_method="silverman")
    kernel_W2 = stats.gaussian_kde(values_W2, bw_method="silverman")

    # Create grid of probabilities for W1 and W2 kernels
    grid_W1 = np.reshape(kernel_W1(positions).T, X.shape) * dx * dy  # grid with probabilities (should sum ~ 1)
    grid_W2 = np.reshape(kernel_W2(positions).T, X.shape) * dx * dy

    # Sort the grid by probabilities in descending order
    grid_W1_sorted = np.copy(grid_W1.ravel())
    grid_W1_sorted[::-1].sort()
    grid_W2_sorted = np.copy(grid_W2.ravel())
    grid_W2_sorted[::-1].sort()

    # Cumulative probability
    W1_cum = np.cumsum(grid_W1_sorted)
    W2_cum = np.cumsum(grid_W2_sorted)

    # Selected region: indexes with a cumulative probability below of 0.5 (descending order = pr > 50%)
    W1_sel_idx = [np.where(grid_W1 == index) for index in grid_W1_sorted[W1_cum <= 0.6]]
    W2_sel_idx = [np.where(grid_W2 == index) for index in grid_W2_sorted[W2_cum <= 0.6]]

    # Select data values within selected region for W1 and W2
    selected_W1 = np.asarray(get_data_from_grid(values_W1[0], values_W1[1], dx, dy, W1_sel_idx))
    selected_W2 = np.asarray(get_data_from_grid(values_W2[0], values_W2[1], dx, dy, W2_sel_idx))

    # Get IDs of paired W1 and W2 falling in the selected region
    selected_W1_ID = set(
        [df_data.loc[(df_data['m2_W1'] == m2) & (df_data['ecc_W1'] == ecc), 'ID'].iloc[0] for m2, ecc in selected_W1])
    selected_W2_ID = set(
        [df_data.loc[(df_data['m2_W2'] == m2) & (df_data['ecc_W2'] == ecc), 'ID'].iloc[0] for m2, ecc in selected_W2])
    selected_IDs = selected_W1_ID.intersection(selected_W2_ID)

    # Label dataset with a "Selected" column
    df_W1['selected'] = np.where(df_W1["ID"].isin(selected_IDs), "sel", "non-sel")
    df_W2['selected'] = np.where(df_W2["ID"].isin(selected_IDs), "sel", "non-sel")
    df_data['selected'] = np.where(df_W1["ID"].isin(selected_IDs), "sel", "non-sel")

    # Create new dataframes only with selected spots
    df_W1_sel = df_W1.loc[(df_W1['selected'] == "sel")].drop(columns=["selected"])
    df_W2_sel = df_W2.loc[(df_W2['selected'] == "sel")].drop(columns=["selected"])

    # Write percentages to log file
    for img in set(df_W1.img):
        num_initial = len(df_W1[df_W1["img"] == img])
        num_selected = len(df_W1_sel[df_W1_sel["img"] == img])
        percent_sel = num_selected * 100 / num_initial
        logging.info("\nImage {} --> {:02} / {:02} "
                     "spots selected.. --> {} %".format(img, num_selected, num_initial, percent_sel))

    percent_total = df_W1_sel.shape[0] * 100 / df_W1.shape[0]

    logging.info("\n\nTotal Paired Percent --> {} %\n".format(percent_total))
    print(("\n\nTotal Paired Percent --> {} %\n".format(percent_total)))

    # Save selected spots in a csv file
    if not os.path.exists(opt.results_dir + "kde/"):
        os.mkdir(opt.results_dir + "kde/")
    df_W1_sel.to_csv(opt.results_dir + "kde/W1_kde_sel.csv", sep=",", encoding="utf-8", header=True, index=False)
    df_W2_sel.to_csv(opt.results_dir + "kde/W2_kde_sel.csv", sep=",", encoding="utf-8", header=True, index=False)

    return df_W1, df_W2, df_data


def plot_kde(df_W1, df_W2, df_data):
    """
    Method to plot KDE results.
    Parameters
    ----------
    df_W1
    df_W2
    df_data
    """
    print("\nPlotting KDE...\n")
    # Plot m2,ecc values for W1 and W2, and compare values between channels
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout=True, figsize=(15, 15))

    # ax1 - W1 (m2 vs ecc)
    ax1.set_title("W1", fontweight="bold", size=20)
    ax1.set_ylabel("ecc_W1", fontsize=20)
    ax1.set_xlabel("m2_W1", fontsize=20)

    sns.kdeplot(x=df_W1["m2"], y=df_W1["ecc"], shade=True, thresh=0.05, cbar=False, ax=ax1,
                bw_method="silverman")
    sns.scatterplot(data=df_W1, x="m2", y="ecc", hue="selected", palette=["red", "black"], alpha=0.6,
                    s=35, zorder=10, ax=ax1)

    # ax2 - W1 vs W2 (ecc)
    ax2.set_title("ecc W1 vs W2", fontweight="bold", size=20)
    ax2.set_ylabel("ecc_W1", fontsize=20)
    ax2.set_xlabel("ecc_W2", fontsize=20)

    sns.kdeplot(x=df_W1["ecc"], y=df_W2["ecc"], shade=True, thresh=0.05, cbar=False, ax=ax2,
                bw_method="silverman")
    sns.scatterplot(data=df_data, x="ecc_W1", y="ecc_W2", hue="selected", palette=["red", "black"], alpha=0.6,
                    s=35, zorder=10, ax=ax2)

    # ax3 - W2 (m2 vs ecc)
    ax3.set_title("W2", fontweight="bold", size=20)
    ax3.set_ylabel("ecc_W2", fontsize=20)
    ax3.set_xlabel("m2_W2", fontsize=20)

    sns.kdeplot(x=df_W2["m2"], y=df_W2["ecc"], shade=True, thresh=0.05, cbar=False, ax=ax3,
                bw_method="silverman")
    sns.scatterplot(data=df_W2, x="m2", y="ecc", hue="selected", palette=["red", "black"], alpha=0.6,
                    s=35, zorder=10, ax=ax3)

    # ax4 - W1 vs W2 (m2)
    ax4.set_title("m2 W1 vs W2", fontweight="bold", size=20)
    ax4.set_ylabel("m2_W1", fontsize=20)
    ax4.set_xlabel("m2_W2", fontsize=20)

    sns.kdeplot(x=df_W1["m2"], y=df_W2["m2"], shade=True, thresh=0.05, cbar=False, ax=ax4,
                bw_method="silverman")
    sns.scatterplot(data=df_data, x="m2_W1", y="m2_W2", hue="selected", palette=["red", "black"], alpha=0.6,
                    s=35, zorder=10, ax=ax4)

    if opt.Picco:
        ax1.set_xlim([2, 10])
        ax1.set_ylim([0.5, 5])
        ax2.set_xlim([0.5, 5])
        ax2.set_ylim([0.5, 5])
        ax3.set_xlim([2, 10])
        ax3.set_ylim([0.5, 5])
        ax4.set_xlim([2, 10])
        ax4.set_ylim([2, 10])

    elif opt.Altair:
        ax1.set_xlim([2, 10])
        # ax1.set_ylim([0.0, 0.7])
        # ax2.set_xlim([0.0, 0.7])
        # ax2.set_ylim([0.0, 0.7])
        ax3.set_xlim([2, 10])
        # ax3.set_ylim([0.0, 0.7])
        ax4.set_xlim([2, 10])
        ax4.set_ylim([2, 10])

    plt.savefig(opt.figures_dir + "KDE.png", dpi=72)


def save_html_kde(path_to_save, channel_image, sub_df, img_num, channel_name):
    """
    Method to save image with scattered spots as html
    Parameters
    ----------
    channel_image: ndimage corresponding to channel W1 or W2
    path_to_save: path to save figure in html
    sub_df: sub-dataframe to work with
    img_num: image number
    channel_name: channel name ("W1" or "W2")

    """
    if not os.path.exists(path_to_save + "kde/"):
        os.mkdir(path_to_save + "kde/")

    selected = sub_df[sub_df["selected"] == "sel"]
    non_selected = sub_df[sub_df["selected"] == "non-sel"]
    percent_sel = round(len(selected) * 100 / (len(selected) + len(non_selected)), 3)
    foo_note = "<br>Number of Selected spots: {} / {} (Percentage = {} %)<br><br>".format(len(selected),
                                                                                          len(selected) + len(
                                                                                              non_selected),
                                                                                          percent_sel)

    # Create figure with lines to the closest contour and closest neighbour
    fig_label_cont = px.imshow(channel_image, color_continuous_scale='gray',
                               title="<b>ImageMD {} {} - KDE selected</b><br>{}".format(img_num,
                                                                                        channel_name, foo_note))
    fig_label_cont.update_layout(coloraxis_showscale=False)  # to hide color bar

    # fig_label_cont.update_layout(
    #     title={
    #         'text': "imageMD_{}_{} - KDE selected".format(img_num, channel_name),
    #         'y': 0.93,
    #         'x': 0.5,
    #         'xanchor': 'center',
    #         'yanchor': 'top'})

    # Plot spots with custom hover information
    fig_label_cont.add_scatter(x=selected["y"], y=selected["x"],
                               mode="markers",
                               marker=dict(color="green", size=7),
                               name="W1",
                               customdata=np.stack(([selected["ecc"],
                                                     selected["m2"]]), axis=1),
                               hovertemplate=
                               '<b>x: %{x: }</b><br>'
                               '<b>y: %{y: } <b><br>'
                               '<b>ecc: %{customdata[0]: }<b><br>'
                               '<b>m2: %{customdata[1]: }<b><br>')

    fig_label_cont.add_scatter(x=non_selected["y"], y=non_selected["x"],
                               mode="markers",
                               marker=dict(color="red", size=7),
                               name="W2",
                               customdata=np.stack(([non_selected["ecc"],
                                                     non_selected["m2"]]), axis=1),
                               hovertemplate=
                               '<b>x: %{x: }</b><br>'
                               '<b>y: %{y: } <b><br>'
                               '<b>ecc: %{customdata[0]: }<b><br>'
                               '<b>m2: %{customdata[1]: }<b><br>')
    # Save figure in output directory
    fig_label_cont.write_html(path_to_save + "kde/" + "image_{}_{}.html".format(img_num, channel_name))


def main_kde():
    """
    3) Main method to run 2D-KDE based on the spot properties
     "second momentum of intensity" and "eccentricity".
    Selection is based on cropping the area of higher density that falls
    in the m2 & ecc space, accumulating a probability of => 50 %
    """
    start = time.time()
    logging.info("\n\n####################################\n"
                 "Initializing KDE Selection \n"
                 "########################################\n\n")
    print("\n\n####################################\n"
          "Initializing KDE Selection \n"
          "########################################\n\n")
    df_W1, df_W2, df_data = load_data_for_kde()
    df_W1_final, df_W2_final, df_data_final = kde(df_W1, df_W2, df_data)
    # Save figure with selected and non-selected spots based on KDE
    for img_ in glob.glob(opt.images_dir + "image_*"):
        image_number = img_.split("/")[-1].split(".")[0].split("_")[1]
        W1 = imread(img_)[0, :, :]
        W2 = imread(img_)[1, :, :]
        df_W1_final_sub = df_W1_final[(df_W1_final["img"] == int(image_number)) & (df_W1_final["selected"] == "sel")]
        df_W2_final_sub = df_W2_final[(df_W2_final["img"] == int(image_number)) & (df_W2_final["selected"] == "sel")]
        save_html_kde(opt.figures_dir, W1, df_W1_final_sub, image_number, "W1")
        save_html_kde(opt.figures_dir, W2, df_W2_final_sub, image_number, "W2")
    plot_kde(df_W1_final, df_W2_final, df_data_final)

    total_time = time.time() - start
    print("KDE analysis done in {} s\n".format(round(total_time, 3)))

    return df_W1.shape[0], df_W1_final[df_W1_final["selected"] == "sel"].shape[0]


if __name__ == "__main__":
    print("KDE Functions :)\n")
    sys.exit(0)
