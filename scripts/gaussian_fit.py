#!/usr/bin/python3.7
# coding=utf-8
"""
Python functions to perform Gaussian fitting
to GFP & RFP spots of radius = 5 px
"""
import os
import sys
import time
import logging
import glob
import options as opt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from skimage.io import imread, imsave
from lmfit import Parameters, minimize


# ======================
# GAUSSIAN FUNCTIONS
# ======================


def gaussian2D(x, y, cen_x, cen_y, sig_x, sig_y, offset):
    """
    Defining gaussian function to 2D data
    :param x:
    :param y:
    :param cen_x:
    :param cen_y:
    :param sig:
    :param offset:
    """
    return np.exp(-(((cen_x - x) / sig_x) ** 2 + ((cen_y - y) / sig_y) ** 2) / 2.0) + offset


def residuals(p, x, y, z):
    height = p["height"].value
    cen_x = p["centroid_x"].value
    cen_y = p["centroid_y"].value
    sigma_x = p["sigma_x"].value
    sigma_y = p["sigma_y"].value
    offset = p["offset"].value
    return z - height * gaussian2D(x, y, cen_x, cen_y, sigma_x, sigma_y, offset)


def clean_spot_boundaries(df, sub_df, image, radius):
    """
    Method to remove those spots close to the boundary of the image,
    from where it cannot be fitted to a gaussian distribution
    Parameters
    ----------
    df: raw dataframe
    sub_df: sub-dataframe of x,y coordinates
    image: image W1 or W2 as ndarray
    radius: mask radius to explore boundaries

    Returns
    -------
    cleaned dataframe
    """
    coords_to_remove = set()
    for coord in zip(sub_df.x.tolist(), sub_df.y.tolist()):
        y, x = int(coord[0]), int(coord[1])
        y_lower, y_upper = y - radius, y + radius
        x_lower, x_upper = x - radius, x + radius
        if y_lower < 0 or x_lower < 0 or y_upper > image.shape[0] or x_upper > image.shape[1]:
            print("Dropping coord {}".format(coord))
            df = df.drop(df[df.x == coord[0]].index)
            sub_df = sub_df.drop(sub_df[sub_df.x == coord[0]].index)
            coords_to_remove.add(coord[0])
    return df, sub_df


def slice_spot(image, coord, r=5, margin=0):
    """
    Slice spot in image by cropping with a radius and
    a margin.
    :param image: ndarray
    :param r: radius of the spot
    :param coord: array with x,y coordinates
    :param margin: margin to enlarge the spot by this 'margin'
    """
    try:
        coord = np.array(coord).astype(int)
        # y, x = coord[0], coord[1]
        # y_lower, y_upper = y - r, y + r
        # x_lower, x_upper = x - r, x + r
        # if y_lower > 0 or x_lower > 0 or y_upper < image.shape[0] or x_upper < image.shape[1]:
        mask = np.ones((r * 2 + margin, r * 2 + margin), dtype=np.int8)
        rect = tuple([slice(c - r, c + r) for c, r in zip(coord, (r, r))])
        slide = np.array(image[rect])
        return mask * slide
    except ValueError as ve:
        sys.stderr.write("{}".format(ve))


def gaussian_fit(spot):
    """
    Perform a gaussian fitting to normalized intensities in
    spot and evaluate the goodness of the fit with
    the R^2 value
    """
    # Create grid for x,y coordinates in spot
    xmin, xmax = 0, len(spot)
    ymin, ymax = 0, len(spot)
    x, y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))

    # Set parameter for initial gaussian function
    initial = Parameters()
    initial.add("height", value=1., vary=False)  # height of the gaussian
    initial.add("centroid_x", value=5.)  # centroid x
    initial.add("centroid_y", value=5.)  # centroid y
    initial.add("sigma_x", value=2.5)  # centroid sigma
    initial.add("sigma_y", value=2.5)  # centroid sigma
    initial.add("offset", value=0., vary=False)  # centroid offset of the gaussian

    # Fit the intensity values to a gaussian function
    # residuals is the function to minimize with the initial function and the arg parameters
    fit = minimize(residuals, initial, args=(x, y, spot))
    # Calculate and return r-squared values to estimate goodness of the fit
    # Also return the sigma_x / sigma_y values, which gives us info about how round a spot is
    # Round spots should have a sigma ratio ~ 1.
    return 1 - fit.residual.var() / np.var(spot), fit.params["sigma_x"].value / fit.params["sigma_y"].value


def save_html_gaussian(path_to_save, channel_image, sub_df, img_num, channel_name):
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
    if not os.path.exists(path_to_save + "gaussian_fit/"):
        os.mkdir(path_to_save + "gaussian_fit/")

    selected = sub_df[sub_df["selected"] == "sel"]
    non_selected = sub_df[sub_df["selected"] == "non-sel"]
    percent_sel = round(len(selected) * 100 / (len(selected) + len(non_selected)), 3)
    foo_note = "<br>Number of Selected spots: {} / {} (Percentage = {} %)<br><br>".format(len(selected),
                                                                                          len(selected) + len(
                                                                                              non_selected),
                                                                                          percent_sel)
    # Create figure with lines to closest contour and closest neighbour
    fig_label_cont = px.imshow(channel_image, color_continuous_scale='gray',
                               title="<b>ImageMD {} {} - Gaussian selected</b><br>{}".format(img_num,
                                                                                             channel_name, foo_note))
    fig_label_cont.update_layout(coloraxis_showscale=False)  # to hide color bar


    # Plot spots with custom hover information
    fig_label_cont.add_scatter(x=selected["y"], y=selected["x"],
                               mode="markers",
                               marker=dict(color="green", size=7),
                               name="W1",
                               customdata=np.stack(([selected["r2_gaussian"],
                                                     selected["sigma_ratio"]]), axis=1),
                               hovertemplate=
                               '<b>x: %{x: }</b><br>'
                               '<b>y: %{y: } <b><br>'
                               '<b>r2_gauss: %{customdata[0]: }<b><br>'
                               '<b>sigma_ratio: %{customdata[1]: }<b><br>')

    fig_label_cont.add_scatter(x=non_selected["y"], y=non_selected["x"],
                               mode="markers",
                               marker=dict(color="red", size=7),
                               name="W2",
                               customdata=np.stack(([non_selected["r2_gaussian"],
                                                     non_selected["sigma_ratio"]]), axis=1),
                               hovertemplate=
                               '<b>x: %{x: }</b><br>'
                               '<b>y: %{y: } <b><br>'
                               '<b>r2_gauss: %{customdata[0]: }<b><br>'
                               '<b>sigma_ratio: %{customdata[1]: }<b><br>')
    # Save figure in output directory
    fig_label_cont.write_html(path_to_save + "gaussian_fit/" + "image_{}_{}.html".format(img_num, channel_name))


def main_gaussian():
    """
    2) Main method to run gaussian fitting on spots sorted from
    yeast segmentation method. Selection is based on R^2 (goddness
    of the gaussian fit).
    """
    print("#############################\n"
          " Gaussian Fitting Selection \n"
          "#############################\n")
    logging.info("\n\n####################################\n"
                 "Initializing Gaussian Fitting Selection \n"
                 "########################################\n\n")
    percent_sel_total_W1 = list()
    percent_sel_total_W2 = list()
    percent_sel_total = list()
    total_data = 0
    total_selected = 0
    r2_cutoff = 0.75  # quality of spots above this r2 value
    # sigma_r_lower = 0.9  # lw sigma ratio  (optional, not optimized)
    # sigma_r_upper = 1.10  # lw sigma ratio (optional, not optimized)
    if os.path.exists(opt.results_dir + "segmentation/") and \
            len(os.listdir(opt.results_dir + "segmentation/")) != 0:
        for img_file in glob.glob(opt.images_dir + "imageMD*"):
            start = time.time()
            image_number = img_file.split("/")[-1].split("_")[-1].split(".")[0]
            print("Processing image {} ...\n".format(image_number))
            # Read image and separate frames
            image = imread(opt.images_dir + "imageMD_{}.tif".format(image_number))
            W1 = image[0]
            W2 = image[1]

            # Load spot coordinates for W1 and W2
            spots_df_W1 = pd.read_csv(opt.results_dir + "segmentation/" +
                                      "detected_seg_{}_W1.csv".format(image_number))  # Spot coordinates W1
            spots_df_W1.loc[:, "img"] = image_number
            spots_df_W2 = pd.read_csv(opt.results_dir + "segmentation/" +
                                      "detected_seg_{}_W2.csv".format(image_number))  # Spot coordinates W2
            spots_df_W2.loc[:, "img"] = image_number
            total_data += spots_df_W1.shape[0]

            # Slice dataframe with columns of interest (this is just for debugging)
            sub_df_W1 = spots_df_W1.loc[:, ["x", "y", "ID"]]
            sub_df_W2 = spots_df_W2.loc[:, ["x", "y", "ID"]]

            spots_df_W1, sub_df_W1 = clean_spot_boundaries(spots_df_W1, sub_df_W1, W1, radius=5)
            spots_df_W2, sub_df_W2 = clean_spot_boundaries(spots_df_W2, sub_df_W2, W2, radius=5)

            # Get coordinates for image and slice spots
            coords_W1 = list(zip(sub_df_W1.x.tolist(), sub_df_W1.y.tolist()))
            coords_W2 = list(zip(sub_df_W2.x.tolist(), sub_df_W2.y.tolist()))
            spots_W1 = [slice_spot(W1, coord) for coord in coords_W1]
            spots_W2 = [slice_spot(W2, coord) for coord in coords_W2]

            # Fit spot's normalized intensities to a gaussian distribution (see gaussian_fit function documentation)
            # Gaussian fit return: R^2 and sigma_ratio (sigma_r ~ 1 if the spot is round)
            r2_W1, sigma_r_W1 = list(
                zip(*[gaussian_fit((spot - spot.min()) / (spot.max() - spot.min())) for spot in spots_W1]))
            r2_W2, sigma_r_W2 = list(
                zip(*[gaussian_fit((spot - spot.min()) / (spot.max() - spot.min())) for spot in spots_W2]))

            # Add R^2 and sigma ratio to dataframes
            sub_df_W1.loc[:, "r2_gaussian"], sub_df_W1.loc[:, "sigma_ratio"] = r2_W1, sigma_r_W1
            sub_df_W2.loc[:, "r2_gaussian"], sub_df_W2.loc[:, "sigma_ratio"] = r2_W2, sigma_r_W2

            # Spot selection based on goodness of the fit, tested manually in different spots
            sub_df_W1.loc[:, 'selected'] = np.where(sub_df_W1["r2_gaussian"] > r2_cutoff, "sel", "non-sel")
            sub_df_W2.loc[:, 'selected'] = np.where(sub_df_W2["r2_gaussian"] > r2_cutoff, "sel", "non-sel")
            # Get percentage of selection for W1 and W2
            percent_sel_W1 = len(sub_df_W1[sub_df_W1["selected"] == "sel"]) * 100 / sub_df_W1.shape[0]
            percent_sel_W2 = len(sub_df_W2[sub_df_W2["selected"] == "sel"]) * 100 / sub_df_W2.shape[0]

            # Pair selected in W1 & W2
            selection_df_paired_W1 = spots_df_W1.loc[(sub_df_W1["selected"] == "sel") &
                                                     (sub_df_W2["selected"] == "sel")]
            selection_df_paired_W2 = spots_df_W2.loc[(sub_df_W2["selected"] == "sel") &
                                                     (sub_df_W1["selected"] == "sel")]
            # Assert shape W1 == shape W2
            assert selection_df_paired_W1.shape == selection_df_paired_W2.shape

            # write to log percentage of selection
            num_selected = selection_df_paired_W1.shape[0]
            percent_sel = num_selected * 100 / spots_df_W1.shape[0]
            logging.info("\nImage {} --> {:02} / {:02} "
                         "spots selected.. --> {} %".format(image_number, num_selected, len(spots_df_W1), percent_sel))
            total_selected += num_selected

            # Save figure with selected and non-selected spots based on goodness of the gaussian fit
            save_html_gaussian(opt.figures_dir, W1, sub_df_W1, image_number, "W1")
            save_html_gaussian(opt.figures_dir, W2, sub_df_W2, image_number, "W2")

            # Save df as csv: gaussian.csv
            if not os.path.exists(opt.results_dir):
                os.mkdir(opt.results_dir)
            if not os.path.exists(opt.results_dir + "gaussian_fit/"):
                os.mkdir(opt.results_dir + "gaussian_fit/")
            sub_df_W1.to_csv(opt.results_dir + "gaussian_fit/" +
                             "all_gauss_{}_{}.csv".format(image_number, "W1"),
                             sep=",", encoding="utf-8", header=True, index=False)
            sub_df_W2.to_csv(opt.results_dir + "gaussian_fit/" +
                             "all_gauss_{}_{}.csv".format(image_number, "W2"),
                             sep=",", encoding="utf-8", header=True, index=False)
            selection_df_paired_W1.to_csv(opt.results_dir + "gaussian_fit/" +
                                          "detected_gauss_{}_{}.csv".format(image_number, "W1"),
                                          sep=",", encoding="utf-8", header=True, index=False)
            selection_df_paired_W2.to_csv(opt.results_dir + "gaussian_fit/" +
                                          "detected_gauss_{}_{}.csv".format(image_number, "W2"),
                                          sep=",", encoding="utf-8", header=True, index=False)

            # Append percentages to list to write in report (log.txt)
            percent_sel_total_W1.append(percent_sel_W1)
            percent_sel_total_W2.append(percent_sel_W2)
            percent_sel_total.append(percent_sel)
            total_time = time.time() - start
            print("Image {} processed in {} s\n".format(image_number, round(total_time, 3)))

        logging.info("\n\nTotal Percent W1 --> {} %\n"
                     "Total Percent W2 --> {} %\n\n"
                     "Total Paired Percent --> {} % \n".format(sum(percent_sel_total_W1) / len(percent_sel_total_W1),
                                                               sum(percent_sel_total_W2) / len(percent_sel_total_W2),
                                                               sum(percent_sel_total) / len(percent_sel_total)))
        #####################################
        # PLOT SELECTED SPOTS AFTER GAUSSIAN
        #####################################
        # Load data ensuring that W1 & W2 are paired
        df_W1 = pd.concat(map(pd.read_csv, sorted(glob.glob(opt.results_dir + "gaussian_fit/all*W1*"))),
                          ignore_index=True)
        df_W2 = pd.concat(map(pd.read_csv, sorted(glob.glob(opt.results_dir + "gaussian_fit/all*W2*"))),
                          ignore_index=True)
        # Combine W1&W2 data into a df and label selected
        df_data = pd.concat([df_W1.r2_gaussian.rename("r2_W1"), df_W2.r2_gaussian.rename("r2_W2")], axis=1)
        df_data.loc[:, 'selected'] = np.where((df_W1["r2_gaussian"] >= r2_cutoff) & (df_W2["r2_gaussian"] >= r2_cutoff),
                                              "sel", "non-sel")
        # Plot values in the R^2 space
        fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 8))
        sns.scatterplot(data=df_data, x="r2_W1", y="r2_W2", hue="selected", palette=["red", "black"], alpha=0.6,
                        s=50, zorder=10, ax=ax)
        ax.set_title("Goodness of the Gaussian Fit", fontweight="bold", size=20)
        ax.set_ylabel("$W2 \ R^{2}$", fontsize=20)
        ax.set_xlabel("$W1 \ R^{2}$", fontsize=20)
        ax.set(xlim=(0, 1))
        ax.set(ylim=(0, 1))

        fig.savefig(opt.figures_dir + "gaussian.png", dpi=150)

        return total_data, total_selected


if __name__ == "__main__":
    print("Gaussian Fit functions :)\n")
    sys.exit(0)
