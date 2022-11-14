import os
import sys
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns
import trackpy as tp
import pims
import numpy as np
from skimage import io
import pandas as pd
from scipy import stats
from statistics import mean, pstdev
import plotly.express as px
import plotly.graph_objects as go
from pymicro.view.vol_utils import compute_affine_transform
import glob


def arg_parser():
    """
    Defining parser for options in command line
    Returns the parser
    """
    parser = ArgumentParser(description='\n\n\n\t\t########################\n\n'
                                        '\t\tPICT DETECTOR FOR IN SITU MODELING OF PROTEIN COMPLEXES\n\n'
                                        '\n\n\t\t########################\n\n'
                                        '\tThis is a python program designed to calculate distances between '
                                        'fluorophores '
                                        'with a precision of 2 nm. From 2D stacks (channels Red/Green), it performs:\n '
                                        '\t\t1. Preprocessing (Background Subtraction and Median Filter) of the '
                                        'image.\n '
                                        '\t\t2. Chromatic aberration correction (Warping) on the red channel.\n'
                                        '\t\t3. Spot Location on the 2 channels, Refinement of detected spots, and '
                                        'linking between spots found in the 2 channels\n'
                                        '\t\t4. Distance calculation based on intensity moments, eccentricity of '
                                        'detected spots, and angle calculation\n\n')

    """parser.add_argument("-i",
                        dest="directory",
                        action="store",
                        type=str,
                        help="Input directory where the raw PICT images are located",
                        required=True)

    parser.add_argument("-o",
                        dest="output_dir",
                        action="store",
                        type=str,
                        default='output',
                        help="Name of the output directory where processed images and data will be saved")

    parser.add_argument("-b",
                        dest="beads_dir",
                        action="store",
                        type=str,
                        default='beads',
                        help="Name of the directory where the beads are for chromatic aberration correction")"""
    parser.add_argument("-i",
                        dest="path",
                        action="store",
                        type=str,
                        help="Input directory where the raw PICT images are located",
                        required=True)
    parser.add_argument('-pc',
                        dest='percentile',
                        action="store",
                        type=float,
                        default=99.7,
                        help="Percentile to select spots according to intensity")

    parser.add_argument('-cf',
                        dest='cutoff',
                        action="store",
                        type=float,
                        default=0.95,
                        help="Cutoff to reject spots brighter than the threshold")

    parser.add_argument('--test',
                        dest='test',
                        action="store_true",
                        help="Testing spot detection with the given arguments, not "
                             "performing the linking afterwards")

    parser.add_argument('-v',
                        '--verbose',
                        dest='verbose',
                        action="store_true",
                        help="Shows what the program is doing")
    return parser.parse_args()


def get_data_from_1d_grid(x, x_min, dx, indexes):
    """
    Get data from a 1d-vector with a list of indexes.

    x: vector with values in x axis.
    dx: pixel size of the grid.
    indexes: desired indexes to select the data from.
    """
    selected_data = list()
    for i in indexes:
        ix = i[0]
        x_values = x[(x > x_min + (ix * dx)) & (x < x_min + (ix + 1) * dx)]
        selected_data += list(x_values.tolist())
    return selected_data


def min_mass_selection(data, bins=1000, min_mass_cutoff=0.01):
    """
    Select those spots that fall in the area below the threshold percentage
    :param data: data in ndarray
    :param bins: bin size
    :param min_mass_cutoff: in tant per 1
    """
    # Discard spots that fall in the 5% of spots with less mass
    # with respect to the total mass in the image
    dx = (data.max() - data.min()) / bins  # jump to explore the left tail of the distribution
    ix = 1
    low_mass_area = np.sum(data[data <= (data.min() + (dx * ix))])
    low_mass_percent = low_mass_area / np.sum(data)
    cf = data.min() + dx
    if low_mass_percent < min_mass_cutoff:
        ix += 1
        while not low_mass_percent > min_mass_cutoff:
            low_mass_area = np.sum(data[data <= (data.min() + (dx * ix))])
            low_mass_percent = low_mass_area / np.sum(data)
            cf = data.min() + (dx * ix)
            ix += 1
    return np.sort(data[data <= cf])


def select_mass_cdf(data, bins=100, min_mass_cutoff=0.01, max_mass_cutoff=0.90, debug=False, verbose=False):
    """
    Estimates the probability density function (PDF) and the cumulative probability
    density function (CDF) of a univariate datapoints. In this case, the function helps
    to reject the m % of spots that are too bright (default: 0.85).

    data: 1d-array of values which are the mass of detected spots.
    bins: smooth parameter to calculate the density (sum of density ~ 1)
    max_mass_cutoff: Selecting spots below this threshold (in tant per 1)
    min_mass_cutoff: Selecting spots above this threshold (in tant per 1)
    """
    # Low-mass spots selection
    low_mass_spots = min_mass_selection(data, bins=1000, min_mass_cutoff=min_mass_cutoff)
    # Use a gaussian kernel to estimate the density
    kernel = stats.gaussian_kde(data, bw_method="silverman")
    positions = np.linspace(data.min(), data.max(), bins)
    dx = positions[1] - positions[0]
    pdf = kernel(positions) * dx
    total_pdf = np.sum(pdf)
    if 0.98 < total_pdf <= 1:
        if verbose:
            print("Sum pdf = {}\n".format(bins, total_pdf))
        pass
    else:
        if debug:
            print("\n++ DEBUG: Optimizing bin size...\n"
                  "\t+ Original bin size = {}\n"
                  "\t+ Sum pdf = {}\n".format(bins, total_pdf))
        search_bin_step = - 5
        while not 0.98 < total_pdf <= 1:
            if bins <= 10:
                break
            bins += search_bin_step
            # print(bins)
            positions = np.linspace(data.min(), data.max(), bins)
            dx = positions[1] - positions[0]
            pdf = kernel(positions) * dx
            total_pdf = np.sum(pdf)
            if total_pdf > 1:
                search_bin_step = 2
        if debug:
            print("\n++ DEBUG: Bin size optimized!\n"
                  "\t+ New bin size = {}\n"
                  "\t+ Sum pdf = {}\n".format(bins, total_pdf))

    density_sorted = np.copy(pdf)
    density_sorted[::-1].sort()
    cum = np.cumsum(density_sorted)
    # Select indexes based on cumulative probability cutoff
    sel_idx = [np.where(pdf == index) for index in density_sorted[cum <= max_mass_cutoff]]
    selected = np.sort(np.asarray(get_data_from_1d_grid(data, np.min(positions), dx, sel_idx)))
    selected = selected[selected >= low_mass_spots.max()]
    return selected


def calculate_distances(df_1, df_2):
    """
    Calculate distances (in nm) between coloc. spots
    """
    return np.sqrt(
        (df_1.x.to_numpy() - df_2.x.to_numpy()) ** 2 + (df_1.y.to_numpy() - df_2.y.to_numpy()) ** 2) * 64.5


def plot_mass(path_to_save, df, image_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    hue_order = ['sel', 'non-sel']
    sns.histplot(data=df, x="mass", hue="selected", palette=["red", "black"], kde=True, alpha=0.2,
                 stat="count", fill=True, ax=ax1, hue_order=hue_order)
    sns.kdeplot(x=df["mass"], y=df["ecc"], fill=True, thresh=0.05, cbar=False, ax=ax2,
                bw_method="silverman", hue_order=hue_order)
    sns.scatterplot(data=df, x="mass", y="ecc", hue="selected", palette=["red", "black"], alpha=0.2,
                    size="selected", sizes=(100, 50), ax=ax2, hue_order=hue_order)
    plt.tight_layout()
    # plt.show()
    plt.savefig(path_to_save + "mass_selection_{}.png".format(image_name))


def plot_distance_distribution(path_to_save, df, image_name):
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.set(font_scale=3)
    ax.set_title("Beads Distances BEFORE Warping\n"
                 "mean = {}\n"
                 "Stdev = {}\n".format(df.distances.mean(), df.distances.std()), fontweight="bold", size=25)
    sns.histplot(data=df, x="distances", kde=True, ax=ax, fill=True)
    ax.axvline(x=df.distances.mean(), color='red', ls='--', lw=2.5, alpha=0.1)
    # plt.show()
    plt.savefig(path_to_save + "distances_{}.png".format(image_name))


def plot_warping(path_to_beads, chose_frame, bead_image, df_f1, df_f2, test=True):
    """
    Method to register and transform W1 beads (mov) to W2 (ref)
    and plot the results in html interactive images.
    """
    # Plot initial coordinates of W1 and W2 and a line indicating pairs
    fig = px.imshow(bead_image[:, :, 0], color_continuous_scale='gray',
                    title="<b>Frame 0 with spots after warping<br>")
    fig.update_layout(coloraxis_showscale=False)  # to hide color bar
    fig.add_scatter(x=df_f1["y"], y=df_f1["x"],
                    mode="markers",
                    marker_symbol="circle-dot",
                    marker=dict(color="red", size=10,
                                line=dict(width=2,
                                          color='red')),
                    name="W1",
                    opacity=0.7)
    fig.add_scatter(x=df_f2["y"], y=df_f2["x"],
                    mode="markers",
                    marker_symbol="square-dot",
                    marker=dict(color="green", size=10,
                                line=dict(width=2,
                                          color='green')),
                    name="W2",
                    opacity=0.7)

    coords_W1 = list(zip(df_f1.y.tolist(), df_f1.x.tolist()))
    coords_W2 = list(zip(df_f2.y.tolist(), df_f2.x.tolist()))
    for i in range(len(coords_W1)):
        y1, y2 = coords_W1[i][0], coords_W2[i][0]
        x1, x2 = coords_W1[i][1], coords_W2[i][1]
        fig.add_trace(go.Scatter(x=[y1, y2], y=[x1, x2],
                                 mode="lines",
                                 line=go.scatter.Line(color="black"),
                                 name="{}".format(i), showlegend=False))
    fig.write_html(path_to_beads + "selection_test/linked_spots/initial_coordinates_{}.html".format(chose_frame))

    # ALIGNMENT (REGISTRATION)
    new_points = registration(path_to_beads, chose_frame, df_f2.x.tolist(), df_f2.y.tolist(), df_f1.x.tolist(),
                              df_f1.y.tolist(), test)

    # PLOT NEW POINTS IN IMAGE AFTER TRANSFORMATION
    fig.add_scatter(x=new_points[:, 1], y=new_points[:, 0],
                    mode="markers",
                    marker_symbol="x-dot",
                    marker=dict(color="orange", size=10,
                                line=dict(width=2,
                                          color='orange')),
                    name="W1_rarped",
                    opacity=0.7)
    fig.write_html(path_to_beads + "selection_test/linked_spots/coordinates_after_warping_{}.html".format(chose_frame))


def save_html_selected_detection(path_to_save, spots_df, image_name, ndimage, percentile, min_mass_cutoff,
                                 max_mass_percent):
    """
    Display selected and non-selected spots in an interactive image
    and save image in html format
    Parameters
    ----------
    image_name: channel name: "W1" or "W2"
    percentile: percentile for selecting bright spots
    max_mass_percent: to reject those m% of spots that are too bright (clusters).
    min_mass_cutoff: discard low mass spots below this threshold
    spots_df: dataframe with spots coordinates for a given image
    ndimage: image (ndarray)
    path_to_save: path to save image
    """
    # Check path and / or create it
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    for f in range(ndimage.shape[0]):
        channel_name = "W1" if f == 0 else "W2"
        # Selected and non selected spots do display on images
        selected = spots_df[(spots_df["frame"] == f) & (spots_df["selected"] == "sel")]
        non_selected = spots_df[(spots_df["frame"] == f) & (spots_df["selected"] == "non-sel")]
        # Foo note information for the image
        foo_note = "<br>Number of Selected spots: {}<br>" \
                   "Number of Non-selected spots: {}<br>" \
                   "Percentile: {}%<br>" \
                   "Max mass cutoff: {}%<br>" \
                   "Low mass cutoff: {}%<br>".format(selected.shape[0], non_selected.shape[0],
                                                     percentile, round((1 - max_mass_percent), 2) * 100,
                                                     min_mass_cutoff * 100)

        fig_label_cont = px.imshow(ndimage[f, :, :], color_continuous_scale='gray',
                                   title="<b>Image {} <br> Frame {} - "
                                         "Spot Selection</b><br>{}".format(channel_name, image_name, foo_note))
        fig_label_cont.update_layout(coloraxis_showscale=False)  # to hide color bar

        # Plot Selected & Non - selected spots with custom hover information
        fig_label_cont.add_scatter(x=selected["y"], y=selected["x"],
                                   mode="markers",
                                   marker_symbol="circle-open-dot",
                                   marker=dict(color="green", size=15,
                                               line=dict(width=2,
                                                         color='floralwhite')),
                                   name="selected",
                                   opacity=0.4,
                                   customdata=np.stack(([selected["mass"],
                                                         selected["size"],
                                                         selected["ecc"],
                                                         selected["signal"]]), axis=1),
                                   hovertemplate=
                                   '<b>x: %{x: }</b><br>'
                                   '<b>y: %{y: } <b><br>'
                                   '<b>mass: %{customdata[0]: }<b><br>'
                                   '<b>size: %{customdata[1]: }<b><br>'
                                   '<b>ecc:  %{customdata[2]: }<b><br>'
                                   '<b>SNR:  %{customdata[3]: }<b><br>')

        fig_label_cont.add_scatter(x=non_selected["y"], y=non_selected["x"],
                                   mode="markers",
                                   marker_symbol="circle-open-dot",
                                   marker=dict(color="red", size=10,
                                               line=dict(width=2,
                                                         color='floralwhite')),
                                   name="selected",
                                   opacity=0.4,
                                   customdata=np.stack(([non_selected["mass"],
                                                         non_selected["size"],
                                                         non_selected["ecc"],
                                                         non_selected["signal"]]), axis=1),
                                   hovertemplate=
                                   '<b>x: %{x: }</b><br>'
                                   '<b>y: %{y: } <b><br>'
                                   '<b>mass: %{customdata[0]: }<b><br>'
                                   '<b>size: %{customdata[1]: }<b><br>'
                                   '<b>ecc:  %{customdata[2]: }<b><br>'
                                   '<b>SNR:  %{customdata[3]: }<b><br>')

        fig_label_cont.write_html(path_to_save + "selected_{}_{}.html".format(image_name, channel_name))
        fig_label_cont.data = list()


def save_html_detection(path_to_save, spots_df, image_name, ndimage, percentile):
    """
    Display detected spots in an interactive image
    and save image in html format. Spots df does
    not have the "selected" column.
    Parameters
    ----------
    image_name: channel name: "W1" or "W2"
    percentile: percentile for selecting bright spots
    spots_df: dataframe with spots coordinates for a given image
    ndimage: image (ndarray)
    path_to_save: path to save image
    """
    # Check path and / or create it
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    for f in range(ndimage.shape[0]):
        channel_name = "W1" if f == 0 else "W2"
        spots_df_channel = spots_df[spots_df["frame"] == f]

        # Foo note information for the image
        foo_note = "<br>Number of Detected spots: {}<br>" \
                   "Percentile: {}%<br>".format(spots_df_channel.shape[0], percentile)

        fig_label_cont = px.imshow(ndimage[f, :, :], color_continuous_scale='gray',
                                   title="<b>Image {} <br> Frame {} - "
                                         "Spot Selection</b><br>{}".format(channel_name, image_name, foo_note))
        fig_label_cont.update_layout(coloraxis_showscale=False)  # to hide color bar

        # Plot Selected & Non - selected spots with custom hover information
        fig_label_cont.add_scatter(x=spots_df_channel["y"], y=spots_df_channel["x"],
                                   mode="markers",
                                   marker_symbol="circle-open-dot",
                                   marker=dict(color="green", size=15,
                                               line=dict(width=2,
                                                         color='floralwhite')),
                                   name="selected",
                                   opacity=0.4,
                                   customdata=np.stack(([spots_df_channel["mass"],
                                                         spots_df_channel["size"],
                                                         spots_df_channel["ecc"],
                                                         spots_df_channel["signal"]]), axis=1),
                                   hovertemplate=
                                   '<b>x: %{x: }</b><br>'
                                   '<b>y: %{y: } <b><br>'
                                   '<b>mass: %{customdata[0]: }<b><br>'
                                   '<b>size: %{customdata[1]: }<b><br>'
                                   '<b>ecc:  %{customdata[2]: }<b><br>'
                                   '<b>SNR:  %{customdata[3]: }<b><br>')

        fig_label_cont.write_html(path_to_save + "detected_{}_{}.html".format(image_name, channel_name))
        fig_label_cont.data = list()


def create_beads_stacks(path_to_beads):
    """
    Split beads W1 and W2 in individual frames and stack
    frame_x_W1 with frame_x_W2 and save in path directory
    """
    # Only stack frames if the number of stacked frames in directory is not 4
    # which means that, since we have 4 frames (tetrastack), we should have 4 pairs
    if len(glob.glob(path_to_beads + "frame*.tif")) != 4:
        print('\tCreating stacks W1-W2...\n ')
        beads_W1 = io.imread(path_to_beads + "W1.tif")
        beads_W2 = io.imread(path_to_beads + "W2.tif")
        for f in range(beads_W2.shape[2]):
            frame_pair = np.stack([beads_W1[:, :, f], beads_W2[:, :, f]])
            io.imsave(path_to_beads + "frame_{}.tif".format(f), frame_pair, plugin="tifffile", check_contrast=False)
    else:
        pass


def detection(path_to_beads, image_name, ndimage, pims_frames, percentile, min_mass_cutoff, max_mass_cutoff,
              verbose=False, test=False):
    """
    Spot selection using Trackpy.

    image_name: name to save image in directory
    ndimage: image frame in a ndarray format
    pims_frames: PIMS frame/s
    percentile:
    min_mass_cutoff:
    max_mas_cutoff:
    """
    # SPOT DETECTION
    f = tp.batch(pims_frames[:], diameter=11, percentile=percentile, engine='python')
    f.loc[:, "ID"] = list(range(1, f.shape[0] + 1))
    f = f.rename(columns={"x": "y", "y": "x"})  # change x,y order
    f.loc[:, 'size'] = f['size'].apply(lambda x: x ** 2)  # remove sqrt from size formula
    # Select spots with a cumulative density probability less than a threshold
    test_mass_selection = select_mass_cdf(f.mass.to_numpy(), bins=100, min_mass_cutoff=min_mass_cutoff,
                                          max_mass_cutoff=max_mass_cutoff, verbose=verbose, debug=False)
    f['selected'] = np.where(f.mass.isin(test_mass_selection), "sel", "non-sel")
    if verbose:
        print("\nInitial number of spots detected by trackpy: {}\n"
              "Number of spots discarded regarding mass: {}\n"
              "Final number of selected spots: {}\n\n".format(f.shape[0],
                                                              len(f[
                                                                      f["selected"] ==
                                                                      "non-sel"]),
                                                              len(test_mass_selection)))
    # SAVING RESULTS
    if test:
        if not os.path.exists(path_to_beads + "selection_test/detected_spots/"):
            os.mkdir(path_to_beads + "selection_test/detected_spots/")
        # PLOT: (mass according to selection) and html with selected spots for each channel
        plot_mass(path_to_beads + "selection_test/detected_spots/", f, image_name)
        save_html_selected_detection(path_to_beads + "selection_test/detected_spots/", f, image_name, ndimage, pc,
                                     max_mass_cutoff, min_mass_cutoff)
    else:
        # Plot selected spots
        plot_mass(path_to_beads, f, image_name)
        save_html_selected_detection(path_to_beads + "detected_spots/", f, image_name, ndimage, percentile,
                                     min_mass_cutoff, max_mass_cutoff)
    return f


def linking(path_to_beads, f_batch_selection, image_name, ndarray, percentile, min_mass_cutoff, max_mass_cutoff,
            test=False):
    """
    Linking (pairing) detected particles from two channels (0/red/W1 and 1/green/W2)
    Selecting only paired detections between two channels and saving selections to files.
    """
    t = tp.link(f_batch_selection, search_range=2, pos_columns=["x", "y"])
    t = t.sort_values(by=["particle", "frame"])  # sort particle according to "particle" and "frame"
    t = t[t.duplicated("particle", keep=False)]  # select paired only
    f_batch = f_batch_selection.copy()
    f_batch.loc[:, "selected"] = np.where(f_batch.ID.isin(t.ID.tolist()), "sel",
                                          "non-sel")
    # PLOT selected after linking the two channels
    save_html_selected_detection(path_to_beads + "linked_spots/", f_batch, image_name, ndarray,
                        percentile, min_mass_cutoff, max_mass_cutoff)

    # Separate frame0 and frame1 in two df
    t_W1 = t[t["frame"] == 0]
    t_W2 = t[t["frame"] == 1]

    # Save coordinates in separate csv files
    t_W1[["x", "y"]].to_csv(path_to_beads + "linked_spots/" +
                            "detected_{}_W1.csv".format(image_name.split("_")[1]),
                            sep="\t",
                            encoding="utf-8", header=True, index=False)
    t_W2[["x", "y"]].to_csv(path_to_beads + "linked_spots/" +
                            "detected_{}_W2.csv".format(image_name.split("_")[1]),
                            sep="\t",
                            encoding="utf-8", header=True, index=False)
    # PLOT INITIAL DISTANCE DISTRIBUTION
    t_W1, t_W2 = t_W1.copy(), t_W2.copy()
    t_W1.loc[:, "distances"] = calculate_distances(t_W1, t_W2)
    t_W2.loc[:, "distances"] = calculate_distances(t_W1, t_W2)
    if test:
        if not os.path.exists(path_to_beads + "selection_test/linked_spots/"):
            os.mkdir(path_to_beads + "selection_test/linked_spots/")
        plot_distance_distribution(path_to_beads + "selection_test/linked_spots/", t_W1, image_name)
    else:
        if not os.path.exists(path_to_beads + "linked_spots/"):
            os.mkdir(path_to_beads + "linked_spots/")
        plot_distance_distribution(path_to_beads + "linked_spots/", t_W1, image_name)

    return t_W1, t_W2


def registration(path_to_beads, chose_frame, ref_x, ref_y, mov_x, mov_y, test=False):
    """
    Method to apply registration (alignment) of a mov coordinates
    to a ref coordinates. This method uses pymicro functions to
    compute and affine transform matrix for the alignment.
    Returns the list of new coordinates (W1_warped) after the alignment
    """
    # ALIGNMENT (REGISTRATION)
    ref = np.asarray(list(zip(ref_x, ref_y)))
    mov = np.asarray(list(zip(mov_x, mov_y)))
    # compute the AFFINE transform from the point set
    translation, transformation = compute_affine_transform(ref, mov)
    # TRANSFORMATION
    ref_centroid = np.mean(ref, axis=0)
    mov_centroid = np.mean(mov, axis=0)
    new_points = np.empty_like(ref)
    for i in range(len(ref)):
        new_points[i] = ref_centroid + np.dot(transformation, mov[i] - mov_centroid)
    # Save transformation matrix
    if test:
        np.save(path_to_beads + 'selection_test/transform_{}.npy'.format(chose_frame.split("_")[1]), transformation)
    else:
        np.save(path_to_beads + 'transform.npy', transformation)
        np.save(path_to_beads + 'mov_centroid.npy', mov_centroid)
        np.save(path_to_beads + 'ref_centroid.npy', ref_centroid)

    # CALCULATE NEW DISTANCE DISTRIBUTION AND PLOT
    if test:
        # CALCULATE NEW DISTANCES FROM W1_WARPED TO W2
        new_distances = np.sqrt(
            (ref[:, 0] - new_points[:, 0]) ** 2 + (ref[:, 1] - new_points[:, 1]) ** 2) * 64.5
        # PLOT NEW DISTANCE DISTRIBUTION
        fig, ax = plt.subplots(figsize=(15, 15))
        sns.set(font_scale=3)
        ax.set_title("Beads Distances AFTER Warping\n"
                     "mean = {}\n"
                     "Stdev = {}\n".format(np.mean(new_distances), np.std(new_distances)), fontweight="bold", size=25)
        sns.histplot(data=new_distances, kde=True, ax=ax, fill=True)
        ax.axvline(x=np.mean(new_distances), color='red', ls='--', lw=2.5, alpha=0.1)
        # plt.show()
        plt.savefig(path_to_beads + "selection_test/linked_spots/distances_after_warping_{}.png".format(chose_frame))

    return new_points


def test_bead_detection(path_to_beads, choose_frame, percentile, max_mass_cutoff, min_mass_cutoff, verbose=False):
    """
    Testing on bead detection for a chosen frame (see dictionary).
    Plot mass distribution for selected and non-selected spots and
    show results in html interactive image.
    """
    beads_dict = {
        "frame_0": 0,
        "frame_1": 1,
        "frame_2": 2,
        "frame_3": 3
    }
    # Create folder for test selection of beads based on intensity
    if not os.path.exists(path_to_beads + "selection_test/"):
        os.mkdir(path_to_beads + "selection_test/")
    # Create stack [W1_x, W2_x]
    beads_W1 = io.imread(path_to_beads + "W1.tif")
    beads_W2 = io.imread(path_to_beads + "W2.tif")
    f = beads_dict[choose_frame]  # # choose frame for testing
    frame_pair = np.stack([beads_W1[:, :, f], beads_W2[:, :, f]])
    io.imsave(path_to_beads + "selection_test/" + "original_{}.tif".format(choose_frame), frame_pair,
              plugin="tifffile",
              check_contrast=False)

    # SELECTION based on percentile and cutoff for max/min intensity
    name_test = choose_frame
    frames_test = pims.open(path_to_beads + "selection_test/" + "original_{}.tif".format(choose_frame))
    # SPOT DETECTION
    f_batch_test = detection(path_to_beads, name_test, frame_pair, frames_test, percentile, min_mass_cutoff,
                             max_mass_cutoff, verbose, test=True)

    # LINK SELECTION
    f_batch_test_sel = f_batch_test[f_batch_test['selected'] == "sel"].drop(columns=["selected"])
    t_test_filtered_W1, t_test_filtered_W2 = linking(path_to_beads, f_batch_test_sel, name_test, frame_pair,
                                                     percentile, min_mass_cutoff, max_mass_cutoff, test=True)

    # WARPING
    plot_warping(path_to_beads, choose_frame, beads_W2, t_test_filtered_W1, t_test_filtered_W2)


def beads_registration(path_to_beads, percentile, min_mass_cutoff, max_mass_cutoff, verbose=False):
    """
    Main function to run beads registration and create transformation matrix to
    warp coordinates for chromatic aberration using green/W2 as reference channel
    and red/W1 as the channel to warp/correct.

    The warping is coordinates based. First, trackpy is used for spot detection,
    rejecting those spots too bright or with low intensity (see detection). Following,
    detected spots from W1 and W2 are paired (linking) using trackpy, only keeping
    spots that are paired. Finally, coordinates from the 4 FOV of each channel are collected
    and used for warping, using functionalities of pymicro
    (https://pymicro.readthedocs.io/projects/pymicro/en/latest/cookbook/pointset_registration.html)

    For testing these methods, one can run the test_bead_detection method. In either case,
    plots about spot detection and selection are saved in directories, as well as coordinates
    before and after detection/linking/warping and the final transformation matrix to use
    later on.

    Args:
        path_to_beads: path where W1.tif and W2.tif files are located
        percentile
        min_mass_cutoff
        max_mass_cutoff

    """
    if verbose:
        print("\n#############################\n"
              "     BEADS REGISTRATION \n"
              "#############################\n")
    x_coords_W1 = list()
    y_coords_W1 = list()
    x_coords_W2 = list()
    y_coords_W2 = list()
    for img in glob.glob(path_to_beads + "frame*.tif"):
        # READ IMAGE, set name and image number, use PIMS to read frames
        name = img.split("/")[-1].split(".")[0]
        frames = pims.open(img)

        # SPOT DETECTION
        f_batch = detection(path_to_beads, name, io.imread(img), frames, percentile, min_mass_cutoff, max_mass_cutoff,
                            verbose)

        # LINK SELECTION
        f_batch_sel = f_batch[f_batch['selected'] == "sel"].drop(columns=["selected"])
        paired_df_W1, paired_df_W2 = linking(path_to_beads, f_batch_sel, name, io.imread(img), percentile,
                                             min_mass_cutoff, max_mass_cutoff)

        # Append x and y coordinates to lists
        x_coords_W1 += paired_df_W1.x.tolist()
        y_coords_W1 += paired_df_W1.y.tolist()
        x_coords_W2 += paired_df_W2.x.tolist()
        y_coords_W2 += paired_df_W2.y.tolist()

    # REGISTRATION & TRANSFORMATION
    ref_coords = np.asarray(list(zip(x_coords_W2, y_coords_W2)))
    mov_coords = np.asarray(list(zip(x_coords_W1, y_coords_W1)))
    # compute the affine transform from the point set
    new_coords = registration(path_to_beads, "_all", x_coords_W2, y_coords_W2, x_coords_W1, y_coords_W1)

    # PLOT ORIGINA vs NEW COORDINATES
    original_distances = np.sqrt(
        (ref_coords[:, 0] - mov_coords[:, 0]) ** 2 + (ref_coords[:, 1] - mov_coords[:, 1]) ** 2) * 64.5
    new_distances = np.sqrt(
        (ref_coords[:, 0] - new_coords[:, 0]) ** 2 + (ref_coords[:, 1] - new_coords[:, 1]) ** 2) * 64.5
    fig, ax = plt.subplots(figsize=(25, 15))
    sns.set(font_scale=3)
    ax.set_title("Beads Distances AFTER Warping\n\n"
                 "mean before = {} nm; stdev before = {} nm\n"
                 "mean after = {} nm; stdev after = {} nm \n".format(np.around(np.mean(original_distances), 2),
                                                                     np.around(np.std(original_distances), 2),
                                                                     np.around(np.mean(new_distances), 2),
                                                                     np.around(np.std(new_distances)), 2),
                 fontweight="bold", size=25)
    sns.histplot(data=original_distances, kde=True, color="sandybrown", ax=ax, fill=True)
    sns.histplot(data=new_distances, kde=True, ax=ax, color="cornflowerblue", fill=True)
    ax.set_xlabel("$Distances \ (nm) $", fontsize=45, labelpad=30)
    ax.set_ylabel("$Count $", fontsize=45, labelpad=30)
    ax.axvline(x=np.mean(original_distances), color='sandybrown', ls='--', lw=2.5, alpha=0.8)
    ax.axvline(x=np.mean(new_distances), color='cornflowerblue', ls='--', lw=2.5, alpha=0.8)
    plt.savefig(path_to_beads + "linked_spots/" + "distances_after_warping_only.png")


if __name__ == "__main__":
    # options = arg_parser()
    # path = options.path
    # pc = options.percentile
    # cutoff = options.cutoff
    path = "../260922_test_warping/F9_test/input/beads/"
    pc = 98.0  # in tant per cent
    max_mass = 0.95  # in tant per 1
    min_mass = 0.05  # in tant per 1
    # Separate frames and create pair stack of beads
    create_beads_stacks(path)

    test = True
    verbose = True
    if test:
        if verbose:
            print("\n\n⁺⁺⁺TESTING MODE ON⁺⁺⁺\n\n")
        frame = input("\n\nChoose frame to test:\n"
                      "frame_0\nframe_1\nframe_2\nframe_3\n\n\t:")
        test_bead_detection(path, frame, pc, max_mass, min_mass, verbose=verbose)
        sys.exit()
    else:
        if verbose:
            print("\n\n⁺⁺⁺BEADS DETECTION, PAIRING, WARPING⁺⁺⁺\n\n")
        beads_registration(path, pc, min_mass, max_mass, verbose=verbose)
        sys.exit()

# END
