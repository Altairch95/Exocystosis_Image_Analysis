#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python functions to preprocess PICT images by cell segmentation and
spot sorting based on nearest distance to contour and closest
neighbour distance.

Cell segmentation is done using YeastSpotter software code
(https://github.com/alexxijielu/yeast_segmentation)

"""
import os
import shutil
import logging
import sys
import time
import glob

import pandas as pd
import numpy as np

import options as opt
from skimage.io import imread, imsave
import plotly.express as px
import plotly.graph_objects as go

from silence_tensorflow import silence_tensorflow

silence_tensorflow()  # Silence Tensorflow WARNINGS

from mrcnn.my_inference import predict_images
from mrcnn.preprocess_images import preprocess_images
from mrcnn.convert_to_image import convert_to_image, convert_to_imagej


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence warnings

# ======================
# SEGMENTATION FUNCTIONS
# ======================


def segment_yeast():
    """
    Method to segment yeast cells using YeastSpotter software
    (https://github.com/alexxijielu/yeast_segmentation)
    """
    if opt.segment_dir != '' and not os.path.isdir(opt.segment_dir):
        os.mkdir(opt.segment_dir)

    if os.path.isdir(opt.segment_dir):
        if len(os.listdir(opt.segment_dir)) > 0:
            logging.error("ERROR: Make sure that the output directory to save masks to is empty.")
        else:
            preprocessed_image_directory = opt.segment_dir + "preprocessed_images/"
            preprocessed_image_list = opt.segment_dir + "preprocessed_images_list.csv"
            rle_file = opt.segment_dir + "compressed_masks.csv"
            output_mask_directory = opt.segment_dir + "masks/"
            output_imagej_directory = opt.segment_dir + "imagej/"

            # Preprocess the images
            if opt.verbose:
                logging.info("\nPreprocessing your images...")
            preprocess_images(opt.images_dir,
                              preprocessed_image_directory,
                              preprocessed_image_list,
                              verbose=opt.verbose)

            if opt.verbose:
                logging.info("\nRunning your images through the neural network...")
            predict_images(preprocessed_image_directory,
                           preprocessed_image_list,
                           rle_file,
                           rescale=opt.rescale,
                           scale_factor=opt.scale_factor,
                           verbose=opt.verbose)

            if opt.save_masks:
                if opt.verbose:
                    logging.info("\nSaving the masks...")

                if opt.output_imagej:
                    convert_to_image(rle_file,
                                     output_mask_directory,
                                     preprocessed_image_list,
                                     rescale=opt.rescale,
                                     scale_factor=opt.scale_factor,
                                     verbose=opt.verbose)

                    convert_to_imagej(output_mask_directory,
                                      output_imagej_directory)
                else:
                    convert_to_image(rle_file,
                                     output_mask_directory,
                                     preprocessed_image_list,
                                     rescale=opt.rescale,
                                     scale_factor=opt.scale_factor,
                                     verbose=opt.verbose)

            os.remove(preprocessed_image_list)

            if not opt.save_preprocessed:
                shutil.rmtree(preprocessed_image_directory)

            if not opt.save_compressed:
                os.remove(rle_file)

            if not opt.save_masks:
                shutil.rmtree(output_mask_directory)


def read_csv_2(file):
    """
    Function to read multiple csv (input data)
    """
    cols = ["x", "y", "m0", "m2", "m11", "m20", "m02"]
    num = file.split("/")[-1].split("_")[2]
    df = pd.read_csv(file, sep="\s+", names=cols, usecols=[0, 1, 2, 3, 9, 10, 11])
    df["img"] = num
    return df


def get_label_coords(contour_label):
    """
    Method to get contour coordinates of labeled segmented cells
    Parameters
    ----------
    contour_label: ndimage with labeled contours

    Returns
    -------
    dictionary with label --> coordinates
    """
    unique_labels = set(contour_label[contour_label != 0].flatten())
    dict_cell_labels = dict()
    for label in unique_labels:
        label_coords = np.where(contour_label == label)
        dict_cell_labels.setdefault(label, np.array(list(zip(label_coords[0], label_coords[1]))))
    return dict_cell_labels


def clean_mother_daugther(img_num, contour_image, labels_image):
    """
    Method to avoid contour lines between mother and
    daughter cells.
    Parameters
    ----------
    img_num: image number ("01", "02", ...)
    contour_image: ndimage with cells contours (contour is 1, background is 0)
    labels_image: ndimage with labeled cells (each cell has a unique label)
    """
    # contour according to labels
    contour_label = contour_image * labels_image

    # Clean labels that are too close (> 2 px)
    # Group coordinates per cell (according to labels) in a dictionary
    dict_cell_labels = get_label_coords(contour_label)

    # Calculate distances between labels of different cells
    threshold = 2.5  # in pixels, to remove contour between two cells
    contour_to_modify = np.copy(contour_label)
    for label_1 in dict_cell_labels.keys():
        for label_2 in dict_cell_labels.keys():
            if label_1 != label_2:  # avoid calculating distances within a cell
                # Iterate over contour coords of cell-label_1 against all coords of cell-label_2
                for coord in dict_cell_labels[label_1]:
                    distances_to_coord = np.linalg.norm(coord - dict_cell_labels[label_2], axis=1)
                    # for coords fulfilling condition, replace values by a zero
                    close_labels = np.argwhere(distances_to_coord <= threshold)
                    if len(close_labels) > 0:
                        x1, x2 = coord[0], coord[1]
                        contour_to_modify[x1, x2] = 0
                        # for coords of cell-label_2 go to dict to get coordinates
                        for c in close_labels:
                            x = dict_cell_labels[label_2][c][0][0]
                            y = dict_cell_labels[label_2][c][0][1]
                            contour_to_modify[x, y] = 0
    imsave(opt.segment_dir + "masks/contour_mod_{}.tif".format(img_num), contour_to_modify, plugin="tifffile",
           check_contrast=False)


def distance_to_contour(df_spots, contour_coordinates):
    """
    Method to get the closest distance from each spot to the cell contour
    Parameters
    ----------
    contour_coordinates: coordinates of cell contour
    df_spots: dataframe with  (x,y) coordinates

    Returns
    -------
    list of ditances
    """
    # Calculate min distance to contour and contour coordinate
    distances_cont = list()
    for coord in df_spots.to_numpy():
        spot_distances = np.linalg.norm(coord - contour_coordinates, axis=1)
        min_distance = spot_distances.min()
        contour_coord = tuple(contour_coordinates[spot_distances.argmin()])
        distances_cont.append((min_distance, contour_coord))
    return distances_cont


def distance_to_neigh(df_spots):
    """
    Method to get the closest neighbour for each spot to determine
    isolated spots
    Parameters
    ----------
    df_spots

    Returns
    -------

    """
    # Calculate min distance to contour and contour coordinate
    distances_neigh = list()
    for coord in df_spots.to_numpy():
        d_neigh = np.linalg.norm(coord - df_spots.to_numpy(), axis=1)
        # check first 2 min distances (the min is  because its against the same spot, the closest spots
        # corresponds to the 2nd min dist)
        min_indexes = np.argpartition(d_neigh, 2)
        closest_neigh_dist = d_neigh[min_indexes[:2][1]]  # get the second min distance
        closest_neigh_idx = np.where(d_neigh == closest_neigh_dist)
        closest_neigh_spot = tuple(df_spots.to_numpy()[closest_neigh_idx][0])
        distances_neigh.append((closest_neigh_dist, closest_neigh_spot))
    return distances_neigh


def sort_by_distances(spots_df, contour_coords, cont_cutoff=10, neigh_cutoff=10):
    """
    Method to sort spots based on distance to contour and distance to neighbour
    Parameters
    ----------
    neigh_cutoff
    cont_cutoff
    spots_df
    contour_coords

    Returns
    -------

    """
    # Get distance to contour and closes neigh
    sub_df = spots_df.loc[:, ["x", "y"]]
    contour_distances = distance_to_contour(sub_df, contour_coords)  # Calculate min distance to contour
    neigh_distances = distance_to_neigh(sub_df)  # Calculate closest neighbour distance

    # Add distances and coords to dataframe
    sub_df.loc[:, "dist_cont"], cont_coord_list = list(zip(*contour_distances))
    sub_df.loc[:, "contour_x"], sub_df.loc[:, "contour_y"] = list(zip(*cont_coord_list))
    sub_df.loc[:, "dist_neigh"], neigh_coord_list = list(zip(*neigh_distances))
    sub_df.loc[:, "neigh_x"], sub_df.loc[:, "neigh_y"] = list(zip(*neigh_coord_list))

    ###########################################################
    # Spot selection based on distance to contour and distance
    ###########################################################
    # Label dataset with a "Selected" column
    sub_df.loc[:, 'selected'] = np.where((sub_df["dist_cont"] <= cont_cutoff) &
                                         (sub_df["dist_neigh"] > neigh_cutoff), "sel", "non-sel")
    # Add selection reason on a description column
    fulfill_text = "- Fulfill dist_cont and dist_neigh <br>"
    non_fulfill_text = "- Do not fulfill dist_cont and/or dist_neigh <br>"
    sub_df.loc[:, 'reason'] = np.where((sub_df["selected"] == "sel"), fulfill_text, non_fulfill_text)
    selection_df = spots_df.loc[(sub_df['selected'] == "sel")]
    # write to log percentage of selection
    num_selected = len(selection_df)
    percent_sel = num_selected * 100 / len(sub_df)

    return selection_df, sub_df, percent_sel


def save_html_figure(path_to_save, spots_df, img_num, img_contour_lab, ch_name="W1"):
    """
    Display selected and non-selected spots in an interactive image
    and save image in html format
    Parameters
    ----------
    ch_name: channel name: "W1" or "W2"
    spots_df: dataframe with spots coordinates for a given image
    img_num: image number ("01", "02", ...)
    img_contour_lab: binary image (ndarray) with contour as 1 and bgn as 0
    path_to_save: path to save image
    """
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    if not os.path.exists(path_to_save + "pp_segmented/"):
        os.mkdir(path_to_save + "pp_segmented/")

    selected = spots_df[spots_df["selected"] == "sel"]
    non_selected = spots_df[spots_df["selected"] == "non-sel"]
    percent_sel = round(len(selected) * 100 / (len(selected) + len(non_selected)), 3)
    # Create figure with lines to closest contour and closest neighbour
    foo_note = "<br>Number of Selected spots: {} / {} (Percentage = {} %)<br><br>".format(len(selected),
                                                                                          len(selected) + len(
                                                                                              non_selected),
                                                                                          percent_sel)
    img_contour = (img_contour_lab > 0).astype("uint8")
    fig_label_cont = px.imshow(img_contour, color_continuous_scale='gray',
                               title="<b>Image {} {}</b><br>{}".format(img_num, ch_name, foo_note))
    fig_label_cont.update_layout(coloraxis_showscale=False)  # to hide color bar
    fig_label_cont.update_layout(coloraxis_showscale=False)

    # Plot spots with custom hover information
    fig_label_cont.add_scatter(x=selected["y"], y=selected["x"],
                               mode="markers",
                               marker=dict(color="green", size=7),
                               name="selected",
                               customdata=np.stack(([selected["dist_cont"],
                                                     selected["dist_neigh"]]), axis=1),
                               hovertemplate=
                               '<b>x: %{x: }</b><br>'
                               '<b>y: %{y: } <b><br>'
                               '<b>dist_cont: %{customdata[0]: }<b><br>'
                               '<b>dist_neigh: %{customdata[1]: }<b><br>')

    fig_label_cont.add_scatter(x=non_selected["y"], y=non_selected["x"],
                               mode="markers",
                               marker=dict(color="red"),
                               name="non-selected",
                               customdata=np.stack(([non_selected["dist_cont"],
                                                     non_selected["dist_neigh"]]), axis=1),
                               hovertemplate=
                               '<b>x: %{x: }</b><br>'
                               '<b>y: %{y: } <b><br>'
                               '<b>dist_cont: %{customdata[0]: }<b><br>'
                               '<b>dist_neigh: %{customdata[1]: }<b><br>')

    # Plot nearest contour spot
    fig_label_cont.add_scatter(x=spots_df["contour_y"], y=spots_df["contour_x"],
                               mode="markers", name="contour")

    # Plot blue lines from spots to its corresponding contour
    for index, row in spots_df.iterrows():
        fig_label_cont.add_trace(
            go.Scatter(
                x=[row["y"], row["contour_y"]],
                y=[row["x"], row["contour_x"]],
                mode="lines",
                line=go.scatter.Line(color="blue"),
                name="{}".format(row["dist_cont"]), showlegend=False))

    # Plot orange lines from spots to its corresponding closest neighbour
    for index, row in spots_df.iterrows():
        fig_label_cont.add_trace(
            go.Scatter(
                x=[row["y"], row["neigh_y"]],
                y=[row["x"], row["neigh_x"]],
                mode="lines",
                line=go.scatter.Line(color="orange"),
                name="{}".format(row["dist_neigh"]), showlegend=False))

    # fig_label_cont.show(config={'modeBarButtonsToAdd': ['drawline',
    #                                                     'drawopenpath',
    #                                                     'drawclosedpath',
    #                                                     'drawcircle',
    #                                                     'drawrect',
    #                                                     'eraseshape'
    #                                                     ]})
    # plotly.io.to_html(path_to_save + "image_{}_{}.html".format(img_num, ch_name))
    fig_label_cont.write_html(path_to_save + "pp_segmented/" + "image_{}_{}.html".format(img_num, ch_name))


def main_segmentation():
    """
    1) Main method to run segmentation preprocessing.
    """
    print("#############################\n"
          " Segmentation Pre-processing \n"
          "#############################\n")
    logging.info("\n\n###############################\n"
                 "Initializing Segmentation Analysis \n"
                 "###################################\n\n")
    ###############
    # SEGMENTATION
    ###############
    # Segment yeast cells if not segmented
    if not os.path.exists(opt.segment_dir):
        os.mkdir(opt.segment_dir)
        segment_yeast()  # saves contour images in output/masks/
        print("\n\nCell Segmentation Finished!\n\n")
    elif not len(glob.glob(opt.images_dir + "image_*")) == len(glob.glob(opt.segment_dir + "masks/image_*")):
        shutil.rmtree(opt.segment_dir)
        segment_yeast()  # saves contour images in output/masks/
        print("\n\nCell Segmentation Finished!\n\n")
    else:
        pass

    percent_sel_total_W1 = list()
    percent_sel_total_W2 = list()
    percent_sel_total = list()
    total_data = 0
    total_selected = 0
    if os.path.exists(opt.segment_dir + "masks/") and len(os.listdir(opt.segment_dir + "masks/")) != 0:
        for img_file in glob.glob(opt.segment_dir + "masks/image_*"):
            start = time.time()
            image_number = img_file.split("/")[-1].split("_")[-1].split(".")[0]
            print("Processing image {} ...\n".format(image_number))
            ###########################################################
            # Calculate distance to contour and closest neighbour distance
            ###########################################################
            # Read contour image and labeled image
            image_labels = imread(opt.segment_dir + "masks/image_{}.tif".format(image_number)).astype(np.uint8)
            image_contour = imread(opt.segment_dir + "masks/contour_image_{}.tif".format(image_number)) \
                .astype(np.uint8)

            # clean mother-bud cell barriers
            # Avoid doing this step if already done
            if not len(glob.glob(opt.segment_dir + "masks/image_*")) == len(glob.glob(opt.segment_dir +
                                                                                            "masks/contour_mod*")):
                print("\tCleaning mother-daugther cells...\n")
                clean_mother_daugther(image_number, image_contour, image_labels)  # generates an contour_mod image
                image_contour_mod = imread(opt.segment_dir + "masks/contour_mod_{}.tif".format(image_number)) \
                    .astype(np.uint8)
            else:
                image_contour_mod = imread(opt.segment_dir + "masks/contour_mod_{}.tif".format(image_number)) \
                    .astype(np.uint8)

            # Load spot coordinates for W1_warped and W2
            if opt.Picco:
                spots_df_W1 = read_csv_2(opt.spots_dir +
                                         "detected_spot_{}_W1_warped".format(image_number))  # Spot coordinates W1
                spots_df_W2 = read_csv_2(opt.spots_dir +
                                         "detected_spot_{}_W2".format(image_number))  # Spot coordinates W2
            elif opt.Altair:
                spots_df_W1 = pd.read_csv(opt.spots_dir + "csv/" +
                                         "detected_spot_{}_W1_warped.csv".format(image_number),
                                          sep="\t", index_col=False)  # Spot coordinates W1
                spots_df_W2 = pd.read_csv(opt.spots_dir + "csv/" +
                                         "detected_spot_{}_W2.csv".format(image_number),
                                          sep="\t", index_col=False)  # Spot coordinates W2
            total_data += spots_df_W1.shape[0]

            # Add ID to each data point (spot)
            spots_df_W1.loc[:, "ID"] = list(range(1, spots_df_W1.shape[0] + 1))
            spots_df_W2.loc[:, "ID"] = list(range(1, spots_df_W2.shape[0] + 1))

            ###############################################
            # Sort by closest distance to contour and neigh
            ###############################################
            print("\tSorting spots...\n")
            cell_contour = np.where(image_contour_mod > 0)  # Group coordinates per cell (according to labels)
            cell_contour_coords = np.array(list(zip(cell_contour[0], cell_contour[1])))
            selection_df_W1, sub_df_W1, percent_sel_W1 = sort_by_distances(spots_df_W1,
                                                                           cell_contour_coords,
                                                                           cont_cutoff=opt.cont_cutoff,
                                                                           neigh_cutoff=opt.neigh_cutoff)
            selection_df_W2, sub_df_W2, percent_sel_W2 = sort_by_distances(spots_df_W2,
                                                                           cell_contour_coords,
                                                                           cont_cutoff=opt.cont_cutoff,
                                                                           neigh_cutoff=opt.neigh_cutoff)
            # Pair selected in W1 & W2
            selection_df_paired_W1 = selection_df_W1.loc[(selection_df_W1["ID"].isin(selection_df_W2["ID"]))]
            selection_df_paired_W2 = selection_df_W2.loc[(selection_df_W2["ID"].isin(selection_df_W1["ID"]))]
            # Assert shape W1 == shape W2
            assert set(selection_df_paired_W1.ID) == set(selection_df_paired_W2.ID)
            # update selected & non-selected values after pairing
            sub_df_W1["selected"] = np.where(~sub_df_W1["x"].isin(selection_df_paired_W1["x"]), "non-sel",
                                             sub_df_W1["selected"])
            sub_df_W2["selected"] = np.where(~sub_df_W2["x"].isin(selection_df_paired_W2["x"]), "non-sel",
                                             sub_df_W2["selected"])

            # write to log percentage of selection
            num_selected = selection_df_paired_W1.shape[0]
            percent_sel = num_selected * 100 / spots_df_W1.shape[0]
            logging.info("\nImage {} --> {:02} / {:02} "
                         "spots selected.. --> {} %".format(image_number, num_selected, len(spots_df_W1), percent_sel))
            total_selected += num_selected

            # Save df as csv: segmentation.csv
            if not os.path.exists(opt.results_dir):
                os.mkdir(opt.results_dir)
            if not os.path.exists(opt.results_dir + "segmentation/"):
                os.mkdir(opt.results_dir + "segmentation/")
            selection_df_paired_W1.to_csv(opt.results_dir + "segmentation/" +
                                          "detected_seg_{}_{}.csv".format(image_number, "W1"),
                                          sep=",", encoding="utf-8", header=True, index=False)
            selection_df_paired_W2.to_csv(opt.results_dir + "segmentation/" +
                                          "detected_seg_{}_{}.csv".format(image_number, "W2"),
                                          sep=",", encoding="utf-8", header=True, index=False)

            # Create figure with lines to the closest contour and closest neighbour
            save_html_figure(opt.figures_dir, sub_df_W1, image_number, image_contour_mod, ch_name="W1")
            save_html_figure(opt.figures_dir, sub_df_W2, image_number, image_contour_mod, ch_name="W2")

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
    print("\n\nTotal Percent W1 --> {} %\n"
          "Total Percent W2 --> {} %\n\n"
          "Total Paired Percent --> {} % \n".format(sum(percent_sel_total_W1) / len(percent_sel_total_W1),
                                                    sum(percent_sel_total_W2) / len(percent_sel_total_W2),
                                                    sum(percent_sel_total) / len(percent_sel_total)))
    return total_data, total_selected


if __name__ == "__main__":
    print("Yeast Segmentation Functions :)\n")
    sys.exit(0)
