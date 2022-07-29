"""
Python functions for spot location using Trackpy modules
"""
import os.path
import trackpy as tp
import pims


def detect_spots(bgn_img, path_to_folder, path_to_output, radius, percentile):
    """
    Method for spot detection from already Background Subtracted images.
    Parameters
    ----------
    bgn_img: name of the background subtracted image
    path_to_folder
    path_to_output: where do you want to save the output?
    radius: float. Radius in pixels of the spots to search.
    percentile: float. Percentile (%) that determines which bright pixels are accepted as Particles.

    Returns
    -------
    DataFrame with detected spots.

        DataFrame([x, y, mass, size, ecc, signal]);

        where mass means total integrated brightness of the blob, size means the radius of gyration
        of its Gaussian-like profile, and ecc is its eccentricity (0 is circular).

    """
    # Load frames using PIMS
    frames = pims.open(path_to_folder + bgn_img)

    # Check if the image is a stack or not. If is a stack, it should contain "stack" in the name.
    # Stack is made out of W1_warped-W2
    stack = False
    img_num = ""  # image number (string)
    out_all_spots_file = "detected_spot_"  # number of final file
    if len(frames) > 1:
        print("\t\t{} frames in stack...\n".format(len(frames)))
        stack += True
        img_num += bgn_img.split(".")[0].split("_")[-1]

        print("# SPOT DETECTION with Trackpy...\n\n\t# READING DATA...\n\n"
              "\t\tParticle radius = {}\n"
              "\t\tPercentile = {}\n\n".format(radius, percentile))

        f_batch = tp.batch(frames[:], radius, percentile=percentile, preprocess=True, engine='python')
        f_batch = f_batch.rename(columns={"x": "y", "y": "x"})  # change x,y order
        frame_0, frame_1 = len(list(f_batch.loc[f_batch["frame"] == 0, "x"].tolist())), \
                           len(list(f_batch.loc[f_batch["frame"] == 1, "x"].tolist()))
        print("\t\t\tFrame 0: {} particles detected\n"
              "\t\t\tFrame 1: {} particles detected\n\n".format(frame_0, frame_1))

        # f_batch_custom = f_batch.loc[:, ["frame", "x", "y", "mass", "size", "ep", "m0", "m2", "ecc", "mu11", "mu20", "mu02"]]
        f_batch_custom = f_batch
        f_batch_custom['size'] = f_batch_custom['size'].apply(lambda x: x**2)  # remove sqrt from size formula 

        return f_batch_custom  # f_batch

    elif len(frames) == 1:
        print("\t\t{} frame in image...\n".format(len(frames)))
        img_num += bgn_img.split(".")[0].split("_")[1]
        img_type = bgn_img.split(".")[0].split("_")[-1]
        if img_type == "warped":
            img_type = "W1_warped"
        out_all_spots_file += "{}".format(img_num)

        print("# SPOT DETECTION with Tracpy...\n\n\t# READING DATA...\n\n"
              "\t\tParticle diameter = {}\n"
              "\t\tPercentile = {}\n\n".format(radius, percentile))

        f_batch = tp.locate(frames[0], radius, percentile=percentile, preprocess=True)  # noise=5, smoothing_size=10
        frame_0 = len(list(f_batch.loc[f_batch["frame"] == 0, "x"].tolist()))
        print("\t\t\tFrame 0: {} particles detected\n".format(frame_0))

        id_list = range(0, f_batch.shape[0])
        f_batch["id"] = id_list
        f_batch = f_batch.set_axis(range(1, f_batch.shape[0] + 1), axis="index")
        f_batch = f_batch.loc[:,
                  ["id", "frame", "x", "y", "mass", "size", "ecc", "signal", "raw_mass", "ep",
                   "m0", "m1", "m2", "m3", "m4", "m5", "mu11", "mu20", "mu02", "ecc_2", "ori"]]
        f_batch_custom = f_batch[["x", "y", "mass", "size", "ecc", "signal", "ep", "ori", "mu11",
                                  "mu20", "mu02"]]
        # f_batch_custom = f_batch[["y", "x", "m0", "m2", "ecc", "m1", "m3", "ori", "mu11",
        #                           "mu20", "mu02"]]
        f_batch_custom.to_csv(path_to_output + out_all_spots_file + "_{}.csv".format(img_type), sep=",",
                              encoding="utf-8", header=True, index=True)
        f_batch_custom.to_csv(path_to_output + out_all_spots_file + "_{}".format(img_type), sep=" ",
                              encoding="utf-8", header=False, index=False)
        print("\t\tSaving results at {} and finishing.\n\n".format(path_to_output + out_all_spots_file))
        return None


def link_particles(f_batch_df, img, path_to_output, maximum_displacement):
    """
    Recurse for linking already found particles into particle trajectories.
    Parameters
    ----------
    f_batch_df: pd.Dataframe. DataFrame([x, y, mass, size, ecc, signal])
    img: image name for saving file
    path_to_output: where to save the output.
    maximum_displacement: how much a particle may move between frames to be considered to link.

    Returns
    -------
    Nothing to return here.

    """
    img_num = img.split(".")[0].split("_")[-1]  # if is a stack it should be [3]
    out_traj_file = "detected_spot_{}".format(img_num)
    print("\t# LINKING INTO PARTICLE TRAJECTORIES...\n\n"
          "\t\tMaximum displacement: {}\n"
          "\t\tAdaptive step = {}\n\n".format(maximum_displacement, 1.0))
    t = tp.link(f_batch_df, maximum_displacement, pos_columns=["x", "y"], adaptive_step=1.0)
    t_sort_particles = t.sort_values(by=["particle", "frame"])
    t_only_paired = t_sort_particles[t_sort_particles.duplicated("particle", keep=False)]
    # index_order = [traj for traj in range(1, int(t_only_paired.shape[0] / 2) + 1) for _ in range(1, 3)]
    # t_filtered = t_only_paired.set_axis(range(1, t_only_paired.shape[0] + 1), axis="index")
    # t_filtered["num_traj"] = index_order

    # Split linked particles by frame
    t_filtered_W1_warped = t_only_paired[t_only_paired["frame"] == 0]
    t_filtered_W2 = t_only_paired[t_only_paired["frame"] == 1]

    # Choose desired columns to output
    # t_filtered_W1_warped = t_filtered_W1_warped[["y", "x", "mass", "size", "ecc", "ep", "mu11", "mu20", "mu02"]]
    # t_filtered_W1_warped = t_filtered_W1_warped.rename(columns={"mass": "m0", "size": "m2"})
    # t_filtered_W2 = t_filtered_W2[["y", "x", "mass", "size", "ecc", "ep", "mu11", "mu20", "mu02"]]
    # t_filtered_W2 = t_filtered_W2.rename(columns={"mass": "m0", "size": "m2"})
    t_filtered_W1_warped = t_filtered_W1_warped[["x", "y", "mass", "size", "ecc", "ep"]]
    t_filtered_W2 = t_filtered_W2[["x", "y", "mass", "size", "ecc", "ep"]]
    print("\t\tNumber of trajectories detected: {}".format(t_only_paired.shape[0] / 2))

    t_filtered_W1_warped.to_csv(path_to_output + out_traj_file + "_W1_warped", sep="\t", encoding="utf-8",
                                header=False, index=False)
    t_filtered_W2.to_csv(path_to_output + out_traj_file + "_W2", sep="\t", encoding="utf-8", header=False,
                         index=False)
    if not os.path.isdir(path_to_output + "csv/"):
        os.mkdir(path_to_output + "csv/")

    t_filtered_W1_warped.to_csv(path_to_output + "csv/" + out_traj_file + "_W1_warped.csv", sep="\t", encoding="utf-8",
                                header=True, index=False)
    t_filtered_W2.to_csv(path_to_output + "csv/" + out_traj_file + "_W2.csv", sep="\t", encoding="utf-8", header=True,
                         index=False)

    print("\t\tSaving trajectories at {}.\n\n".format(path_to_output + out_traj_file))


if __name__ == "__main__":
    print("Hi mate! This file only contains python functions for SPOT DETECTION using Trackpy."
          "You may want to use them, open it and have a look ;)")
