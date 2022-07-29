#!/usr/bin/python3.7
# coding=utf-8
"""
BioImage Analysis functions to calculate
distances from PICT experiments.
"""
from pystackreg import StackReg
from argparse import ArgumentParser

from custom import *
from spot_detection_functions import detect_spots, link_particles
from segmentation_pp import *
from gaussian_fit import *
from kde import *
from outlier_rejection import outlier_rejection

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")  # avoid cv2 and pyqt5 to be incompatible
logging.getLogger('matplotlib.font_manager').disabled = True

__author__ = "Altair C. Hernandez"
__copyright__ = 'Copyright 2022, The Exocystosis Modeling Project'
__credits__ = ["Oriol Gallego", "Radovan Dojcilovic", "Andrea Picco", "Damien P. Devos"]
__version__ = "2.0"
__maintainer__ = "Altair C. Hernandez"
__email__ = "altair.chinchilla@upf.edu"
__status__ = "Development"


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
                        dest="directory",
                        action="store",
                        type=str,
                        help="Input directory where the raw PICT images are located",
                        required=True)
    parser.add_argument('-r',
                        dest='radius',
                        action="store",
                        type=int,
                        default=70,
                        help="Rolling Ball Radius (pixels)")

    parser.add_argument('-m',
                        dest='median_filter',
                        action="store",
                        type=int,
                        default=10,
                        help="Median Radius (pixels)")

    parser.add_argument('-w',
                        '--option',
                        dest='option',
                        action="store",
                        default="main",
                        help="Option to process: 'main' (whole workflow), 'pp' (preprocessing), "
                             "'warping' (pp + warping), 'sl' (pp + warping + spot location&linking + "
                             "moment calculation), 'distances' (distance estimation from spot_location tables in "
                             "trackpy directory). Default: 'main'")

    parser.add_argument('-d',
                        '--dirty',
                        dest='dirty',
                        action="store_true",
                        default=False,
                        help="Generates an output file for each processing step to track what is happening")

    parser.add_argument('-v',
                        '--verbose',
                        dest='verbose',
                        action="store_true",
                        default=False,
                        help="Shows what the program is doing")

    return parser.parse_args()


def pp():
    """
    Pre-processing = Background Subtraction & Median Filter
    """
    if opt.verbose:
        print("#############################\n"
              "     Image Preprocessing \n"
              "#############################\n")
    # Do Bead Registration
    sr = register_beads(opt.beads_dir)
    for file in glob.glob(opt.pict_images_dir + "*.tif"):
        try:
            # BACKGROUND SUBTRACTION, MEDIAN FILTER, WARPING
            image = BioImage(file).subtract_background()
            imageMD = image.median_filter()
            image.do_warping(sr)

        except FileNotFoundError as fnf:
            sys.stderr.write("{}\nWas not possible to do bead registration. Exit.".format(fnf))


def register_beads(beads_dir=opt.beads_dir):
    """
    Using PyStackReg package for registration. This is part of the
    chromatic aberration correction part.

    W2 and W1 are both a stack of bead images (4 Fields of View (FOV))
    and will be used for the registration. In the registration, we have
    a reference BEAD image and a mov BEAD image. The algorithm tries to fit the mov
    against the ref in the optimal way, and it generates a matrix of rotation
    and translation (Transformation Matrix), that will be used to transform the
    sample images of cells (imageXX_MD.tiff).
    pyStackReg provides the following five types of distortion:

    - translation
    - rigid body (translation + rotation)
    - scaled rotation (translation + rotation + scaling)
    - affine (translation + rotation + scaling + shearing)
    - bilinear (non-linear transformation; does not preserve straight lines)

    # BEADS FOR DOING THE REGISTRATION (Alignment of one or more images to a common reference image)
    # WARNING! BEAD IMAGES MUST BE FROM SAME DAY AND MICROSCOPE AS THE SAMPLE IMAGES!

    Parameters
    ----------
    beads_dir (string): path to W1 and W2 bead files

    Returns
    -------
    StackReg object with registration ready for transformation on images.
    """
    try:
        w1_path = beads_dir + "W1.tif"
        w2_path = beads_dir + "W2.tif"
        print("Registration on beads {}/{} using PyStackReg...\n".format(w1_path, w2_path))
        # We will register all the frames of W1 (mov, to warp) against all the frames
        # of W2 (reference)
        w2 = io.imread(w2_path)  # 3 dimensions : frames x width x height - uint16 bit image
        w1 = io.imread(w1_path)  # same dim as w2 - uint16 bit image

        # both stacks must have the same shape
        assert w2.shape == w1.shape

        # creating "sr" object, that will contain everything.
        # We can apply different types of registration: TRANSLATION, RIGID_BODY (translation + rotation),
        # SCALED_ROTATION (translation + rotation + scaling), AFFINE (translation + rotation + scaling + shearing),
        # BILINIEAR (non-linear transformation; does not preserve straight lines)
        sr = StackReg(StackReg.SCALED_ROTATION)

        # Register all frames of red channel (W1) to all frames of green channel (W2)
        for frame in range(w2.shape[2]):
            sr.register(w2[:, :, frame], w1[:, :, frame])
        print("\tBeads registered!\n")
        return sr
    except IndexError as idx_error:
        sys.stderr.write("{}".format(idx_error))


def spot_detection():
    """
    Method for spot detection from already Background Subtracted images.

    Returns
    -------
    DataFrame with detected spots.

        DataFrame([x, y, mass, size, ecc, signal]);

        where mass means total integrated brightness of the blob, size means the radius of gyration
        of its Gaussian-like profile, and ecc is its eccentricity (0 is circular).
    """
    if os.path.exists(opt.images_dir) and len(glob.glob(opt.images_dir)) != 0:
        if os.path.exists(opt.warped_dir) and len(glob.glob(opt.warped_dir)) != 0:
            # Create path to Trackpy spots in the case it is not created
            if not os.path.isdir(opt.spots_dir):
                os.mkdir(opt.spots_dir)
            ########################################
            # ---Run SPOT LOCATION AND LINKING---
            ########################################
            if opt.verbose:
                print("##########################################\n"
                      " ---Running SPOT LOCATION AND TRACKING--- \n"
                      "##########################################\n")
            for img in glob.glob(opt.warped_dir + "stacks/*imageMD*.tif"):
                img = img.split("/")[-1]
                if opt.verbose:
                    print("# IMAGE {} \n#".format(img))
                # SPOT DETECTION
                f_batch = detect_spots(img, opt.warped_dir + "stacks/", opt.spots_dir, opt.particle_radius,
                                       opt.percentile)
                if f_batch is not None:
                    # LINKING
                    print("\nSpot Detection and Linking done\n\tAligning W1 and W2 files..\n")
                    link_particles(f_batch, img, opt.spots_dir, opt.max_displacement)

            # Scatter detected spots directly in image
            for spot_file in glob.glob(opt.spots_dir + "csv/detected*.csv"):
                img_num, ch_name = spot_file.split("/")[-1].split("_")[2], \
                                   spot_file.split("/")[-1].split("_")[3].split(".")[0]
                if ch_name == "W1":
                    img = imread(opt.warped_dir + "imageMD_{}_W1_warped.tif".format(img_num), plugin="tifffile")
                    df = pd.read_csv(spot_file, sep="\t")
                    save_html_figure(opt.figures_dir, df, img_num, img, "W1")
                elif ch_name == "W2":
                    img = imread(opt.warped_dir + "imageMD_{}_W2.tif".format(img_num), plugin="tifffile")
                    df = pd.read_csv(spot_file, sep="\t")
                    save_html_figure(opt.figures_dir, df, img_num, img, "W2")


def save_html_figure(path_to_save, spots_df, img_num, ndimage, ch_name="W1"):
    """
    Display selected and non-selected spots in an interactive image
    and save image in html format
    Parameters
    ----------
    ch_name: channel name: "W1" or "W2"
    spots_df: dataframe with spots coordinates for a given image
    img_num: image number ("01", "02", ...)
    ndimage: image (ndarray)
    path_to_save: path to save image
    """
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    if not os.path.exists(path_to_save + "spot_detection/"):
        os.mkdir(path_to_save + "spot_detection/")

    fig_label_cont = px.imshow(ndimage, color_continuous_scale='gray',
                               title="<b>Image {} {}</b><br>".format(img_num, ch_name))
    fig_label_cont.update_layout(coloraxis_showscale=False)  # to hide color bar

    # Plot spots with custom hover information
    fig_label_cont.add_scatter(x=spots_df["y"], y=spots_df["x"],
                               mode="markers",
                               marker=dict(color="green", size=7),
                               name="selected",
                               customdata=np.stack(([spots_df["size"],
                                                     spots_df["ecc"]]), axis=1),
                               hovertemplate=
                               '<b>x: %{x: }</b><br>'
                               '<b>y: %{y: } <b><br>'
                               '<b>size: %{customdata[0]: }<b><br>'
                               '<b>ecc: %{customdata[1]: }<b><br>')

    fig_label_cont.write_html(path_to_save + "spot_detection/" + "image_{}_{}.html".format(img_num, ch_name))


def main():
    """Main function to do everything"""
    if opt.preprocessing:
        ######################
        # IMAGE PREPROCESSING
        ######################
        pp()

    if opt.detect_spots:
        ###########################
        # SPOT DETECTION & LINKING
        ###########################
        spot_detection()

    if opt.segmentation_preprocess:
        ##############################
        # SEGMENTATION PRE-PROCESSING
        ##############################
        total_data, seg_selected = main_segmentation()

    ##############################
    # GAUSSIAN FITTING & SELECTION
    ##############################
    if opt.gaussian_fit:
        gauss_initial, gauss_selected = main_gaussian()

    ##############################
    # KERNEL DENSITY ESTIMATION
    ##############################
    if opt.kde:
        kde_initial, kde_selected = main_kde()

    #############################
    # OUTLIER REJECTION
    #############################
    if opt.outlier_rejection:
        outlier_rejection()
