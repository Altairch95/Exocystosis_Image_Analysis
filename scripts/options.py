"""
============
OPTIONS FILE
============
In this file you will find all the options to run the program PICT-MODELLER,
without the need of using a parser from the terminal. Most options are
already set with a "default value" for "Spot Detection" and "Spot Selection".

Only the name of your dataset (e.g, test) and the option to run (e.g, all) must
be specified by the user.

dataset: Name of the dataset where the input/ directory is located
option: "Option to process:"
         " 'all' (whole workflow),"
         " 'beads' (bead registration),"
         " 'pp' (preprocessing),"
         " 'spt' (spot detection and linking),"
         " 'warping' (transform XY spot coordinates using the beads warping matrix),"
         " 'segment' (yeast segmentation),"
         " 'gaussian' (gaussian fitting),"
         " 'kde' (2D Kernel Density Estimate),"
         " 'outlier (outlier rejection using the MLE)'."
         " Default: 'main'"
"""
import sys
import time
from argparse import ArgumentParser
import matplotlib

matplotlib.use('Agg')

#########
# PARSER
#########

parser = ArgumentParser(
    prog='PICT-MODELLER',
    description='Computing the distance distribution between '
                'fluorophores tagging the protein complex (e.g, exocyst) '
                'with a precision up to 5 nm.')
parser.add_argument("-d", "--dataset",
                    dest="dataset",
                    action="store",
                    type=str,
                    help="Name of the dataset where the input/ directory is located",
                    required=False)

parser.add_argument("--test",
                    dest="test",
                    action="store_true",
                    default=False,
                    help="Runs the test dataset")

parser.add_argument('-o',
                    '--option',
                    dest='option',
                    action="store",
                    default="all",
                    help="Option to process:"
                         " 'all' (whole workflow),"
                         " 'beads' (bead registration),"
                         " 'pp' (preprocessing),"
                         " 'spt' (spot detection and linking),"
                         " 'warping' (transform XY spot coordinates using the beads warping matrix),"
                         " 'segment' (yeast segmentation),"
                         " 'gaussian' (gaussian fitting),"
                         " 'kde' (2D Kernel Density Estimate),"
                         " 'outlier (outlier rejection using the MLE)'."
                         " Default: 'main'")

parsero = parser.parse_args()

# ===================
# INPUTS AND OUTPUTS
# ====================
if parsero.test:
    dataset = "test"
    sel_option = "all"
elif parsero.dataset is None:
    sys.stderr.write("\n\n\tPICT-MODELLER-WARNING: Please, specify the name of your directory. Thanks!\n\n")
    time.sleep(2)
    parser.print_help()
    sys.exit(1)
else:
    dataset = parsero.dataset
    sel_option = parsero.option

# Working Directory
working_dir = f"../{dataset}/"
input_dir = working_dir + "input/"
output_dir = working_dir + "output/"

# Spot Detection Paths
# Input directory of images to be segmented
pict_images_dir = input_dir + "pict_images/"  # *** paste your PICT images here
beads_dir = input_dir + "beads/"  # *** paste your bead stacks here
images_dir = output_dir + "images/"
warped_dir = output_dir + "warped/"
spots_dir = output_dir + "spots/"
# Output directory to save masks to
segment_dir = output_dir + "segmentations/"
figures_dir = output_dir + "figures/"
results_dir = output_dir + "results/"

# ======================

###########
# OPTIONS
# #########
# PREPROCESSING: Background Subtraction, Medial Filter, Warping
bead_registration = False
preprocessing = False
# DETECTION & LINKING
detect_spots = False
mass_selection = False
# WARPING
warping = False
# SELECTION: Segmentation, Gaussian Fitting, KDE, Outlier Rejection
segmentation_preprocess = False
gaussian_fit = False
kde = False
outlier_rejection = False

if parsero.option == 'all':
    bead_registration = True
    preprocessing = True
    detect_spots = True
    warping = True
    segmentation_preprocess = True
    gaussian_fit = True
    kde = True
    outlier_rejection = True
if sel_option == 'beads' or 'beads' in sel_option.split(','):
    bead_registration = True
if sel_option == 'pp' or 'pp' in sel_option.split(','):
    preprocessing = True
if sel_option == 'spt' or 'spt' in sel_option.split(','):
    detect_spots = True
if sel_option == 'warping' or 'warping' in sel_option.split(','):
    warping = True
if sel_option == 'segment' or 'segment' in sel_option.split(','):
    segmentation_preprocess = True
if sel_option == 'gaussian' or 'gaussian' in sel_option.split(','):
    gaussian_fit = True
if sel_option == 'kde' or 'kde' in sel_option.split(','):
    kde = True
if sel_option == 'outlier' or 'outlier' in sel_option.split(','):
    outlier_rejection = True

# ======================

###########
# PARAMETERS
# #########
# IMAGE PRE-PROCESSING: Background Subtraction, Medial Filter, Warping
rolling_ball_radius = 70
median_filter_radius = 10
# SPOT DETECTION AND LINKING
particle_radius = 11  # must be an odd number.
percentile = 99.7  # float. Percentile (%) that determines which bright pixels are accepted as spots.
max_displacement = 2  # LINK PARTICLES INTO TRAJECTORIES
min_mass_cutoff = 0.01
max_mass_cutoff = 0.95

# SEGMENTATION:
# Cutoffs to select spots based on distance to contour and closest neighbour
cont_cutoff = 13
neigh_cutoff = 9

# OUTLIER REJECTION
# Deprecated, know using the median and std from the raw distribution of distances
# to start the optimization
mu_ini = None
sigma_ini = None

# Set to true to rescale the input images to reduce segmentation time
rescale = False
scale_factor = 2  # Factor to downsize images by if rescale is True

# Set to true to save preprocessed images as input to neural network (useful for debugging)
save_preprocessed = True

# Set to true to save a compressed RLE version of the masks for sharing
save_compressed = False

# Set to true to save the full masks
save_masks = True

# Set to true to have the neural network print out its segmentation progress as it proceeds
verbose = True

# Set to true to output ImageJ-compatible masks
output_imagej = False

# Save contour images
save_contour = True

# Save contour modified images
save_contour_mod = True
