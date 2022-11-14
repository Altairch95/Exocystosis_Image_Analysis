#!/usr/bin/python3.7
# coding=utf-8
"""
           26/09/22         ****************
                            ** TO MODIFY **
                            ****************

#####################################################################
################# PICT-MOD MAIN PROGRAM ############################
####################################################################

BioImage Analysis script to calculate distances from PICT experiments.

[Elaborate program]

0) Initialize program
- Make sure that input directories are created
- Make sure pict_images/ is not empty and files end with .tif
    - print in report: number of pict_images, number of channels per image,
    bit size (16-bit)
- Make sure beads/ is not empty and contain two files: W1.tif & W2.tif
- Create output/ if not exist
- Initialize log file

1) Image Preprocessing: Background Subtraction, Median Filter,
Chromatic aberration correction (Warping)
- Read images from pict_images/ and apply background subtractions and
subtract median filter from background.
    - Parameters in options.py (# IMAGE PRE-PROCESSING)
    - Save BGN Subtracted image as image_XX.tif in output/images/
    - Save BGN and MD image as imageMD_XX.tif in output/images/
- Correct for chromatic aberration (PyStackReg)
    - using green channel (beads/W2.tif) as reference to align beads
    - apply registration and transformation
    - save warped images and stacks in output/warped/
        - possibility to warp the whole image or only bead positions

2) Spot Detection & Linking (Trackpy)
- Input images in output/warped/stacks/
- Parameters in options.py (# SPOT DETECTION AND LINKING)
    - Save detected_spot_XX_(W1/W2).csv in output/spots/

3) Spot Selection: cell segmentation, gaussian fitting, KDE,
outlier rejection
- Cell Segmentation using YeastSpotter and sort according to
the nearest distance to contour and the distance to the nearest neighbour
- Fit spots to a gaussian function and sort according to R^2
- Do KDE and get spots from the most dense area (Pr > 0.5)
- Outlier Rejection (bootstrapping)
- output.csv --> mu = XX +- SErrxx ; sigma = YY +- SErryy ; n = Z

"""
from calculate_PICT_distances import *
from detect_beads import beads_registration, create_beads_stacks
import matplotlib as mpl

mpl.rcParams.update({'figure.max_open_warning': 0})

__author__ = "Altair C. Hernandez"
__copyright__ = 'Copyright 2022, The Exocystosis Modeling Project'
__credits__ = ["Oriol Gallego", "Radovan Dojcilovic", "Andrea Picco", "Damien P. Devos"]
__version__ = "2.0"
__maintainer__ = "Altair C. Hernandez"
__email__ = "altair.chinchilla@upf.edu"
__status__ = "Development"

# Set up logging files
print(opt.working_dir)
if not os.path.exists(opt.working_dir + "log.txt"):
	logging.basicConfig(filename=opt.working_dir + "log.txt", level=logging.DEBUG,
						format="%(asctime)s %(message)s", filemode="w")
else:
	logging.basicConfig(filename=opt.working_dir + "log.txt", level=logging.DEBUG,
						format="%(asctime)s %(message)s", filemode="a")

logging.info("\n\n############################\n"
			 "Image Analysis for LiveCellPICT \n"
			 "################################\n\n"
			 "\tDataset: {}\n"
			 "\tWorking directory: {}\n"
			 "\tInput directory: {}\n"
			 "\tOutput directory: {}\n\n".format(opt.dataset, opt.working_dir, opt.input_dir, opt.output_dir))

if not os.path.exists(opt.output_dir):
	os.mkdir(opt.output_dir)

beads_percentile = 98.0
beads_max_mass_cutoff = 0.90  # Reject the upper right of this cutoff (in tant per 1)
beads_min_mass_cutoff = 0.01  # Reject the upper left  of this cutoff (in tant per 1)
if opt.bead_registration:
	logging.info("\n\n#######################\n"
				 "Initializing BEAD REGISTRATION \n"
				 "############################\n\n")
	create_beads_stacks(opt.beads_dir)
	beads_registration(opt.beads_dir, beads_percentile, beads_min_mass_cutoff, beads_max_mass_cutoff,
					   verbose=opt.verbose)
	print("\tBEADS REGISTERED! \n")

if opt.preprocessing:
	logging.info("\n\n#######################\n"
				 "Initializing PREPROCESSING \n"
				 "############################\n\n")
	######################
	# IMAGE PREPROCESSING
	######################
	pp(opt.pict_images_dir, opt.images_dir, opt.rolling_ball_radius,
	   opt.median_filter_radius, verbose=opt.verbose)

if opt.detect_spots:
	logging.info("\n\n#######################\n"
				 "Initializing SPOT DETECTION \n"
				 "############################\n\n")
	###########################
	# SPOT DETECTION & LINKING
	###########################
	spot_detection(opt.images_dir, opt.spots_dir, opt.results_dir, opt.figures_dir,
				   opt.particle_radius, opt.percentile, opt.min_mass_cutoff, opt.max_mass_cutoff,
				   opt.max_displacement, verbose=opt.verbose, mass_selection=opt.mass_selection)

if opt.warping:
	logging.info("\n\n#######################\n"
				 "Initializing WARPING \n"
				 "############################\n\n")
	######################
	# WARPING
	######################
	# Check if transformation matrix is in beads directory
	if not os.path.exists(opt.beads_dir + "transform.npy"):
		create_beads_stacks(opt.beads_dir)
		beads_registration(opt.beads_dir, beads_percentile, beads_min_mass_cutoff, beads_max_mass_cutoff)
	warping(opt.beads_dir, opt.spots_dir, opt.figures_dir, opt.results_dir)
	plot_links(opt.images_dir + "imageMD*.tif", opt.spots_dir, opt.figures_dir + "spot_detection/")

if opt.segmentation_preprocess:
	logging.info("\n\n#######################\n"
				 "Initializing SPOT SELECTION \n"
				 "#########################\n\n")
	##############################
	# SEGMENTATION PRE-PROCESSING
	##############################
	total_data, seg_selected = main_segmentation(opt.segment_dir, opt.images_dir, opt.spots_dir, opt.results_dir,
												 opt.figures_dir, opt.scale_factor, opt.cont_cutoff,
												 opt.neigh_cutoff, rescale=opt.rescale, verbose=opt.verbose)
	plot_links(opt.segment_dir + "masks/contour_mod*.tif", opt.spots_dir,
			   opt.figures_dir + "pp_segmented/")

if opt.gaussian_fit:
	##############################
	# GAUSSIAN FITTING & SELECTION
	##############################
	gauss_initial, gauss_selected = main_gaussian(opt.results_dir, opt.images_dir, opt.figures_dir)

if opt.kde:
	##############################
	# KERNEL DENSITY ESTIMATION
	##############################
	kde_initial, kde_selected = main_kde(opt.images_dir, opt.results_dir, opt.figures_dir)

if opt.outlier_rejection:
	####################
	# OUTLIER REJECTION
	####################
	outlier_rejection(opt.results_dir, opt.figures_dir, test=False, mu_ini=opt.mu_ini, sigma_ini=opt.sigma_ini)

print("\n\n#######################\n"
	  "Analysis Done!! \n"
	  "#########################\n\n")

sys.exit(0)
