"""
============
OPTIONS FILE
============
In this file you will find all the options to run the program PICT-MOD,
without the need of using a parser from the terminal. Most options are
already set with a "default value" for "Spot Detection" and "Spot Selection".

Please change:
- input directory path name
- output directory path name
"""
####################
# INPUTS AND OUTPUTS
# ##################
# import sys

dataset = ""                                 # *** replace with your working dir name
test = True
if test:
    dataset += "sla2/sla2_C"  # sys.argv[1]
# ======================
# Working Directory
working_dir = "../{}/".format(dataset)
input_dir = working_dir + "input/"
output_dir = working_dir + "output/"

# Spot Detection Paths
# Input directory of images to be segmented
pict_images_dir = input_dir + "pict_images/"  # *** paste your PICT images here
beads_dir = input_dir + "beads/"              # *** paste your bead stacks here
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
# Picco or Altair? Or both?
Picco = False
Altair = True

# Sla2 test: only True if running with Sla2 images
sl2 = True  # warping is done with green - red, not red - green 
# Preprocessing: Background Subtraction, Medial Filter, Warping
preprocessing = True
# Spot Detection and Linking
detect_spots = True
# Spot Selection: Segmentation, Gaussian Fitting, KDE, Outlier Rejection
segmentation_preprocess = True
gaussian_fit = True
kde = True
outlier_rejection = True
# ======================

###########
# PARAMETERS
# #########
# IMAGE PRE-PROCESSING: Background Subtraction, Medial Filter, Warping
rolling_ball_radius = 70
medial_filter_radius = 10
# SPOT DETECTION AND LINKING
particle_radius = 11   # must be an odd number.
percentile = 99.6      # float. Percentile (%) that determines which bright pixels are accepted as Particles.
max_displacement = 3   # LINK PARTICLES INTO TRAJECTORIES

# SEGMENTATION:
# Cutoffs to select spots based on distance to contour and closest neighbour
cont_cutoff = 10
neigh_cutoff = 9

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
