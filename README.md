# PICT-MODELLER


<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->
- [What is it?](#what-is-it)
- [How does it work?](#how-does-it-work)
- [Instructions](#instructions)
  - [Use Colab](#use-colab)
  - [Run it locally](#run-it-locally)

<!-- /TOC -->

What is it?
-----------

**PICT-MODELLER** is a Python-based software that provide the tools to model the near-physiological 3D architecture of 
protein assemblies. This software is the Python implementation of our previous work
<a href=https://www.sciencedirect.com/science/article/pii/S0092867417300521> (Picco A., et al, 2017)</a> where we combined 
<a href=https://www.sciencedirect.com/science/article/pii/S0092867417300521> PICT<a> (yeast engineering & live-cell imaging)
and integrative modeling to reconstruct the molecular architecture of the exocyst complex in its cellular environment.

How does it work?
-----------

PICT-MODELLER utilizes bioimage analysis tools that allows for the **processing** (*Background subtraction*, *chromatic
aberration correction*, and *spot detection*) and **analysis** of live-cell imaging (fluorescence microscopy) data to 
estimate the pair-wise distance between a fluorophore flagging the terminus of a protein complex (prey-GFP) and a static
intracellular anchor site (anchor-RFP). From a dataset of 20 - 30 images, PICT-MODELLER estimates the μ and σ parameters 
of the final distance distribution with a precision below 5 nm.

PICT-MODELLER can use the set of pairwise distances to determine the relative fluorophore positions by trilateration.
The Integrative Modeling Platform <a href=https://integrativemodeling.org/>(IMP)</a> can be used to integrate the 
distribution of fluorophore positions (determined from *in situ* data) with available structural information (obtained
 *in vitro*) to recapitulate the architecture of the protein complex in its close-to-native environment.


Instructions
-----------
### Use Colab

You can use <a href=https://colab.research.google.com/drive/1kSOnZdwRb4xuznyQIpRNWUBBFKms91M8?usp=sharing>PICT Colab</a>
to run the image analysis and modeling scripts without the need of installation. 

### Run it locally
You will need a machine running Linux, 

1) Download the git repo in your local computer and get into the folder, or run:

```bash
  $ git clone https://github.com/Altairch95/Exocystosis_Image_Analysis
  $ cd Exocystosis_Image_Analysis
 ```

2Create a conda environment with python3.7.7:

```bash
  $ conda create -n {ENV_NAME} python=3.7.7 anaconda
 ```

3) Install the requiments listed in *requirements.txt*:
```bash
  $ pip install -r requirements.txt
 ```



At this pont, the directory *Exocystosis_Image_Analysis* should contain the files and directories described bellow:

#### Package tree

The package has the following structure:

    Exocystosis_Image_Analysis/
      README.md
      scripts/
          calculate_PICT_distances.py
          custom.py
          gaussian_fit.py
          kde.py
          lle.py
          rnd.py
          measure_pict_distances.py  (main script)
          options.py                 (User selections)
          outlier_rejections.py
          segmentation_pp.py
          spot_detection_functions.py
          mrcnn/                      (YeastSpotter)
          weights/                    (for Yeasr Spoter weights)  
              
      sla2/
          sla2_C/
              input/
                  pict_images/
                  beads/
         sla2_N_C/
              input/
                  pict_images/
                  beads/
          

* README.md: the tutorial and information about our application.
* scripts: a folder with the following scripts:
  - measure_pict_distances.py: the command-line script to launch the program.
  - custom.py: a module requiered by measure_pict_distances.py where are defined the classes of the program.
  - calculate_PICT_distances.py: a module required by measure_pict_distances.py where are defined the functions of the program.
  - options.py: file with parameters to modify by the user to run the program (see tutorial).
  
* sla2 (Test): a directory with two input datasets that serve as input to test the program.

-----

Tutorial
--------

### Input Files

This program needs an input of brightfield TIFF images (central quadrant, 16-bit) captured as stacks of two channels: 
  - Channel 1: Red channel    (W1) --> observing RFP spots.
  - Channel 2: Green channel  (W2) --> observing GFP spots.
  
From the input images, the program runs through different steps: **image preprocesing**, **Spot Detection** (Trackpy), **Spot selection** and **outlier rejection**.

* Input images are first preprocessed with a *background subtraction* and *median filter* algorithm to reduce the extracellular and citoplasmic noise of the images. 
* Chromatic aberration correction using synthetic beads. Beads in W1 (red) are aligned to beads of W2 (green, reference).
* Spot Detection and linking using trackpy, to detect and link bright spots of radius ~ 5nm in both channels.
* Spot selection:
    - Select spots in W1 and W2 based on distance to the contour of the cell (max distance: 8 - 10 px).
    - Select spots in W1 and W2 based on distance to the closest neighbour spot (min distance: 9 px).
    - Select spots in W1 and W2 based on the goodness of the gaussian fit, after fiting spot intensitites to a gaussian distribution.
    - Select spots in W1 and W2 based on a density probability estimation (KDE), assuming that here we discard all spots that are not "in focus".
    - Outlier rejection: fitting the final distribution of distances to a non-gaussian distribution described in [Churchman et al.,2006](https://duckduckgo.com) .


### Tutorial

In this section we make a brief explanation of how to use Exocystosis_Image_Analysis.

#### Command line arguments

Within your working directory, first you need to modify some parameters of the program. Those, 
are related to your working directory, and the processes to run (by default: all). 

Then, you can run the following command:

```bash
    $ measure_pict_distances.py {DATA_FOLDER}
```

By default, if not specified in the options.py file, the program will run a test with 
sla2 C-terminal within the sla2 folder.


### Output Example



