# PICT-MODELLER


<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->
- [What is it?](#what-is-it)
- [How does it work?](#how-does-it-work)
- [Instructions](#instructions)
  - [Use Colab](#use-colab)
  - [Run it locally](#run-it-locally)
- [Tutorial](#tutorial)
- [Notes](#notes)

<!-- /TOC -->

What is it?
-----------

**PICT-MODELLER** is a Python-based software that provide the tools to model the near-physiological 3D architecture of 
protein assemblies. This software is the Python implementation of our previous work 
[Picco A., et al, 2017](https://www.sciencedirect.com/science/article/pii/S0092867417300521) where we combined 
[PICT](https://www.sciencedirect.com/science/article/pii/S0092867417300521) (yeast engineering & live-cell imaging)
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

You can use [PICT Colab](https://colab.research.google.com/drive/1kSOnZdwRb4xuznyQIpRNWUBBFKms91M8?usp=sharing)
to run the image analysis and modeling scripts without the need of installation. 

### Run it locally

You will need a machine running Linux, it does not support other operating systems.

1) Download the git repo in your local computer and get into the folder, or run:

```bash
  $ git clone https://github.com/Altairch95/PICT-MODELLER
  $ cd PICT-MODELLER
 ```
2) Download weights

PICT-MODELLER utilizes the pre-trained weights for the neural network that is used for yeast cell segmentation in 
[Yeast Spotter](http://yeastspotter.csb.utoronto.ca/). The weights are necessary to run the software, but are too large
 to share on GitHub. You can download the zip file from this [Zenodo](https://zenodo.org/record/3598690) repository. 

Once downloaded, simply unzip it and move it to the *scripts/* directory. You can also run the following command:

```bash
  $ unzip weights.zip
  $ mv weights/ path/to/PICT-MODELLER/scripts/
 ```

3) Create a conda environment:

```bash
  $ conda create -n {ENV_NAME} python=3.7 anaconda
  $ conda activate {ENV_NAME}
 ```

4) Install the requirements listed in *requirements.txt*:
5) 
```bash
  $ pip install -r requirements.txt
 ```

At this pont, the directory *PICT-MODELLER* should contain the files and directories described bellow:

#### Package tree

The package has the following structure:

    PICT-MODELLER/
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
          weights/                    (for mrcnn yeast segmentation)  
              
      test/
          input/
              pict_images/
              beads/


Image Analysis Tutorial
-----------------------

Image analysis can be divided into two steps. First, we processed images to measure the centroid positions of the 
GFP and RFP tags. Then, we analyzed the distribution of these centroid positions to estimate the true separation between
the GFP and RFP fluorophores using Single-molecule High-REsolution Colocalization (SHREC)
([Churchman et al. 2005](https://www.pnas.org/doi/abs/10.1073/pnas.0409487102), 
[Churchman et al. 2006](https://www.sciencedirect.com/science/article/pii/S0006349506722457)).

#### Command line arguments

```bash
  $ python3 measure_pict_distances.py -h
```
        Computing the distance distribution between fluorophores tagging the protein
        complex (e.g, exocyst) with a precision up to 5 nm.
        
        optional arguments:
          -h, --help            show this help message and exit
          -d DATASET, --dataset DATASET
                                Name of the dataset where the input/ directory is
                                located
          --test                Runs the test dataset
          -o OPTION, --option OPTION
                                Option to process: 'all' (whole workflow), 'beads'
                                (bead registration), 'pp' (preprocessing), 'spt' (spot
                                detection and linking), 'warping' (transform XY spot
                                coordinates using the beads warping matrix), 'segment'
                                (yeast segmentation), 'gaussian' (gaussian fitting),
                                'kde' (2D Kernel Density Estimate), 'outlier (outlier
                                rejection using the MLE)'. Default: 'main'


1) Run a test to check that everything is installed and running as expected:

```bash
  $ conda activate {ENV_NAME}  # Make sure to activate your conda environment
  $ python3 measure_pict_distances.py --test 
 ```

The test is composed by 5 raw images located in the *input/pict_images/* folder. By running the test, 
you should visualize on the terminal all the log of the image processing and image analysis steps. You 
can also get track about the program steps in the *log.txt* file.

You can check that the results are generated and saved in the *output/* folder with different sub-folders:
<ul>
    <li>images: contains the processed images.</li>
    <li>spots: contains the data from spot detection on your PICT images.</li>
    <li>segmentation: contains the segmented images, masks, and contour images.</li>
    <li>results: contains the resulting files from each processing/analysis step</li>
    <li>figures: contains HTML and png files to get track of the detected and selected spots for each 
    image, on each step, as well as the distance distribution for each step. It also contains PDF file with
    the final distance distribution and params estimates (mu and sigma) </li>
</ul>

2) Create a directory with the name of your system and the same structure as the *test/*: add to it the containing-beads
   directory (*beads/*) and the containing-PICT-images directory (*pict_images/*).

```bash
  $ mkdir my_dir_name
  $ cd my_dir_name
  # Create beads/ and pict_images/ if not there
  $ mkdir beads/
  $ mkdir pict_images/
  # Move the beads W1.tif and W2.tif to beads/ and your PICT images to pict_images/
  $ mv path/to/beads/*.tif path/to/my_dir_name/beads/
  $ mv path/to/pict-images/*.tif path/to/my_dir_name/pict_images/
 ```

Run the software:

```bash
  $ python3 measure_pict_distances.py -d my_dir 
 ```

You may grab a coffee while waiting for the results :)

Tutorial
--------


Notes
--------

#### Note 1: Input files (Beads and PICT images)

This program needs an input of bright-field TIFF images (central quadrant, 16-bit) captured as stacks of two channels: 

  - Channel 1: Red channel    (W1) --> observing RFP spots.
  - Channel 2: Green channel  (W2) --> observing GFP spots. 

**Beads**: To calibrate this protocol, the imaging of [TetraSpeck](https://www.thermofisher.com/order/catalog/product/T7279) 
in both the red and green channel is required. For each channel, the user should acquire images from  
4 fields of view (FOV) with isolated beads (avoiding clusters) and minimizing void areas (each FOV should have 
a homogeneity distribution of beads to cover all the possible coordinates. Finally, the 4 FOV for each channel 
should be stacked (e.g, stack-1 contains frame_FOV_0, frame_FOV_1, frame_FOV_2, frame_FOV_3) and named as **W1** for the
red channel, and **W2** for the green channel.

**PICT images**: *PICT images* is the name used to reference the images gathered from the 
[PICT experiment]((https://www.sciencedirect.com/science/article/pii/S0092867417300521)). Each *pict_image.tif* is a 
stack of red/green channels. Diffraction-limited spots should be visualized when opening the image with ImageJ or any 
other image processing software. 

#### Note 2: Running the software

From the input images, the program runs through different steps: 

##### 1) **Beads Registration**:
- Bead registration: isolated beads are detected for each channel. Agglomerations of beads, or beads shinning with
   low intensity are excluded based on the 0-th moment $$M_{00}$$ of brightness (excluding the beads with an /(M_{00}/)
   falling on the 1st and 95th percentile).
- Bead transformation: selected beads are transformed (aligned) and the transformation matrix is saved. For the 
   alignment, beads selected in W1 (red, mov) are aligned to the beads selected in W2 (green, reference).
        > Explanation: because we are imaging the same beads on the red and green channel, the XY coordinates should not
        change. However, because we are imaging at different wavelengths, each channel will refract the light differently
      (the refractive index of the lens varies with wavelength). The inability of the lens to bring the green and red 
       spots into a common focus results in a slightly different image size and focal point for each wavelength. This 
       artifact is commonly known as chromatic aberration, and must be corrected.

##### 2) **Image pre-procesing**:
- Background subtraction: Raw PICT images are corrected for the extracellular noise using the Rolling-Ball 
   algorithm. The size for estimating the rolling ball kernel is based on the maximum separation between two yeast 
   cells (a radius around 70 px.)
- Median filter: correction for the intracellular noise is also applied with a median filter of 10 px.
   
##### 3) **Spot Detection**: 
    - 
4) **Spot selection** 
5) **Outlier rejection**


* Input images are first preprocessed with a *background subtraction* and *median filter* algorithm to reduce the extracellular and citoplasmic noise of the images. 
* Chromatic aberration correction using synthetic beads. Beads in W1 (red) are aligned to beads of W2 (green, reference).
* Spot Detection and linking using trackpy, to detect and link bright spots of radius ~ 5nm in both channels.
* Spot selection:
    - Select spots in W1 and W2 based on distance to the contour of the cell (max distance: 8 - 10 px).
    - Select spots in W1 and W2 based on distance to the closest neighbour spot (min distance: 9 px).
    - Select spots in W1 and W2 based on the goodness of the gaussian fit, after fiting spot intensitites to a gaussian distribution.
    - Select spots in W1 and W2 based on a density probability estimation (KDE), assuming that here we discard all spots that are not "in focus".
    - Outlier rejection: fitting the final distribution of distances to a non-gaussian distribution described in [Churchman et al.,2006](https://duckduckgo.com) .

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


### Tu



