# PICT-MOD
----------
## Sub-pixel distance estimation between fluorohores using live-cell imaging.

What is it?
-----------

**PICT-MOD** is a Python package (will be) for estimating distances between fluorophores flagging the termini of protein complexes of interest with a 2 - 5 nm precision. 


### Download 

1) Clone the git repo in your local computer and get into the folder, or:

```bash
  $ git clone https://github.com/Altairch95/Exocystosis_Image_Analysis
  $ cd Exocystosis_Image_Analysis
 ```
2) Downloading the Weights
We need the pre-trained weights for the neural network of YeastSpotter, which are too large to share on GitHub. You can grab them from our webserver as a zip file (http://hershey.csb.utoronto.ca/weights.zip) or from Zenodo (https://zenodo.org/record/3598690).

3) Create a conda environment with python3.7.7:

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
          

* README.md: the files containing the tutorial and information about our application.
* scripts: a fold with the following scripts:
  - measure_pict_distances.py: the command-line script to launch the program.
  - custom.py: a module requiered by measure_pict_distances.py where are defined the classes of the program.
  - calculate_PICT_distances.py: a module required by measure_pict_distances.py where are defined the functions of the program.
  - options.py: file with parameters to modify by the user to run the program (see tutorial).
  
* sla2: a directory with two input datasets that serve as input to test the program.


### Input Files

This program needs an input of brightfield TIFF images (central quadrant, 16-bit) captured as stacks of two channels: 
  - Channel 1: Red channel    (W1)
  - Channel 2: Green channel  (W2)
  
From the input images, the program runs through different steps: **image preprocesing**, **Spot Detection** (Trackpy), and **Spot selection.**

* Input images are first preprocessed with a *background subtraction* and *median filter* to reduce the background noise of the images. 
* Chromatic aberration correction using synthetic beads. Beads in W1 (red) are aligned to beads of W2 (green, reference).
* Spot Detection and linking using trackpy, to detect and link bright spots of radius ~ 5nm in both channels.
* Spot selection:
    - Select spots in W1 and W2 based on distance to the contour of the cell (max distance: 8 - 10 px).
    - Select spots in W1 and W2 based on distance to the closest neighbour spot (min distance: 9 px).
    - Select spots in W1 and W2 based on the goodness of the gaussian fit, after fiting spot intensitites to a gaussian distribution.
    - Select spots in W1 and W2 based on a density probability estimation (KDE), assuming that here we discard all spots that are not "in focus".
    - Outlier rejection.



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


