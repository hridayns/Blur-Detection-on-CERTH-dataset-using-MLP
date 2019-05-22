## Project Details

Used *Laplace of the Variance* as the sole feature to train MLP (MultiLayer Perceptron) that can determine if an image is blurred or not, with an **accuracy of 83.04%**.

**Dataset used:** CERTH image blur dataset found [here](https://mklab.iti.gr/results/certh-image-blur-dataset/)

## Steps to install and run

1. Download and install the latest version of Anaconda (Python 3.6) from here(https://www.anaconda.com/download) (Will include *numpy*, *pandas* and *scikit-learn* by default).
2. Both *pip* and *conda* are included in Anaconda and Miniconda, so you do not need to install them separately.
3. Install latest version of OpenCV using `pip install opencv-python`
4. Run script (both train and test script included) using `python blur_detector.py` and wait for the prompt to stop.
