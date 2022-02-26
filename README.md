# Generative Adversarial Network
GAN implementation for Realtor.com Images

What GAN is and what it does. 
The scope of the project
We want to create fake images for properties in a Real Estate company that do not have any images. 
- image of the house outside - residential(single, semi-attached, condo, highrise)?
- interior (room, ...)
- garden...
We need over 10_000 images.

<p align="center">
  <img src="https://github.com/artanzand/GAN/blob/main/examples/train_sample.JPG" />
</p>

- animation

# Usage
## Webscraping

```
from scr.utils import google_downloader

google_downloader('houses vancouver', 10, 'chromedrive.exe')

```
`google_downloader` uses [Selenium](https://selenium-python.readthedocs.io/) version 4.1.0

To download images from Google using this function the user should define the following three arguments: 

1. search : The title of your google search 
2. path: Local path to the chromedriver.exe file required to use the Selenium library
3. num_images: The number of images to be downloadede

## Training the model
The function to train. This doesn't take any arguments.

## Image Creation
The function with predicted images
```
```
num_images




# Data

## Original Data source

- House pricesand images socal - [Kaggle](https://www.kaggle.com/ted8080/house-prices-and-images-socal)
- Google images searches of houses


## Webscraping

Analyzing the quality of the images of the [Kaggle dataset](https://www.kaggle.com/ted8080/house-prices-and-images-socal) it was decided to implement webscrapping techniques to increase the number of images of houses that are similar and in this way the model can capture better patterns.

Using the [Selenium](https://selenium-python.readthedocs.io/) framework, webscrapping was performed to download the images resulting from the following searches:
"vancouver houses", "front yard houses", ''american houses", "canadian houses".

After having a pool of images downloaded, the best possible house images were selected, the same was done for the images from [Kaggle dataset](https://www.kaggle.com/ted8080/house-prices-and-images-socal).

As part of the cleaning and quality control of the images, they were cropped and resized. The functions used to complete this job are found in [utils.py](https://github.com/artanzand/GAN/blob/main/src/utils.py)


## Final Data for Download

The final [dataset was uploaded to Kaggle](https://www.kaggle.com/ramiromep/house-thumbnail) for public usage.



# Dependencies
python packages
project environment needs to be updated too


## Results
probability and loss results images

## Lessons Learned
- data size
- Outdoor images
- Weight decay
- hyperparameters

# References
[1] pytorch GAN Tutorial  - [tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
[2] pytorch reference NN - [website](https://pytorch.org/docs/stable/nn.html)
[3] Sampling Generative Networks - [paper](https://arxiv.org/abs/1609.04468)
