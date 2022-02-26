# Generative Adversarial Network
GAN implementation for Realtor.com Images by Artan Zandian & Ramiro Mejia

## Objective

This project aims to implement a Generative Adversarial Networks (GANs) model to create synthetic thumbnail images of the exterior of houses that mimic real training images.

GANs modelling is a difficult task since it is computational expensive, it requires GPU-accelerated frameworks to be trained. Also GANs require tipically thousands of training images to produce a high-quality model. In this project [Pytorch](https://pytorch.org/) deep learning framework that we will use to train the models and the calculations will be performed using GPU. 


The following images show the the model outputs synthetic images of houses through epochs:

## Real images

<p align="center">
  <img src="https://github.com/artanzand/GAN/blob/main/examples/train_sample.JPG" />
</p>

## Outputs through epochs

<p align="center">
  <img src="https://github.com/artanzand/GAN/blob/main/examples/evolution.gif" />
</p>


## What is a Generative Adversarial Network (GAN)?

GANs are an approach to generate new data that is identical from the real data existing in a dataset using deep learning techniques.

GANs modeling is considered an unsupervised learning task which focus on learning patterns to produce new images that are realistic as possible. GANs invention is credited to Ian Goodfellow, they were  introduced in his famouse paper [Generative Adversarial Nets](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1acccf3-Paper.pdf).

A GAN is made of two different models, a **generator** and a **discriminator**. The **generator** creates 'fake' images that look like the real training images. The **discriminator**  analyze an image and decide whether or not it is a real training image or a fake image created by the generator.

The name **adversarial** comes from the performance of both models during the training phase.  The generator is trying to outsmart the discriminator by generating better 'fakes', while the discriminator is working on classify  better the real and fake images.

The stability of this model  comes when the generator is generating perfect 'fakes' images that look like the training example, thus the generator is confused and always guess at 50% that image is real or fake. 

This is an example of a structure of a GAN:

![gans](https://user-images.githubusercontent.com/37048819/155857629-17fdc777-5056-4f97-864c-d7c9dad5fce6.png)
source: [Manning](https://freecontent.manning.com/practical-applications-of-gans-part-1/)

This is an example of GAN model to produce images of handwritten numbers

![dcgan](https://user-images.githubusercontent.com/37048819/155862019-3cd98231-aff4-4900-867d-db70886b1195.gif)




# Usage
## Cloning the Repo
Clone this Github repository and install the dependencies by running the following commands at the command line/terminal from the root directory of the project:

```
conda env create --file environment.yaml 
conda activate GAN
```

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

Analyzing the quality of the images of the [Kaggle dataset](https://www.kaggle.com/ted8080/house-prices-and-images-socal) it was decided to implement webscrapping techniques to increase the number of images of houses that are similar thus the model can capture better patterns.

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
<p align="center">
  <img src="https://github.com/artanzand/GAN/blob/main/examples/prob_loss.JPG" />
</p>

<p align="center">
  <img src="https://github.com/artanzand/GAN/blob/main/examples/combined.JPG" />
</p>



## Lessons Learned
- GAN modeling is a difficult task. Finding a good architecture is not enough, good quality images and a large training dataset are needed to have a successful model. The data collection is in particular difficult since there is not many appropiate images in the public datasets
- 
- Outdoor images
- Weight decay
- hyperparameters

# References
- [1] pytorch GAN Tutorial  - [tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [2] pytorch reference NN - [website](https://pytorch.org/docs/stable/nn.html)
- [3] Sampling Generative Networks - [paper](https://arxiv.org/abs/1609.04468)
- [4] Jason Brownlee article about GANs - [article](https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/)
