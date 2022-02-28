# Generative Adversarial Network - Fake Houses

GAN implementation for House-Thumbnail Images by Artan Zandian & Ramiro Mejia  
February 2022

## Objective

This project aims to implement a Generative Adversarial Networks (GANs) model to create synthetic thumbnail images of the exterior of houses that mimic real training images. This was a real-world problem for Realtor.com who did a capstone project in partnership with UBC Master of Data Science program 2 years ago. Our target was to recreate the results and simulate the problem by creating our own house images [dataset](https://www.kaggle.com/ramiromep/house-thumbnail) in two weeks rather than the original capstone timeline of two months.

GANs modelling is a difficult task since it is computationally expensive and requires GPU-accelerated frameworks to be trained. Also, GANs require typically thousands of training images to produce a high-quality model. In this project, [PyTorch](https://pytorch.org/) deep learning framework we will be used to train the models and the calculations will be performed using GPU.

The [House prices SoCal](https://www.kaggle.com/ted8080/house-prices-and-images-socal) is the base dataset used for this project, but due to low image count and poor quality of input data, additional web scraping of house images was performed to increase the number of images in the dataset and improve their quality.

In the following visualizations the real input images are presented and below that an animation with the synthetic images produced by the GAN model:

## Real images

<p align="center">
  <img src="https://github.com/artanzand/GAN/blob/main/examples/train_sample.JPG" />
</p>

## Outputs through epochs

<p align="center">
  <img src="https://github.com/artanzand/GAN/blob/main/examples/evolution.gif" />
</p>

## What is a Generative Adversarial Network (GAN)?

GANs are an approach to generate new data that is identical to the real data existing in a dataset using deep learning techniques.

GANs modeling is considered an unsupervised learning task which focus on learning patterns to produce new images that are as realistic as possible. GANs invention is credited to Ian Goodfellow, and was introduced in his famous paper [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661).

A GAN is made of two different models, a **generator** and a **discriminator**. The **generator** creates 'fake' images that look like the real training images. The **discriminator**  analyzes an image and decides whether or not it is a real training image or a fake image created by the generator.

The name **adversarial** comes from the performance of both models during the training phase.  The generator is trying to outsmart the discriminator by generating better 'fakes', while the discriminator is working on better classifying the real and fake images.

The stability of this model comes when the generator is generating perfect 'fake' images that look like the real training example, making the discriminator confused by always guessing a 50% probability that image is real or fake.

This is an example of a structure of a GAN:

![gans](https://user-images.githubusercontent.com/37048819/155857629-17fdc777-5056-4f97-864c-d7c9dad5fce6.png)
source: [Manning](https://freecontent.manning.com/practical-applications-of-gans-part-1/)

This is an example of a GAN model to produce images of handwritten numbers.

![dcgan](https://user-images.githubusercontent.com/37048819/155862019-3cd98231-aff4-4900-867d-db70886b1195.gif)

source: [Tensorflow](https://www.tensorflow.org/tutorials/generative/dcgan)

# Usage

## Cloning the Repo

Clone this Github repository and install the dependencies by running the following commands at the command line/terminal from the root directory of the project:

```
conda env create --file houseGAN.yaml 
conda activate houseGAN
```

## Web scraping

The below Python script can be used to download more images from Google.

```
from src.utils import google_downloader

google_downloader(search="houses vancouver", num_images=10, PATH="chromedrive.exe")
```

`google_downloader` uses [Selenium](https://selenium-python.readthedocs.io/) version 4.1.0.

To download images from Google using this function the user should define the following three arguments:


1. search : The title of your google search
2. path: Local path to the chromedriver.exe file required to use the Selenium library. The user can dowload the driver in [ChromeDriver](https://chromedriver.chromium.org/downloads) website and place it in a local folder.
3. num_images: The number of images to be downloaded

## Training the model

To train the GAN model and save the model weights run the below command line script in the `src/` directory. The two required arguments are the data (image) directory and the path to save the model weights. For a list of other options run `python train.py --help`.

```
python train.py --data_dir=../data --save_path=../model/model.pt
```

## Image Generation

Once model weights are created or saved in the `model/` directory, run the below command in the `src/` directory to create fake house images! Please replace the correct values for the arguments below. For list of other options run `python generate.py --help`.  

Note: The most important thing to know is that both training and image generation need to be only done on either GPU or CPU. For example, if you train on GPU you won't be able to generate images on CPU. The provided model weights in the repo are trained on cloud GPU.

```
python generate.py --num_examples=10 --save_path=../examples/house --model_path=../model/model.pt
```

# Data

## Original Data source

- [House Prices and images Socal](https://www.kaggle.com/ted8080/house-prices-and-images-socal)
- Google image searches of houses in North America

## Web scraping

After analyzing the quality of the images of the [Kaggle dataset](https://www.kaggle.com/ted8080/house-prices-and-images-socal), it was decided to implement web scraping to increase the number of consistent house images (house facades) so that the model could recognize more reliable patterns.

Using the [Selenium](https://selenium-python.readthedocs.io/) framework, web scraping was performed to download the images resulting from the following searches:
"Vancouver houses", "front yard houses", ''American houses", "Canadian houses".

After having a pool of images downloaded, the best possible house images were selected, the same was done for the images from [Kaggle dataset](https://www.kaggle.com/ted8080/house-prices-and-images-socal).

As part of the cleaning and quality control of the images, they were cropped and resized. The functions used to complete this job are found in [utils.py](https://github.com/artanzand/GAN/blob/main/src/utils.py).

## Final Data for Download

The final [dataset was uploaded to Kaggle](https://www.kaggle.com/ramiromep/house-thumbnail) for public usage and contains 9777 thumbnail images of houses.

# Dependencies

| Package      | Version |
|--------------|---------|
| matplotlib   | 3.5.1   |
| numpy        | 1.21.5  |
| pandas       | 1.4.1   |
| pytorch      | 1.10.2  |
| torchaudio   | 0.10.2  |
| torchvision  | 0.11.3  |
| urllib3      | 1.26.8  |
| selenium     | 4.1.0   |
| scikit-learn | 1.0.2   |
| scipy        | 1.7.3   |

## Results

A perfect model would create probability scores for the real image and the fake image which hover around 0.5. This would mean that the generator model has got to a state that creates images which discriminator is not able to distinguish from real! On the right, we wee the loss for both generator (loss_gen) and discriminator (loss_gen). In our experience, these losses oscillate a lot and at some points even cross each other which seems to be fine as long as they are under control. For some reason, we were not able to stabilize the models to continue improving images after 100 epochs.
<p align="center">
  <img src="https://github.com/artanzand/GAN/blob/main/examples/prob_loss.JPG" />
</p>

<p align="center">
  <img src="https://github.com/artanzand/GAN/blob/main/examples/combined.JPG" />
</p>

## Lessons Learned

- GAN modeling is a difficult task and finding a good architecture is not enough. Good quality representative images and a large training dataset are needed to have a successful model. The data collection is in particular difficult since there are not many appropriate images in public datasets.
- Training GANs is computationally demanding so setting up a virtual machine is recommended to reproduce this project. We used instance of Google Cloud Platform and AWS for this project.
- Outdoor images - It is very important for the images to be of same type, e.g., this model won't work if it is given images of house facades, perspective images, interiors, and site plans as input data.
- Weight decay - Using weight decay is a double-edge sword. Although it helps with model stabilization, the learning will stop as the model will get stuck in local optima.
- hyperparameters play the most important role after a good dataset. For recommendations on good starting points for hyperparameter values refer to this [repository](https://github.com/soumith/ganhacks).

# References

- [1] PyTorch GAN Tutorial - [tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [2] PyTorch reference NN - [website](https://pytorch.org/docs/stable/nn.html)
- [3] Sampling Generative Networks - [paper](https://arxiv.org/abs/1609.04468)
- [4] Jason Brownlee article about GANs - [article](https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/)
- [5] Andres Pitta article - [article](https://ubc-mds.github.io/2020-07-10-realistic-neighbourhoods/)
