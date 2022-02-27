# author: Artan Zandian
# date: 2022-02-28

"""
Generates requested number of images based of number of examples.

Usage: python generate.py --num_examples=<num examples> --save_path=<save path> --model_path=<model path> --batch_size=<batch size> --latent_size=<latent size>

Options:
--num_examples=<num examples>     number of examples to generate     
--save_path=<save path>           path to where the generate images will be saved
--model_path=<model path>         path where the model weights is stored
--batch_size=<batch size>         size of batches for the training
--latent_size=<latent size>       size of the initial input vector to the generator
"""

import torch
from torch import nn
from torchvision import utils
from torchvision.utils import save_image
import sys
from docopt import docopt

opt = docopt(__doc__)


def generate(
    num_examples,
    save_path,
    model_path="../model/model.pt",
    batch_size=32,
    latent_size=96,
):
    """
    Generates requested number of images based of number of examples.

    Parameters
    ----------
    num_examples: int
        number of examples to generate
    save_path: str
        path to where the generate images will be saved
    model_path: str, optional
        path where the model weights is stored
    batch_size: int, optional
        size of batches for the training
    latent_size: int, optional
        size of the initial input vector to the generator
    """
    if num_examples > batch_size:
        sys.exit("num_examples should be smaller than the batch size")

    try:
        # Create an instance of the generator
        generator = Generator(latent_size)

        # Load model weights and losses
        checkpoint = torch.load(model_path)
        generator.load_state_dict(checkpoint["generator_state_dict"])

        # Set generator to evaluation mode
        generator.eval()

        # sending the loaded models to cuda (the previous model has been trained on one)
        # moving model to gpu needs to happen before constructing optimizer
        if torch.cuda.is_available():
            generator.cuda()

        # Create an image list of batch size
        img_list = []

        # Use gpu if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device.type}")

        fixed_noise = torch.randn(batch_size, latent_size, 1, 1).to(device)
        # Keeping track of the evolution of a fixed noise latent vector
        with torch.no_grad():
            generated_img = generator(fixed_noise).detach().cpu()
            for i in range(batch_size):
                # Isolate each noise tensor
                img_list.append(generated_img[i, :, :, :])

        for i, image in enumerate(img_list[:num_examples]):
            save_image(image, save_path + str(i) + ".png")
            print(f"Image {i} saved.")

    except Exception as ex:
        print(ex)


class Generator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        self.main = nn.Sequential(
            # input dim: [-1, latent_size, 1, 1]
            nn.ConvTranspose2d(
                latent_size, 512, kernel_size=(3, 4), stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            # output dim: [-1, 512, 3, 4]
            nn.ConvTranspose2d(
                512, 256, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            # output dim: [-1, 256, 6, 8]
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            # output dim: [-1, 128, 12, 16]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.3),
            # output dim: [-1, 64, 24, 32]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),
            # output dim: [-1, 32, 48, 64]
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1, inplace=True),
            # output dim: [-1, 3, 96, 128]
            nn.Tanh(),
        )

    def forward(self, input):
        output = self.main(input)
        return output


if __name__ == "__main__":
    generate(
        opt["--num_examples"],
        opt["--save_path"],
        opt["--model_path"],
        opt["--batch_size"],
        opt["--latent_size"],
    )
