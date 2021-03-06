# author: Artan Zandian
# date: 2022-02-28

"""
Accepts the image data directory and saves the weights of the Generative Adversarial Network in a .pt file.

Usage: train.py [--data_dir=<data_directory> --save_path=<save_path> --batch_size=<batch_size> --latent_size=<latent_size> --epochs=<num_epochs>]

Options:
--data_dir=<data_directory>       path to the image directory [default: ../data]  
--save_path=<save_path>           path to where the model weights will be saved [default: ../model/model.pt]
--batch_size=<batch_size>         size of batches for the training [default: 32]
--latent_size=<latent_size>       size of the initial input vector to the generator [default: 96]
--epochs=<num_epochs>             number of epochs [default: 50]
"""

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from statistics import mean
from docopt import docopt

opt = docopt(__doc__)


def main(
    data_dir="../data",
    save_path="../model/model.pt",
    batch_size=32,
    latent_size=96,
    epochs=50,
):
    """
    Accepts the image data directory and saves the weights of the Generative Adversarial Network in a model.pt file.

    Parameters
    ----------
    data_dir: str
        path to the image directory
    save_path: str
        path to where the model weights will be saved
    batch_size: int, optional
        size of batches for the training
    latent_size: int, optional
        size of the initial input vector to the generator
    epochs: int, optional
        number of epochs

    Returns
    -------
    Saved generator and discriminator weights on save_path
    """
    try:
        batch_size = int(batch_size)
        latent_size = int(latent_size)
        epochs = int(epochs)
    except:
        print("Did you give integer values for numbers and sizes?")

    # Use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transformer setup
    transformer = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    try:
        # Create dataset and loader
        dataset = datasets.ImageFolder(root=data_dir, transform=transformer)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        # Initiate generator and discriminator and weights
        generator = Generator(latent_size)
        discriminator = Discriminator()

        generator.apply(weights_init)
        discriminator.apply(weights_init)

        # Use GPU if available
        generator.to(device)
        discriminator.to(device)

        # Set criterion and optimizers
        criterion = nn.BCELoss()
        optimizerG = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        optimizerD = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

        # Train Generator and Distcriminator
        D_real_epoch, D_fake_epoch, loss_dis_epoch, loss_gen_epoch = trainer(
            data_loader,
            generator,
            discriminator,
            criterion,
            optimizerG,
            optimizerD,
            epochs=epochs,
        )

        # Save model
        torch.save(
            {
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizerG_state_dict": optimizerG.state_dict(),
                "optimizerD_state_dict": optimizerD.state_dict(),
                "D_real_epoch": D_real_epoch,
                "D_fake_epoch": D_fake_epoch,
                "loss_dis_epoch": loss_dis_epoch,
                "loss_gen_epoch": loss_gen_epoch,
            },
            save_path,
        )

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


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            # input dim: [-1, 64, 96, 128]
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(
                64
            ),  # Turning bias off for Conv2d as batchnorm has a built-in bias
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3),
            # output dim: [-1, 64, 48, 64]
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3),
            # output dim: [-1, 128, 24, 32]
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3),
            # output dim: [-1, 256, 12, 16]
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3),
            # output dim: [-1, 512, 6, 8]
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3),
            # output dim: [-1, 1, 3, 5]
            nn.Flatten(),
            # output dim: [-1, 15]
            nn.Linear(15, 1),
            # output dim: [-1]
            nn.Sigmoid()
            # output dim: [-1]
        )

    def forward(self, input):
        output = self.main(input)
        return output


def trainer(
    data_loader,
    generator,
    discriminator,
    criterion,
    optimizerG,
    optimizerD,
    latent_size=96,
    epochs=10,
):
    """ """
    torch.manual_seed(13)

    # Use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type}")

    print("Training started:\n")
    # Track overal metrics per epoch
    D_real_epoch, D_fake_epoch, loss_dis_epoch, loss_gen_epoch = [], [], [], []

    for epoch in range(epochs):
        # Initiate losses for this epoch
        D_real_iter, D_fake_iter, loss_dis_iter, loss_gen_iter = [], [], [], []

        for real_batch, _ in data_loader:

            # STEP 1: train discriminator
            # ==================================
            # Train with real data
            discriminator.zero_grad()

            real_batch = real_batch.to(device)
            real_labels = torch.ones((real_batch.shape[0],), dtype=torch.float).to(
                device
            )

            output = discriminator(real_batch).view(-1)
            loss_real = criterion(output, real_labels)

            # Iteration book-keeping
            D_real_iter.append(output.mean().item())

            # Train with fake data
            noise = torch.randn(real_batch.shape[0], latent_size, 1, 1).to(device)

            fake_batch = generator(noise)
            fake_labels = torch.zeros_like(real_labels)

            output = discriminator(fake_batch.detach()).view(-1)
            loss_fake = criterion(output, fake_labels)

            # Update discriminator weights
            loss_dis = loss_real + loss_fake
            loss_dis.backward()
            optimizerD.step()

            # Iteration book-keeping
            loss_dis_iter.append(loss_dis.mean().item())
            D_fake_iter.append(output.mean().item())

            # STEP 2: train generator
            # ==================================
            generator.zero_grad()
            output = discriminator(fake_batch).view(-1)
            loss_gen = criterion(output, real_labels)
            loss_gen.backward()

            # Book-keeping
            loss_gen_iter.append(loss_gen.mean().item())

            # Update generator weights and store loss
            optimizerG.step()

        print(
            f"Epoch ({epoch + 1}/{epochs})\t",
            f"Loss_G: {mean(loss_gen_iter):.4f}",
            f"Loss_D: {mean(loss_dis_iter):.4f}\t",
            f"D_real: {mean(D_real_iter):.4f}",
            f"D_fake: {mean(D_fake_iter):.4f}",
        )

        # Epoch book-keeping
        loss_gen_epoch.append(mean(loss_gen_iter))
        loss_dis_epoch.append(mean(loss_dis_iter))
        D_real_epoch.append(mean(D_real_iter))
        D_fake_epoch.append(mean(D_fake_iter))

    print("\nTraining ended.")
    return D_real_epoch, D_fake_epoch, loss_dis_epoch, loss_gen_epoch


def weights_init(layer):
    """
    Initializes the weights of the generator and discriminator for optimized performance.

    Parameters
    ----------
    layer: nn.Module layer
        Weights of a layer
    """
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(layer.weight.data, 0.0, 0.01)
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.normal_(layer.weight.data, 1.0, 0.01)
        nn.init.constant_(layer.bias.data, 0)


if __name__ == "__main__":
    main(
        opt["--data_dir"],
        opt["--save_path"],
        opt["--batch_size"],
        opt["--latent_size"],
        opt["--epochs"],
    )
