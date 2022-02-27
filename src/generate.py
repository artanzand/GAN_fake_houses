import torch
from torch import nn
from torchvision import utils
from torchvision.utils import save_image
import sys



def generate(num_examples, save_path, model_path="../input/models/model50 GAN_normal0.01_batch32.pt", BATCH_SIZE=32, LATENT_SIZE=96):
    """
    
    """
    if (num_examples > BATCH_SIZE):
        sys.exit("num_examples should be smaller than the batch size")
        
    # Create an instance of the generator
    generator = Generator(LATENT_SIZE)

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device.type}")

    fixed_noise = torch.randn(BATCH_SIZE, LATENT_SIZE, 1, 1).to(device)
    # Keeping track of the evolution of a fixed noise latent vector
    with torch.no_grad():
        generated_img = generator(fixed_noise).detach().cpu()
        for i in range(BATCH_SIZE):
            # Isolate each noise tensor
            
            img_list.append(generated_img[i,:,:,:])

    for i, image in enumerate(img_list[:num_examples]):
        save_image(image, save_path + str(i) + ".png") 
        print(f"Image {i} saved.")



class Generator(nn.Module):
    
    def __init__(self, LATENT_SIZE):
        super().__init__()
        
        self.main = nn.Sequential(
            # input dim: [-1, LATENT_SIZE, 1, 1]
            
            nn.ConvTranspose2d(LATENT_SIZE, 512, kernel_size=(3, 4), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),    
            # output dim: [-1, 512, 3, 4]

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            # output dim: [-1, 256, 6, 8]

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
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
            
            nn.Tanh()
        )
        
    def forward(self, input):
        output = self.main(input)
        return output