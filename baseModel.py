import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#Encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # (B, 32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # (B, 64, 7, 7)
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

#Reparameterization
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

#Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # (B, 32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),   # (B, 1, 28, 28)
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 64, 7, 7)
        return self.deconv(x)

#Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

#Training and Testing 
def train(model_components, dataloader, optimizers, criterion, epochs=10):
    encoder, decoder, discriminator = model_components
    opt_vae, opt_disc = optimizers

    for epoch in range(epochs):
        for i, (x, _) in enumerate(dataloader):
            x = x.cuda()
            batch_size = x.size(0)

            #VAE
            mu, logvar = encoder(x)
            z = reparameterize(mu, logvar)
            x_hat = decoder(z)

            recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum') / batch_size
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

            d_fake = discriminator(x_hat)
            gan_loss = torch.mean(torch.log(d_fake + 1e-8))

            vae_loss = recon_loss + kl_div - 0.001 * gan_loss

            opt_vae.zero_grad()
            vae_loss.backward()
            opt_vae.step()

            #Discriminator
            d_real = discriminator(x)
            d_fake = discriminator(x_hat.detach())

            d_loss = -torch.mean(torch.log(d_real + 1e-8) + torch.log(1 - d_fake + 1e-8))

            opt_disc.zero_grad()
            d_loss.backward()
            opt_disc.step()

        print(f"Epoch [{epoch+1}/{epochs}] | VAE Loss: {vae_loss.item():.4f} | D Loss: {d_loss.item():.4f}")


transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

latent_dim = 20

encoder = Encoder(latent_dim).cuda()
decoder = Decoder(latent_dim).cuda()
discriminator = Discriminator().cuda()

opt_vae = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
opt_disc = optim.Adam(discriminator.parameters(), lr=1e-4)

train((encoder, decoder, discriminator), dataloader, (opt_vae, opt_disc), nn.BCELoss(), epochs=100)
