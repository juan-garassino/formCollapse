import torch
import torch.nn as nn
import torch.optim as optim

def train_gan(generator, discriminator, dataloader, num_epochs, latent_dim, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for i, real_data in enumerate(dataloader):
            batch_size = real_data.size(0)
            real_data = real_data.to(device)

            # Train Discriminator
            optimizer_D.zero_grad()
            
            label = torch.full((batch_size,), 1, device=device)
            output = discriminator(real_data).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_data = generator(noise)
            label.fill_(0)
            output = discriminator(fake_data.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            
            errD = errD_real + errD_fake
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            label.fill_(1)
            output = discriminator(fake_data).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizer_G.step()

        print(f'[Epoch {epoch}/{num_epochs}] Loss_D: {errD.item():.4f}, Loss_G: {errG.item():.4f}')

# Usage:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# generator.to(device)
# discriminator.to(device)
# train_gan(generator, discriminator, dataloader, num_epochs=50, latent_dim=100, device=device)