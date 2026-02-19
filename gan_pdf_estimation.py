import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

ROLL = 102303497
DATA_FILE = r"C:\Users\SARTHAK GABA\Manager\Thapar\3rd year\evensem (6th sem)\predictive stats\datasets\India Air Quality Data\data.csv"

df = pd.read_csv(DATA_FILE, low_memory=False, encoding='latin1')
x = df['no2'].dropna().values.astype(np.float32)

r = ROLL
a_r = 0.5 * (r % 7)
b_r = 0.3 * ((r % 5) + 1)
z = x + a_r * np.sin(b_r * x)

print(f"Roll: {ROLL}")
print(f"a_r = {a_r}, b_r = {b_r}")
print(f"Total samples after dropping NaN: {len(z)}")

z_mean = z.mean()
z_std = z.std()
z_norm = (z - z_mean) / z_std

real_data = torch.tensor(z_norm, dtype=torch.float32).unsqueeze(1)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


gen = Generator()
disc = Discriminator()

lr = 0.0002
opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

loss_fn = nn.BCELoss()

epochs = 2000
batch_size = 256
n_samples = len(real_data)

d_losses = []
g_losses = []

print("Training GAN...")

for epoch in range(epochs):
    idx = torch.randint(0, n_samples, (batch_size,))
    real_batch = real_data[idx]

    noise = torch.randn(batch_size, 1)
    fake_batch = gen(noise)

    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    d_real_out = disc(real_batch)
    d_fake_out = disc(fake_batch.detach())

    d_loss_real = loss_fn(d_real_out, real_labels)
    d_loss_fake = loss_fn(d_fake_out, fake_labels)
    d_loss = d_loss_real + d_loss_fake

    opt_d.zero_grad()
    d_loss.backward()
    opt_d.step()

    noise2 = torch.randn(batch_size, 1)
    fake_batch2 = gen(noise2)
    g_out = disc(fake_batch2)
    g_loss = loss_fn(g_out, real_labels)

    opt_g.zero_grad()
    g_loss.backward()
    opt_g.step()

    d_losses.append(d_loss.item())
    g_losses.append(g_loss.item())

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

print("Training done.")

num_generated = 20000
noise_final = torch.randn(num_generated, 1)
with torch.no_grad():
    generated_norm = gen(noise_final).numpy().flatten()

generated_z = generated_norm * z_std + z_mean

print(f"\nReal data  -> mean: {z.mean():.4f}, std: {z.std():.4f}")
print(f"Generated  -> mean: {generated_z.mean():.4f}, std: {generated_z.std():.4f}")

kde = gaussian_kde(generated_z, bw_method='scott')
z_range = np.linspace(z.min() - 10, z.max() + 10, 1000)
kde_vals = kde(z_range)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(z, bins=100, density=True, alpha=0.5, color='steelblue', label='Real Data (z)')
axes[0].plot(z_range, kde_vals, 'r-', linewidth=2, label='GAN KDE Estimate')
axes[0].set_xlabel('z')
axes[0].set_ylabel('Density')
axes[0].set_title('PDF Estimation using GAN')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(d_losses, label='Discriminator Loss', alpha=0.7)
axes[1].plot(g_losses, label='Generator Loss', alpha=0.7)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Training Loss Curves')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gan_pdf_result.png', dpi=150)
print("Saved plot to gan_pdf_result.png")

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.hist(z, bins=100, density=True, alpha=0.4, color='steelblue', label='Real Data (z)')
ax2.hist(generated_z, bins=100, density=True, alpha=0.4, color='salmon', label='Generated Data')
ax2.set_xlabel('z')
ax2.set_ylabel('Density')
ax2.set_title('Real vs Generated Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gan_comparison.png', dpi=150)
print("Saved comparison plot to gan_comparison.png")
