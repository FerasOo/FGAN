import torch
import torch.nn.functional as F
from torch import nn

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 1024)
        self.fc2 = nn.Linear(1024, 7 * 7 * 128)
        self.t_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.t_conv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(7 * 7 * 128)
        self.bn3 = nn.BatchNorm2d(64)

        for layer in [self.fc1, self.fc2, self.t_conv1, self.t_conv2]:
            torch.nn.init.normal_(layer.weight, mean=0, std=0.02)

    def forward(self, x):
        out = self.bn1(F.relu(self.fc1(x)))
        out = self.bn2(F.relu(self.fc2(out)))
        out = out.reshape(-1, 128, 7, 7)
        out = self.bn3(F.relu(self.t_conv1(out)))
        out = F.tanh(self.t_conv2(out))
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024, 1024)
        self.fc2 = nn.Linear(1024, 1)

        for layer in [self.conv1, self.conv2, self.fc1, self.fc2]:
            torch.nn.init.normal_(layer.weight, mean=0, std=0.02)

    def forward(self, x):
        #   print(x.shape)
        out = x.reshape(-1, 1, 28, 28)
        out = self.leaky_relu(self.conv1(out))
        out = self.leaky_relu(self.conv2(out))
        out = out.reshape(-1, 7 * 7 * 64)
        out = self.leaky_relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out


def generator_loss(G_out, D_out, alpha, beta, power=2):
    loss_e = F.binary_cross_entropy(D_out, torch.full_like(D_out, alpha))

    center = torch.mean(G_out, dim=0)
    distance = torch.subtract(G_out, center)
    norm = torch.norm(distance, dim=(1, 2, 3), p=power)
    loss_d = torch.reciprocal(torch.mean(norm))

    return loss_e + beta * loss_d


def discriminator_loss(y_pred_real, y_pred_gen, gamma):
    real_loss = F.binary_cross_entropy(y_pred_real, torch.ones_like(y_pred_real))
    gen_loss = F.binary_cross_entropy(y_pred_gen, torch.zeros_like(y_pred_gen))

    return real_loss + gamma * gen_loss

def train_discriminator(D, G, D_optimizer, X_real, args):
    G.eval()
    D.train()
    X_real = X_real.to(args.device)
    X_gen = torch.normal(0, 1, size=(args.batch_size, args.latent_dim)).to(args.device)

    y_pred_real = D(X_real)
    y_pred_gen = D(G(X_gen))

    loss = discriminator_loss(y_pred_real, y_pred_gen, args.gamma)

    D_optimizer.zero_grad()
    loss.backward()
    D_optimizer.step()

    return loss.item()


def train_generator(D, G, G_optimizer, args):
    G.train()
    D.eval()
    X_gen = torch.normal(0, 1, size=(args.batch_size, args.latent_dim)).to(args.device)

    G_out = G(X_gen)
    D_out = D(G_out)
    loss = generator_loss(G_out, D_out, args.alpha, args.beta)

    G_optimizer.zero_grad()
    loss.backward()
    G_optimizer.step()

    return loss.item()

