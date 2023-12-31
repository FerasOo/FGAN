import torch
import torch.nn.functional as F
import torch.nn as nn
from data import get_data


class Generator(nn.Module):
    def __init__(self, hidden_dim=10):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out) + x
        return out


class Discriminator(nn.Module):
    def __init__(self, hidden_dim=15):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        return out


def generator_loss(G_out, D_out, alpha, beta, power):
    loss_e = F.binary_cross_entropy(D_out, torch.full_like(D_out, alpha))

    center = torch.mean(G_out, dim=0)
    distance = torch.subtract(G_out, center)
    norm = torch.norm(distance, dim=1, p=power)
    loss_d = torch.reciprocal(torch.mean(norm))

    return loss_e + beta * loss_d


def discriminator_loss(y_pred_real, y_pred_gen, gamma):
    real_loss = F.binary_cross_entropy(y_pred_real, torch.ones_like(y_pred_real))
    gen_loss = F.binary_cross_entropy(y_pred_gen, torch.zeros_like(y_pred_real))

    return real_loss + gamma * gen_loss


def train_discriminator(D, G, D_optimizer, args):
    X_real = get_data(args.distribution, args.batch_size).to(args.device)
    X_gen = get_data('noise', args.batch_size).to(args.device)

    y_pred_real = D(X_real)
    y_pred_gen = D(G(X_gen))
    loss = discriminator_loss(y_pred_real, y_pred_gen, args.gamma)

    D_optimizer.zero_grad()
    loss.backward()
    D_optimizer.step()

    return loss.item()


def train_generator(D, G, G_optimizer, args):
    X_gen = get_data('noise', args.batch_size).to(args.device)
    G_out = G(X_gen)
    D_out = D(G_out)
    loss = generator_loss(G_out, D_out, args.alpha, args.beta, args.power)

    G_optimizer.zero_grad()
    loss.backward()
    G_optimizer.step()

    return loss.item()
