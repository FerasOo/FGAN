import torch
import numpy as np
from torch.optim import Adam
import os

from fgan import Generator, Discriminator, train_generator, train_discriminator
from data import plot_boundary



def train_fgan(args):
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.pictures_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    G = Generator(args.g_hidden).to(args.device)
    D = Discriminator(args.d_hidden).to(args.device)

    G_optimizer = Adam(G.parameters(), lr=args.d_lr)
    D_optimizer = Adam(D.parameters(), lr=args.g_lr)
    print("========== PRETRAIN START ==========")
    for epoch in range(1, args.pretrain + 1):
        d_loss = train_discriminator(D, G, D_optimizer, args)
    print("d_loss:{}".format(d_loss))
    print("========== PRETRAIN END ==========")

    print("========== GAN TRAIN START ==========")
    for epoch in range(1, args.epochs + 1):
        d_loss = train_discriminator(D, G, D_optimizer, args)
        g_loss = train_generator(D, G, G_optimizer, args)
        if epoch % args.loss_freq == 0:
            print("epoch:{}, d_loss:{}, g_loss:{}".format(epoch, d_loss, g_loss))
        if epoch % args.plot_freq == 0:
            plot_boundary(G, D, args, title='model_{}epoch'.format(epoch))
    torch.save(G.state_dict(), "{}/model_G.pth".format(args.models_dir))
    torch.save(D.state_dict(), "{}/model_D.pth".format(args.models_dir))
    print("========== GAN TRAIN END ==========")