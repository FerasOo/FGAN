import torch
import numpy as np
from torch.optim import Adam
import os

from data import get_data, compute_auprc
from fgan import Generator, Discriminator, train_generator, train_discriminator

def train_fgan(args):
    os.makedirs(args.models_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, valid_loader, test_loader = get_data(args.ano_class, args.batch_size)

    G = Generator(args.latent_dim).to(args.device)
    D = Discriminator().to(args.device)
    G_optimizer = Adam(G.parameters(), lr=args.g_lr, weight_decay=1e-4, betas=(0.5, 0.999))
    D_optimizer = Adam(D.parameters(), lr=args.d_lr, weight_decay=1e-4, betas=(0.5, 0.999))

    print("========== START TRAINING GAN -- ANOMALOUS CLASS:{} ==========".format(args.ano_class))
    for epoch in range(1, args.pretrain + 1):
        total_loss_d, total_size = 0, 0
        for data, _ in train_loader:
            d_loss = train_discriminator(D, G, D_optimizer, data, args)
            total_size += 1
            total_loss_d += d_loss
        if epoch % args.loss_freq == 0:
            print("epoch:{}, d_loss:{:.6f}".format(epoch, total_loss_d / total_size))


    best_auprc = 0
    for epoch in range(1, args.epochs + 1):

        # train
        total_loss_d, total_loss_g, total_size = 0, 0, 0
        for data, _ in train_loader:
            d_loss = train_discriminator(D, G, D_optimizer, data, args)
            g_loss = train_generator(D, G, G_optimizer, args)
            total_size += 1
            total_loss_d += d_loss
            total_loss_g += g_loss

        # valid
        # if epoch < 10:
        #     continue

        auprc = compute_auprc(D, valid_loader, args.device)
        if auprc > best_auprc:
            best_auprc = auprc
            g_path = "{}/g_anomalous_{}_best.pth".format(args.models_dir, args.ano_class)
            d_path = "{}/d_anomalous_{}_best.pth".format(args.models_dir, args.ano_class)
            torch.save(G.state_dict(), g_path)
            torch.save(D.state_dict(), d_path)
            if epoch % args.loss_freq == 0:
                print("epoch:{}, d_loss:{:.6f}, g_loss:{:.4f}, AUPRC:{:.4f}"
                      .format(epoch, total_loss_d/total_size, total_loss_g/total_size, auprc))

    g_path = "{}/g_anomalous_{}_final.pth".format(args.models_dir, args.ano_class)
    d_path = "{}/d_anomalous_{}_final.pth".format(args.models_dir, args.ano_class)
    torch.save(G.state_dict(), g_path)
    torch.save(D.state_dict(), d_path)

    # load best model to evaluate on test set
    d_best_path = "{}/d_anomalous_{}_best.pth".format(args.models_dir, args.ano_class)
    D.load_state_dict(torch.load(d_best_path))
    auprc_best = compute_auprc(D, test_loader, args.device)
    print("SCORE ON TEST SET : {:.4f}".format(auprc_best))