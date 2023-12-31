import argparse
from fgan_train import train_fgan
parser = argparse.ArgumentParser('2D experiment')

parser.add_argument('--ano_class', type=int, default=0, help='digit to set at anomalous class')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
parser.add_argument('--beta', type=float, default=30, help='beta')
parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
parser.add_argument('--power', type=int, default=2, help='norm used for dispersion loss')
parser.add_argument('--pretrain', type=int, default=15, help='number of pretrain epoch')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=200, help='batch size')
parser.add_argument('--latent_dim', type=int, default=200, help='latent dimension of Gaussian noise input to Generator')
parser.add_argument('--device', type=str, default='cpu', help='cpu | cuda')


parser.add_argument('--models_dir', type=str, default='MNIST/models', help='folder to save model weights')
parser.add_argument('--seed', type=int, default=0, help='Numpy and Pytorch seed')
parser.add_argument('--d_lr', type=float, default=1e-5, help='learning_rate of discriminator')
parser.add_argument('--g_lr', type=float, default=2e-5, help='learning rate of generator')
parser.add_argument('--loss_freq', type=int, default=1, help='epoch frequency to print loss')

args = parser.parse_args()

train_fgan(args)
