import argparse
from fgan_train import train_fgan
parser = argparse.ArgumentParser('2D experiment')

parser.add_argument('--alpha', type=float, default=0.5, help='alpha')
parser.add_argument('--beta', type=float, default=15, help='beta')
parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
parser.add_argument('--power', type=int, default=2, help='norm used for dispersion loss')
parser.add_argument('--pretrain',type=int, default=50, help='number of pretrain epoch')
parser.add_argument('--epochs', type=int, default=30000, help='number of epochs')
parser.add_argument('--batch_size',type=int, default=100, help='batch size')
parser.add_argument('--distribution',type=str, default='normal', help='normal | square | triangular | bow_shaped | oval')
parser.add_argument('--device', type=str, default='cpu', help='cpu | cuda')


parser.add_argument('--models_dir', type=str, default='2D/models', help='folder to save model weights')
parser.add_argument('--pictures_dir', type=str, default='2D/pictures', help='folder to save images')
parser.add_argument('--seed', type=int, default=0, help='Numpy and Pytorch seed')
parser.add_argument('--d_lr', type=float, default=1e-3, help='learning_rate of discriminator')
parser.add_argument('--g_lr', type=float, default=1e-3, help='learning rate of generator')
parser.add_argument('--d_hidden', type=int, default=8, help='number of neurons in hidden layers of discriminator')
parser.add_argument('--g_hidden', type=int, default=16, help='number of neurons in hidden layers of generator')
parser.add_argument('--plot_freq', type=int, default=5000, help='epoch frequency to save images')
parser.add_argument('--loss_freq', type=int, default=1000, help='epoch frequency to print loss')

args = parser.parse_args()

train_fgan(args)
