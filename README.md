# Fence GAN
Pytorch implementation of [Fence GAN: Towards Better Anomaly Detection](https://arxiv.org/abs/1904.01209)

### 2D Experiment
```
python3 2D/main.py
```
### training options
```
usage: 2D experiment [-h] [--alpha ALPHA] [--beta BETA] [--gamma GAMMA] [--power POWER] [--pretrain PRETRAIN] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                     [--distribution DISTRIBUTION] [--device DEVICE] [--models_dir MODELS_DIR] [--pictures_dir PICTURES_DIR] [--seed SEED] [--d_lr D_LR]
                     [--g_lr G_LR] [--d_hidden D_HIDDEN] [--g_hidden G_HIDDEN] [--plot_freq PLOT_FREQ] [--loss_freq LOSS_FREQ]

options:
  -h, --help            show this help message and exit
  --alpha ALPHA         alpha
  --beta BETA           beta
  --gamma GAMMA         gamma
  --power POWER         norm used for dispersion loss
  --pretrain PRETRAIN   number of pretrain epoch
  --epochs EPOCHS       number of epochs
  --batch_size BATCH_SIZE
                        batch size
  --distribution DISTRIBUTION
                        normal | square | triangular | bow_shaped | oval
  --device DEVICE       cpu | cuda
  --models_dir MODELS_DIR
                        folder to save model weights
  --pictures_dir PICTURES_DIR
                        folder to save images
  --seed SEED           Numpy and Pytorch seed
  --d_lr D_LR           learning_rate of discriminator
  --g_lr G_LR           learning rate of generator
  --d_hidden D_HIDDEN   number of neurons in hidden layers of discriminator
  --g_hidden G_HIDDEN   number of neurons in hidden layers of generator
  --plot_freq PLOT_FREQ
                        epoch frequency to save images
  --loss_freq LOSS_FREQ
                        epoch frequency to print loss
```

### MNIST
```
python3 MNIST/main.py
```
### training options
```
usage: 2D experiment [-h] [--ano_class ANO_CLASS] [--alpha ALPHA] [--beta BETA] [--gamma GAMMA] [--power POWER] [--pretrain PRETRAIN] [--epochs EPOCHS]
                     [--batch_size BATCH_SIZE] [--latent_dim LATENT_DIM] [--device DEVICE] [--models_dir MODELS_DIR] [--seed SEED] [--d_lr D_LR]
                     [--g_lr G_LR] [--loss_freq LOSS_FREQ]

options:
  -h, --help            show this help message and exit
  --ano_class ANO_CLASS
                        digit to set at anomalous class
  --alpha ALPHA         alpha
  --beta BETA           beta
  --gamma GAMMA         gamma
  --power POWER         norm used for dispersion loss
  --pretrain PRETRAIN   number of pretrain epoch
  --epochs EPOCHS       number of epochs
  --batch_size BATCH_SIZE
                        batch size
  --latent_dim LATENT_DIM
                        latent dimension of Gaussian noise input to Generator
  --device DEVICE       cpu | cuda
  --models_dir MODELS_DIR
                        folder to save model weights
  --seed SEED           Numpy and Pytorch seed
  --d_lr D_LR           learning_rate of discriminator
  --g_lr G_LR           learning rate of generator
  --loss_freq LOSS_FREQ
                        epoch frequency to print loss

```

###
you will find two notebooks in "jupyter notebook" file, these notebooks contain modified and additional code from Python files. The notebooks were used to generate images that were used during the presentation

### DISCLAIMER: The notebook cells were executed during different sessions and environments. To re-run the cells on your device you might need to modify the code slightly  
