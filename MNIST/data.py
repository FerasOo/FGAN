import torch
from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

def get_data(anomalous_class, batch_size):
    # load data
    train_data = MNIST(root='./MNIST/data/', train=True, download=True)
    test_data = MNIST(root='./MNIST/data/', train=False, download=True)

    # concat data and preprocess
    X_full = torch.concat([train_data.data, test_data.data])
    y_full = torch.concat([train_data.targets, test_data.targets])
    preprocess = transforms.Lambda(lambda x: (x / 255 - 0.5) * 2)
    X_full = preprocess(X_full)

    # split normal and anomalous classes
    y_normal_idx = y_full == anomalous_class
    X_normal = X_full[~y_normal_idx]
    X_anomal = X_full[y_normal_idx]

    # create data sets
    normal_dataset = TensorDataset(X_normal, torch.ones(X_normal.shape[0]))
    anomal_dataset = TensorDataset(X_anomal, torch.zeros(X_anomal.shape[0]))
    normal_train, normal_test = random_split(normal_dataset, [0.80, 0.20])
    anomal_dataset = ConcatDataset([anomal_dataset, normal_test])
    anomal_valid, anomal_test = random_split(anomal_dataset, [0.25, 0.75])

    # create data loaders
    normal_train_loader = DataLoader(normal_train, shuffle=True, drop_last=True, batch_size=batch_size)
    anomal_valid_loader = DataLoader(anomal_valid, batch_size=batch_size)
    anomal_test_loader = DataLoader(anomal_test, batch_size=batch_size)

    return normal_train_loader, anomal_valid_loader, anomal_test_loader

def compute_auprc(D, data_loader, device):
    D.eval()
    with torch.no_grad():
        y_prob_pred, y_true = np.array([]), np.array([])
        for data, targets in data_loader:
            data = data.to(device)
            p = D(data).cpu()
            y_prob_pred = np.append(y_prob_pred, p)
            y_true = np.append(y_true, targets)
    precision, recall, _ = precision_recall_curve(y_true, y_prob_pred)
    auprc = auc(recall, precision)
    return auprc