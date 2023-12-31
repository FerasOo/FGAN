import matplotlib.pyplot as plt
import numpy as np
import torch


def generate_triangular_distribution(n):
    r1, r2 = np.random.rand(n), np.random.rand(n)
    sqrt_r1 = np.sqrt(r1)
    barycentric = (1 - sqrt_r1, sqrt_r1 * (1 - r2), r2 * sqrt_r1)
    points = np.array([[10], [10]]) * barycentric[0] + np.array([[30], [10]]) * barycentric[1] + np.array(
        [[20], [30]]) * barycentric[2]
    return torch.tensor(points.T, dtype=torch.float32)


def generate_bow_shape_distribution(n):
    theta = np.linspace(0, np.pi, n)
    x = 10 * np.cos(theta) + np.random.normal(0, 1.8, size=n) + 20
    y = 15 * np.sin(theta) + np.random.normal(0, 1.8, size=n) + 10
    return torch.tensor(np.vstack((x, y)).T, dtype=torch.float32)


def generate_oval_distribution(n):
    angles = np.random.uniform(0, 2 * np.pi, n)
    radii = np.sqrt(np.random.uniform(0, 1, n))
    x = 10 * radii * np.cos(angles) + 20
    y = 5 * radii * np.sin(angles) + 20
    return torch.tensor(np.vstack((x, y)).T, dtype=torch.float32)


def generate_square_distribution(n):
    x = np.random.uniform(15, 25, n)
    y = np.random.uniform(15, 25, n)
    return torch.tensor(np.vstack((x, y)).T, dtype=torch.float32)


def get_data(distribution, n):
    if distribution == 'noise':
        return torch.normal(0, 1, size=(n, 2))

    if distribution == 'normal':
        return torch.normal(20, 3, size=(n, 2))

    if distribution == 'square':
        return generate_square_distribution(n)

    if distribution == 'triangular':
        return generate_triangular_distribution(n)

    if distribution == 'bow_shaped':
        return generate_bow_shape_distribution(n)

    if distribution == 'oval':
        return generate_oval_distribution(n)

def plot_boundary(G, D, args, title=''):
    # color background
    plt.figure()
    x1s = np.linspace(0, 40, 40)
    x2s = np.linspace(0, 40, 40)
    x1, x2 = np.meshgrid(x1s, x2s)
    Input = np.column_stack([x1.ravel(), x2.ravel()])
    with torch.no_grad():
        Output = D(torch.tensor(Input, device=args.device, dtype=torch.float32)).cpu().numpy()
    Z = Output.reshape(40, 40)
    ticks = np.arange(0, 1.01, 0.2)
    plt.contourf(x1, x2, Z, ticks, cmap='coolwarm', alpha=0.6, vmin=0, vmax=1)
    plt.colorbar()

    X_real = get_data(args.distribution, 500).numpy()

    X_gen = get_data('noise', 200).to(args.device)
    with torch.no_grad():
        Out = G(X_gen).cpu().numpy()
    plt.scatter(X_real[:, 0], X_real[:, 1], color='red', alpha=0.8, label='real')
    plt.scatter(Out[:, 0], Out[:, 1], color='blue', alpha=0.8, label='generated')
    plt.xlim(0, 40)
    plt.ylim(0, 40)
    plt.title(title)
    plt.legend()
    plt.savefig("{}/{}.png".format(args.pictures_dir, title), dpi=600)