from parser import *
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def apply_kmeans(data1, data2, clusters=3):
    data = list(zip(data1, data2))
    data = np.vstack(data)

    kmeans = KMeans(n_clusters=clusters, random_state=0, n_init=1)
    kmeans.fit(data)
    initial_means = kmeans.cluster_centers_

    gmm = GaussianMixture(n_components=clusters, covariance_type='full', means_init=initial_means)
    gmm.fit(data)

    return data, initial_means, gmm


def plot_contours(digit, first, second, data, means, gmm, ax_num):
    x = np.linspace(data[:, 0].min() - 1, data[:, 0].max() + 1, num=100)
    y = np.linspace(data[:, 1].min() - 1, data[:, 1].max() + 1, num=100)
    X, Y = np.meshgrid(x, y)
    XX = np.column_stack([X.ravel(), Y.ravel()])

    Z = -gmm.score_samples(XX)
    Z = Z.reshape(X.shape)

    plt.subplot(1, 3, ax_num)
    plt.contour(X, Y, Z, levels=100, linewidths=5, cmap='viridis', alpha=0.5)

    plt.scatter(data[:, 0], data[:, 1], s=15, alpha=0.5)

    plt.scatter(means[:, 0], means[:, 1], s=300, c='red', alpha=0.75, marker='.')

    plt.title(f'GMM of MFCC{first} vs MFCC{second} for Digit {digit}')
    plt.xlabel('X')
    plt.ylabel('Y')


if __name__ == "__main__":
    training = parse_train_data()
    digit = 0

    mfcc1 = get_mfcc(training, digit, 1)
    mfcc2 = get_mfcc(training, digit, 2)
    mfcc3 = get_mfcc(training, digit, 3)

    data12, means12, gmm12 = apply_kmeans(mfcc1, mfcc2)
    data23, means23, gmm23 = apply_kmeans(mfcc2, mfcc3)
    data13, means13, gmm13 = apply_kmeans(mfcc1, mfcc3)

    plot_contours(digit, 1, 2, data12, means12, gmm12, 1)
    plot_contours(digit, 2, 3, data23, means23, gmm23, 2)
    plot_contours(digit, 1, 3, data13, means13, gmm13, 3)
    plt.show()

