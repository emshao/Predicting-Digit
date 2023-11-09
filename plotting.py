from parser import *
from kmeans import *
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def save_plots(data_object, digit):
    mfcc1 = get_mfcc(data_object, digit, 1)
    mfcc2 = get_mfcc(data_object, digit, 2)
    mfcc3 = get_mfcc(data_object, digit, 3)

    data12, means12, gmm12 = apply_kmeans(mfcc1, mfcc2)
    data23, means23, gmm23 = apply_kmeans(mfcc2, mfcc3)
    data13, means13, gmm13 = apply_kmeans(mfcc1, mfcc3)

    plot_contours(digit, 1, 2, data12, means12, gmm12, 1)
    plot_contours(digit, 2, 3, data23, means23, gmm23, 2)
    plot_contours(digit, 1, 3, data13, means13, gmm13, 3)
    
    # plt.show()
    plt.savefig(f'..\\Results\\gmm_{digit}.png')


if __name__ == "__main__":
    
    training = parse_train_data()

    for digit in range(10):
        save_plots(training, digit)


