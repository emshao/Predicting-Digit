import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import configurations as cfg
from data_parser import *
import matplotlib.pyplot as plt

def apply_kmeans(digit, data):
    clusters = cfg.data_dims['phenomns'][digit]

    kmeans = KMeans(n_clusters=clusters, random_state=0, n_init='auto')

    kmeans.fit(data)
    initial_means = kmeans.cluster_centers_

    print("finished training")
    return initial_means, kmeans.labels_






if __name__ == "__main__":
    digit = 0
    matrix = get_digit_data(digit)
    means, labels = apply_kmeans(digit, matrix)

    clusters = cfg.data_dims['phenomns'][digit]

    # shouldn't I just always use a full matrix?
    gmm = GaussianMixture(n_components=clusters, means_init=means, covariance_type='full')
    gmm.fit(matrix)   


    digit0 = get_digit_data(0, mfcc=13, test=True)
    digit1 = get_digit_data(1, mfcc=13, test=True)
    digit2 = get_digit_data(2, mfcc=13, test=True)
    digit3 = get_digit_data(3, mfcc=13, test=True)
    digit4 = get_digit_data(4, mfcc=13, test=True)
    digit5 = get_digit_data(5, mfcc=13, test=True)
    digit6 = get_digit_data(6, mfcc=13, test=True)
    digit7 = get_digit_data(7, mfcc=13, test=True)
    digit8 = get_digit_data(8, mfcc=13, test=True)
    digit9 = get_digit_data(9, mfcc=13, test=True)

    datas = [digit0, digit1, digit2, digit3, digit4, digit5, digit6, digit7, digit8, digit9]

    loglikelihoods = []
    for i in range(10):
        loglikelihoods.append(gmm.score(datas[i]))
    
    likelihoods = np.exp(loglikelihoods)

    plt.figure()
    plt.bar(np.arange(10), likelihoods) #, marker='o')
    plt.show()


    # samples, _ = gmm.sample(n_samples=100)
    # print(samples)

    # sample_data_0 = []
    # sample_data_1 = []

    # for row in samples:
    #     sample_data_0.append(row[0])
    #     sample_data_1.append(row[1])

    # plt.scatter(sample_data_0, sample_data_1)
    # plt.title(f'Pair-Wise Scatter of MFCC 0 vs MFCC 1 for Digit {digit}')
    # plt.xlabel(f'MFCC 0')
    # plt.ylabel(f'MFCC 1')
    # plt.xlim(-15, 15)
    # plt.ylim(-15, 15)
    # plt.show()

    # mfcc1 = [x[0] for x in matrix]
    # mfcc2 = [x[1] for x in matrix]
    # mfcc3 = [x[2] for x in matrix]

    # # print(len(mfcc1))

    # plt.figure()
    # # plt.scatter(mfcc2, mfcc3, c=labels)


    # ax = plt.axes(projection='3d')
    # ax.scatter3D(mfcc1, mfcc2, mfcc3, c=labels)
    # ax.set_zlabel('MFCC 3')
    # ax.set_zlim(-10, 10)
    

    # plt.xlabel(f'MFCC 2')
    # plt.ylabel(f'MFCC 3')
    # plt.xlim(-15, 15)
    # plt.ylim(-15, 15)

    
    
    plt.show()
