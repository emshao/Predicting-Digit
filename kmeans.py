import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import configurations as cfg
from data_parser import *

def apply_kmeans(digit):
    clusters = cfg.data_dims['phenomns'][digit]
    kmeans = KMeans(n_clusters=clusters, random_state=0, n_init=1)

    matrix = get_digit_data(digit)
    kmeans.fit(matrix)
    initial_means = kmeans.cluster_centers_

    print(initial_means)
    print("finished training")

if __name__ == "__main__":
    apply_kmeans(0)


