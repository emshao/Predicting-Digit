from data_parser import *
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt


digit = 2

# get all the training data from a digit
digit_data = get_digit_data(digit, mfcc=13, test=False)


# create 10 folds from training data
total_analysis_windows = len(digit_data)
fold_size = total_analysis_windows // 10

separate_data = []
separate_data.append(digit_data[0 : 8*fold_size])
separate_data.append(digit_data[8*fold_size : 10*fold_size])


# data0 = get_digit_data(0)[0:4668]
# data1 = get_digit_data(1)[0:4668]
# data2 = get_digit_data(2)[0:4668]
# data3 = get_digit_data(3)[0:4668]
# data4 = get_digit_data(4)[0:4668]
# data5 = get_digit_data(5)[0:4668]
# data6 = get_digit_data(6)[0:4668]
# data7 = get_digit_data(7)[0:4668]
# data8 = get_digit_data(8)[0:4668]
# data9 = get_digit_data(9)[0:4668]

# test_data = [data0, data1, data2, data3, data4, data5, data6, data7, data8, data9]


# get initial cluster information
# phonemes = cfg.data_dims['phonemes'][digit]
# min_clusters = phonemes
# max_clusters = 2 * phonemes - 1


# # apply kmeans with different clusters
# for c in range(min_clusters, max_clusters + 1):

# get initial center points from KMeans
kmeans = KMeans(n_clusters=4, random_state=0, n_init='auto')
kmeans.fit(separate_data[0])
centers = kmeans.cluster_centers_

# put initial centers into EM model
gmm = GaussianMixture(n_components=c, means_init=centers, covariance_type='full')
gmm.fit(separate_data[0])  


plt.figure(figsize=(8, 6))
for i in range(10):
    log_lh = gmm.score_samples(test_data[i])
    log_lh_reshaped = np.array(log_lh).reshape(-1, 1)

    kde = KernelDensity(bandwidth=4) # graph of densities
    kde.fit(log_lh_reshaped)


    x_values = np.linspace(min(log_lh_reshaped), max(log_lh_reshaped), 1000)
    pdf = kde.score_samples(x_values)
    y_val = np.exp(pdf)

    plt.plot(x_values, y_val, lw=2)


plt.title(f"Outputs for {c} Clusters")
plt.legend([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.show()

# score = np.exp(score)

# average = sum(score) / len(score)

# print(c, " => ", average)

# print(c, np.exp(score))



# function to calculate likelihoods (not built in functions), find likelihood for each frame
# log-likelihood of each frame, add them up = log-likelihood of one utterance