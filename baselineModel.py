from data_parser import *
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt


def train_data(digit):
    path_to_data = f"Data/Digit_Data/digit_{digit}.csv"

    matrix = []

    with open(path_to_data, 'r') as open_file:
        for line in open_file:
            if line !='\n':
                matrix.append([float(e) for e in line.split(',')])

    data = np.array(matrix)

    return data

train0 = train_data(0)


def train_model(data, clusters):
    # KMeans
    kmeans = KMeans(n_clusters=clusters, random_state=0, n_init='auto')
    kmeans.fit(data)

    centers = kmeans.cluster_centers_

    # EM in GMM
    gmm = GaussianMixture(n_components=clusters, means_init=centers, covariance_type='full')
    gmm.fit(data)

    return gmm

gmm4 = train_model(train0, 4)
gmm5 = train_model(train0, 5)
gmm6 = train_model(train0, 6)
gmm7 = train_model(train0, 7)


# testing / results
def test_data(digit):
    path_to_test_data = f'Data/Test_Digit_Data/digit_{digit}_test.csv'

    utterance_list = []

    with open(path_to_test_data, 'r') as open_file:
        current_utterance = []
        for line in open_file:
            if line == '\n':
                utterance_list.append(np.array(current_utterance))
                current_utterance = []
            else:
                # append frame as a part of utterance
                current_utterance.append([float(e) for e in line.split(',')])

    print(len(utterance_list)) # 219 utterances
    return utterance_list

test0 = test_data(0)
test1 = test_data(1)
test2 = test_data(2)
test3 = test_data(3)
test4 = test_data(4)
test5 = test_data(5)
test6 = test_data(6)
test7 = test_data(7)
test8 = test_data(8)
test9 = test_data(9)


def get_log_densities(utterance_list, gmm):
    log_LH = []

    for utterance in utterance_list:
        log_lh_frames = gmm.score_samples(utterance)
        log_lh_utterance = np.sum(log_lh_frames) # just like multiplying probability for all the frames
        log_LH.append(log_lh_utterance)

    log_LH = np.array(log_LH)[:, np.newaxis]

    kde = KernelDensity(bandwidth=30).fit(log_LH)

    x_values = np.linspace(-1500, 0, 3000)

    log_densities = kde.score_samples(x_values[:, np.newaxis])

    return log_densities

gmm_r = gmm7
log_den0 = get_log_densities(test0, gmm_r)
log_den1 = get_log_densities(test1, gmm_r)
log_den2 = get_log_densities(test2, gmm_r)
log_den3 = get_log_densities(test3, gmm_r)
log_den4 = get_log_densities(test4, gmm_r)
log_den5 = get_log_densities(test5, gmm_r)
log_den6 = get_log_densities(test6, gmm_r)
log_den7 = get_log_densities(test7, gmm_r)
log_den8 = get_log_densities(test8, gmm_r)
log_den9 = get_log_densities(test9, gmm_r)

print(len(list(np.exp(log_den0))))
mostLikely0 = list(np.exp(log_den0)).index(max(np.exp(log_den0)))
print((mostLikely0//2)-1500)





x_values = np.linspace(-1500, 0, 3000)
plt.figure()

# plt.plot(x_values, log_densities)
plt.plot(x_values, np.exp(log_den0))
plt.plot(x_values, np.exp(log_den1))
plt.plot(x_values, np.exp(log_den2))
plt.plot(x_values, np.exp(log_den3))
plt.plot(x_values, np.exp(log_den4))
plt.plot(x_values, np.exp(log_den5))
plt.plot(x_values, np.exp(log_den6))
plt.plot(x_values, np.exp(log_den7))
plt.plot(x_values, np.exp(log_den8))
plt.plot(x_values, np.exp(log_den9))

plt.xlabel("Log Likelihoods")
plt.ylabel("Density")
plt.legend(['0','1','2','3','4','5','6','7','8','9'])

plt.show()






