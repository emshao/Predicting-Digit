from data_parser import *
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from sklearn import metrics
import random
from matplotlib.ticker import MaxNLocator
from scipy.stats import multivariate_normal


# method to get all the training data for each digit
# results return in matrix form (n_observations, 13)
# input     digit
# returns   matrix
def train_data(digit):
    path_to_data = f"Data/Digit_Data/digit_{digit}.csv"

    matrix = []

    with open(path_to_data, 'r') as open_file:
        for line in open_file:
            if line !='\n':
                matrix.append([float(e) for e in line.split(',')])

    data = np.array(matrix)

    return data


# method to train models through KMeans and EM with certain cluster size
# input     data, cluters
# returns   model
def train_model(data, clusters, km=True):
    # KMeans
    kmeans = KMeans(n_clusters=clusters, random_state=0, n_init='auto')
    kmeans.fit(data)

    centers = kmeans.cluster_centers_
    label_list = kmeans.labels_

    data_by_clusters = []
    covars = []
    pis = []

    for i in range(clusters):
        indices = [n for n, val in enumerate(label_list) if val==i]
        cluster_data = []
        for ind in indices:
            demeaned_data = data[ind] - centers[i]
            cluster_data.append(demeaned_data)
        data_by_clusters.append(cluster_data)

        covariance = np.cov(cluster_data) # 13 * 13 dim data

        covars.append(covariance)

        pis.append(len(cluster_data) / len(data))
    

    return centers, covars, pis


def generate_model_Kmeans(data, pis, means, covars, digit):

    means1n2 = means[:,0:2]
    covar1n2 = []

    for cov in covars:
        first_two = cov[0][0:2]
        second = cov[1][0:2]
        covar1n2.append([first_two, second])

    
    data1n2 = [n[0:2] for n in data]


    print(np.array(data1n2).shape)
    print(data1n2[0][1])
    x_val = [n[0] for n in data1n2]
    y_val = [n[1] for n in data1n2]

    x = np.linspace(min(x_val), max(x_val), num=150)
    y = np.linspace(min(y_val), max(y_val), num=150)
    
    X, Y = np.meshgrid(x, y)
    XX = np.dstack((X, Y))


    pdf = np.array([[[0.0] * 150] * 150])

    for i in range(len(pis)):
        rv = multivariate_normal(means1n2[i], covar1n2[i], seed=0)
        pdf = np.add(pdf, (pis[i] * np.array(rv.pdf(XX))))


    plt.figure()
    plt.contourf(X, Y, pdf[0], alpha=1)

    plt.scatter(x_val, y_val, s=1, alpha=0.1)

    plt.scatter(means1n2[:, 0], means1n2[:, 1], s=30, c='red', alpha=0.7, marker='.')

    plt.title(f"KMeans Clusting Model Output for MFCC 1 vs MFCC 2 for Digit {digit}")
    plt.xlabel("MFCC 1")
    plt.ylabel("MFCC 2")

    plt.savefig("C:/Users/Emily Shao/Desktop/Predicting-Digit/Results/kmeans_for_{digit}.png")

# method to get all the testing data for each digit 
# results return in list form of size number_of_utterances
# each utterance has size (n_frames, 13)
# input     digit
# returns   list
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

    return utterance_list

# # NOT TRUE TO MODEL
# # method get the cumulative log densities from the testing data
# # input     test_list, model
# # returns   log_density
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

# true_val = test5

# log_density0 = get_log_densities(true_val, gmm0)
# log_density1 = get_log_densities(true_val, gmm1)
# log_density2 = get_log_densities(true_val, gmm2)
# log_density3 = get_log_densities(true_val, gmm3)
# log_density4 = get_log_densities(true_val, gmm4)
# log_density5 = get_log_densities(true_val, gmm5)
# log_density6 = get_log_densities(true_val, gmm6)
# log_density7 = get_log_densities(true_val, gmm7)
# log_density8 = get_log_densities(true_val, gmm8)
# log_density9 = get_log_densities(true_val, gmm9)

# all_log_den = [log_density0, log_density1, log_density2,
#                log_density3, log_density4, log_density5, 
#                log_density6, log_density7, log_density8, log_density9]

# print("calculated log likelihoods")

# # method to find the maximum log densitiy and makes a guess
# # input     list
# # return    guess
# def calculate_guess(log_densities):
#     mostLikelyLogs = []
    
#     for log_density in log_densities:
#         mostLikelyLogs.append(list(np.exp(log_density)).index(max(np.exp(log_density))))

#     return mostLikelyLogs.index(max(mostLikelyLogs))

# guess = calculate_guess(all_log_den)

# print("final guess: ", guess)

# method to guess based on one utterance
# input     utterance, gmms
# output    guess
def get_log_densities_per_utterance(one_utterance, gmm_list):
    log_LH = []

    for gmm in gmm_list:
        log_lh_frames = gmm.score_samples(one_utterance)
        log_lh_utterance = np.sum(log_lh_frames) # just like multiplying probability for all the frames
        log_LH.append(log_lh_utterance)

    return log_LH.index(max(log_LH))

# one_test0 = test0[0]
# one_test1 = test1[0]
# one_test2 = test2[0]
# one_test3 = test3[0]
# one_test4 = test4[0]
# one_test5 = test5[0]
# one_test6 = test6[0]
# one_test7 = test7[0]
# one_test8 = test8[0]
# one_test9 = test9[0]

# one_guess = get_log_densities_per_utterance(one_test3, all_gmm)

# print("final guess: ", one_guess)

def create_test_data_and_labels(all_tests, test0):
    
    tests = []
    lengths = []
    for digit in all_tests:
        lengths.append(len(digit))
        for test in digit:
            tests.append(test)

    actual_labels = []
    for i in range(10):
        actual_labels.extend([i] * lengths[i])
        
    combined_data = list(zip(tests, actual_labels))
    random.shuffle(combined_data)

    utterances, labels = zip(*combined_data)

    return utterances, labels

def run_model_tests(utternace_list, gmm_list):
    results = []

    for utterance in utternace_list:
        guess = get_log_densities_per_utterance(utterance, gmm_list)
        results.append(guess)

    return results

def create_confusion_matrix(actual, predicted, show=False):
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    
    accuracy = metrics.accuracy_score(actual, predicted)
    accuracy_per_digit = confusion_matrix.diagonal()/confusion_matrix.sum(axis=1)

    if show:
        display_labels = ['0','1','2','3','4','5','6','7','8','9']
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = display_labels)
        cm_display.plot()
        plt.title("Accuracy = " + str(accuracy))
        plt.show()

    return accuracy_per_digit
    

train0 = train_data(0)
train1 = train_data(1)
train2 = train_data(2)
train3 = train_data(3)
train4 = train_data(4)
train5 = train_data(5)
train6 = train_data(6)
train7 = train_data(7)
train8 = train_data(8)
train9 = train_data(9)

print("obtained training data")

all_train = [train0, train1, train2, train3, train4, train5, train6, train7, train8, train9]

# test0 = test_data(0)
# test1 = test_data(1)
# test2 = test_data(2)
# test3 = test_data(3)
# test4 = test_data(4)
# test5 = test_data(5)
# test6 = test_data(6)
# test7 = test_data(7)
# test8 = test_data(8)
# test9 = test_data(9)

# all_tests = [test0, test1, test2, test3, test4, test5, test6, test7, test8, test9]

# print("obtained testing data")

# # accuracy_list = []
# # useKM = False

# # for i in range(10):
# #     ten = False
# #     if (i==9):
# #         ten = True

for i in range(len(all_train)):
    useKM = True
    centers, covars, pis = train_model(all_train[i], 6, useKM)
# gmm1 = train_model(train1, 6, useKM)
# gmm2 = train_model(train2, 6, useKM)
# gmm3 = train_model(train3, 6, useKM)
# gmm4 = train_model(train4, 6, useKM)
# gmm5 = train_model(train5, 6, useKM)
# gmm6 = train_model(train6, 6, useKM)
# gmm7 = train_model(train7, 6, useKM)
# gmm8 = train_model(train8, 6, useKM)
# gmm9 = train_model(train9, 6, useKM)

# all_gmm = [gmm0, gmm1, gmm2, gmm3, gmm4, gmm5, gmm6, gmm7, gmm8, gmm9]

    print("finished training models")

    generate_model_Kmeans(all_train[i], pis, centers, covars, i)


# utterances, labels = create_test_data_and_labels(all_tests, test0)
# results = run_model_tests(utterances, all_gmm)
# #     show = False
# #     if (ten):
# #         show = True

# accuracy = create_confusion_matrix(labels, results, show=False)

# #     accuracy_list.append(accuracy)

# print(accuracy)






