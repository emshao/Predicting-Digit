from data_parser import *
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from sklearn import metrics
import random
from matplotlib.ticker import MaxNLocator


# method to get all the training data for each digit
# results return in matrix form (n_observations, 13)
# input     digit
# returns   matrix
def train_data(digit, mfcc=13):
    path_to_data = f"Data/Digit_Data/digit_{digit}.csv"

    matrix = []

    with open(path_to_data, 'r') as open_file:
        for line in open_file:
            if line !='\n':
                matrix.append([float(e) for e in line.split(',')][0:mfcc])

    data = np.array(matrix)

    return data


# method to train models through KMeans and EM with certain cluster size
# input     data, cluters
# returns   model
def train_model(data, clusters, km=True):

    if (km):
        # KMeans
        kmeans = KMeans(n_clusters=clusters, random_state=0, n_init='auto')
        kmeans.fit(data)

        centers = kmeans.cluster_centers_

        # EM in GMM
        gmm = GaussianMixture(n_components=clusters, means_init=centers, covariance_type='full')
    else:
        gmm = GaussianMixture(n_components=clusters, init_params='random', covariance_type='full')
    
    gmm.fit(data)

    return gmm


# method to get all the testing data for each digit 
# results return in list form of size number_of_utterances
# each utterance has size (n_frames, 13)
# input     digit
# returns   list
def test_data(digit, mfcc=13):
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
                current_utterance.append([float(e) for e in line.split(',')][0:mfcc])

    return utterance_list



def get_log_densities_per_utterance(one_utterance, gmm_list):
    log_LH = []

    for gmm in gmm_list:
        log_lh_frames = gmm.score_samples(one_utterance)
        log_lh_utterance = np.sum(log_lh_frames) # just like multiplying probability for all the frames
        log_LH.append(log_lh_utterance)

    return log_LH.index(max(log_LH))


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

def create_confusion_matrix(actual, predicted, mfcc, show=False):
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    
    accuracy = metrics.accuracy_score(actual, predicted)
    accuracy_per_digit = confusion_matrix.diagonal()/confusion_matrix.sum(axis=1)

    if show:
        display_labels = ['0','1','2','3','4','5','6','7','8','9']
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = display_labels)
        cm_display.plot()
        # plt.title("Accuracy = " + str(accuracy))
        plt.title(f"Confusion Matrix for {mfcc} MFCC")
        # plt.show()
        plt.savefig(f"C:\\Users\\Emily Shao\\Desktop\\Predicting-Digit\\Results\\mfcc_{mfcc}.png")

    return accuracy_per_digit


mfcc_acc = []
  
for i in range(13):
    mfcc = i+1

    train0 = train_data(0, mfcc)
    train1 = train_data(1, mfcc)
    train2 = train_data(2, mfcc)
    train3 = train_data(3, mfcc)
    train4 = train_data(4, mfcc)
    train5 = train_data(5, mfcc)
    train6 = train_data(6, mfcc)
    train7 = train_data(7, mfcc)
    train8 = train_data(8, mfcc)
    train9 = train_data(9, mfcc)

    print("obtained training data")

    test0 = test_data(0, mfcc)
    test1 = test_data(1, mfcc)
    test2 = test_data(2, mfcc)
    test3 = test_data(3, mfcc)
    test4 = test_data(4, mfcc)
    test5 = test_data(5, mfcc)
    test6 = test_data(6, mfcc)
    test7 = test_data(7, mfcc)
    test8 = test_data(8, mfcc)
    test9 = test_data(9, mfcc)

    all_tests = [test0, test1, test2, test3, test4, test5, test6, test7, test8, test9]

    print("obtained testing data")

    # accuracy_list = []
    # useKM = False

    # for i in range(10):
    #     ten = False
    #     if (i==9):
    #         ten = True

    useKM = False
    gmm0 = train_model(train0, 6, useKM)
    gmm1 = train_model(train1, 6, useKM)
    gmm2 = train_model(train2, 6, useKM)
    gmm3 = train_model(train3, 6, useKM)
    gmm4 = train_model(train4, 6, useKM)
    gmm5 = train_model(train5, 6, useKM)
    gmm6 = train_model(train6, 6, useKM)
    gmm7 = train_model(train7, 6, useKM)
    gmm8 = train_model(train8, 6, useKM)
    gmm9 = train_model(train9, 6, useKM)

    all_gmm = [gmm0, gmm1, gmm2, gmm3, gmm4, gmm5, gmm6, gmm7, gmm8, gmm9]

    print("finished training models")

    utterances, labels = create_test_data_and_labels(all_tests, test0)
    results = run_model_tests(utterances, all_gmm)
    #     show = False
    #     if (ten):
    #         show = True

    accuracy = create_confusion_matrix(labels, results, mfcc, show=True)

    #     accuracy_list.append(accuracy)

    # print(accuracy)
    mfcc_acc.append((np.mean(accuracy)))


print(mfcc_acc)


def show_avg_mfcc_accuracies():
    mfcc_accuracies = [0.2954337899543379, 0.4283105022831051, 0.4698630136986301, 0.5109589041095891, 0.5589041095890411, 0.6328767123287671, 0.7374429223744292, 0.8671232876712329, 0.8735159817351598, 0.8789954337899543, 0.8949771689497718, 0.8945205479452054, 0.8863013698630136]

    x = np.arange(13)+1

    plt.figure()
    plt.plot(x, mfcc_accuracies, marker='o')
    plt.ylim(0, 1)
    plt.title("Average Model Accuracies Based on Number of MFCCs Used")
    plt.xlabel("Number of MFCCs Used (in Order)")
    plt.ylabel("Accuracy (%)")
    plt.show()

