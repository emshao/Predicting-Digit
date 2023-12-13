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
def train_model(data, clusters):
    # KMeans
    kmeans = KMeans(n_clusters=clusters, random_state=0, n_init='auto')
    kmeans.fit(data)

    centers = kmeans.cluster_centers_

    # EM in GMM - random?
    gmm = GaussianMixture(n_components=clusters, means_init=centers, covariance_type='full')
    gmm.fit(data)

    return gmm

# [7, 9, 7, 7, 5, 7, 7, 7, 11, 7]



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
                current_utterance.append([float(e) for e in line.split(',')])

    return utterance_list

# # NOT TRUE TO MODEL
# # method get the cumulative log densities from the testing data
# # input     test_list, model
# # returns   log_density
# def get_log_densities(utterance_list, gmm):
#     log_LH = []

#     for utterance in utterance_list:
#         log_lh_frames = gmm.score_samples(utterance)
#         log_lh_utterance = np.sum(log_lh_frames) # just like multiplying probability for all the frames
#         log_LH.append(log_lh_utterance)

#     log_LH = np.array(log_LH)[:, np.newaxis]

#     kde = KernelDensity(bandwidth=30).fit(log_LH)
#     x_values = np.linspace(-1500, 0, 3000)
#     log_densities = kde.score_samples(x_values[:, np.newaxis])
#     return log_densities

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
    display_labels = ['0','1','2','3','4','5','6','7','8','9']
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = display_labels)
    
    accuracy = metrics.accuracy_score(actual, predicted)

    if show:
        cm_display.plot()
        plt.title("Accuracy = " + str(accuracy))
        plt.show()

    return accuracy
    



# buckets = ['four', 'five', 'six', 'seven', 'eight', 'nine']
# accuracies = [0.8799, 0.8872, 0.9041, 0.8886, 0.8826, 0.8867]

# plt.bar(buckets, accuracies)
# plt.ylim(0.5, 1.0)
# plt.xlabel("Number of Clusters")
# plt.ylabel("Accuracy")
# plt.title("Accuracy of the Model with Different Cluster Numbers")
# plt.show()

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

all_tests = [test0, test1, test2, test3, test4, test5, test6, test7, test8, test9]

print("obtained testing data")

accuracy_list = []


for i in range(5):

    temp = []
    for cluster in range(20):

        c = cluster+1
        gmm0 = train_model(train0, c)
        gmm1 = train_model(train1, c)
        gmm2 = train_model(train2, c)
        gmm3 = train_model(train3, c)
        gmm4 = train_model(train4, c)
        gmm5 = train_model(train5, c)
        gmm6 = train_model(train6, c)
        gmm7 = train_model(train7, c)
        gmm8 = train_model(train8, c)
        gmm9 = train_model(train9, c)

        all_gmm = [gmm0, gmm1, gmm2, gmm3, gmm4, gmm5, gmm6, gmm7, gmm8, gmm9]

        print("finished training models for cluster: ", cluster)

        utterances, labels = create_test_data_and_labels(all_tests, test0)
        results = run_model_tests(utterances, all_gmm)
        accuracy = create_confusion_matrix(labels, results, show=False)

        temp.append(accuracy)
    
    accuracy_list.append(temp)
    print("finished trial 1")

print(accuracy_list)

mean = np.sum(accuracy_list, axis = 0) / len(accuracy_list)

buckets = np.arange(20) + 1
plt.plot(buckets, mean)

plt.show()


# from previous runs
def plot_cluster_accuracies():
    accuracy_list = [[0.6808219178082192, 0.8328767123287671, 0.8762557077625571, 0.869406392694064, 0.8904109589041096, 0.8954337899543379, 0.8981735159817351, 0.9045662100456621, 0.8890410958904109, 0.8707762557077625, 0.8611872146118722, 0.8812785388127854, 0.8634703196347032, 0.865296803652968, 0.8721461187214612, 0.8570776255707763, 0.8570776255707763, 0.8529680365296803, 0.8406392694063927, 0.8424657534246576], [0.6808219178082192, 0.8333333333333334, 0.8908675799086758, 0.8844748858447489, 0.8872146118721461, 0.8876712328767123, 0.8794520547945206, 0.8776255707762557, 0.8831050228310502, 0.8876712328767123, 0.8639269406392694, 0.8643835616438356, 0.8703196347031964, 0.869406392694064, 0.8680365296803653, 0.8648401826484018, 0.8575342465753425, 0.8602739726027397, 0.8534246575342466, 0.8589041095890411], [0.6808219178082192, 0.8328767123287671, 0.8735159817351598, 0.8785388127853881, 0.8885844748858448, 0.908675799086758, 0.8853881278538813, 0.8794520547945206, 0.8840182648401826, 0.8785388127853881, 0.8703196347031964, 0.8557077625570776, 0.85662100456621, 0.8598173515981735, 0.8639269406392694, 0.8616438356164383, 0.8534246575342466, 0.8529680365296803, 0.8488584474885845, 0.8689497716894977], [0.6808219178082192, 0.8812785388127854, 0.8767123287671232, 0.8872146118721461, 0.8844748858447489, 0.9045662100456621, 0.8794520547945206, 0.8817351598173516, 0.8844748858447489, 0.8867579908675799, 0.8744292237442922, 0.8753424657534247, 0.8744292237442922, 0.8776255707762557, 0.8721461187214612, 0.8593607305936073, 0.8776255707762557, 0.8547945205479452, 0.845662100456621, 0.8666666666666667], [0.6808219178082192, 0.8337899543378996, 0.8789954337899544, 0.8812785388127854, 0.8940639269406393, 0.8968036529680365, 0.8858447488584474, 0.8785388127853881, 0.871689497716895, 0.8771689497716895, 0.8785388127853881, 0.8625570776255708, 0.8776255707762557, 0.8794520547945206, 0.8684931506849315, 0.8616438356164383, 0.867579908675799, 0.8643835616438356, 0.8575342465753425, 0.8534246575342466]]

    means = np.sum(accuracy_list, axis=0) / len(accuracy_list)
    print((means))
    print(np.arange(20)+1)

    x_val = []

    for i in range(5):
        for j in range(20):
            x_val.append(j+1)

    print(len(x_val))

    fig = plt.figure(figsize=(8, 5))
    ax = fig.gca()
    pts = []

    for list in accuracy_list:
        for i in list:
            pts.append(i)

    print(len(pts))

    plt.scatter(x_val, pts)
    plt.plot(np.arange(20)+1, means, marker='o', c='orange')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Number of Clusters in Model")
    plt.ylabel("Overall Model Accuracy")
    plt.title("Accuracy of Model with Different Cluster Numbers")
    plt.show()