from data_parser import *
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn import metrics
import random
import numpy as np







def train_data(digit, mfcc=13):
    path_to_data = f"Data/Digit_Data/digit_{digit}.csv"

    matrix = []

    with open(path_to_data, 'r') as open_file:
        for line in open_file:
            if line !='\n':
                matrix.append([float(e) for e in line.split(',')][0:mfcc])

    data = np.array(matrix)

    return data

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


# clusters = 6, MFCC = 8, covariance = Full

def return_model(data, clusters):

    gmm = GaussianMixture(n_components=clusters, covariance_type='full')  
    gmm.fit(data)

    return gmm



def shuffle_all_test_data_and_labels(all_tests):
    
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

def get_log_densities_per_utterance(one_utterance, gmm_list):
    log_LH = []

    for gmm in gmm_list:
        log_lh_frames = gmm.score_samples(one_utterance)
        log_lh_utterance = np.sum(log_lh_frames) # just like multiplying probability for all the frames
        log_LH.append(log_lh_utterance * len(one_utterance))

    return log_LH.index(max(log_LH))

def run_model_tests(utternace_list, gmm_list):
    results = []

    for utterance in utternace_list:
        guess = get_log_densities_per_utterance(utterance, gmm_list)
        results.append(guess)

    return results


# function to add value labels
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])


def create_confusion_matrix(actual, predicted, mfcc, trial, show=False):
    print(f"For {mfcc} MFCCs")
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    
    accuracy = metrics.accuracy_score(actual, predicted)
    accuracy_per_digit = confusion_matrix.diagonal()/confusion_matrix.sum(axis=1)

    accurRound = [round(i, 2) for i in accuracy_per_digit]

    if show:
        display_labels = ['0','1','2','3','4','5','6','7','8','9']
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = display_labels)
        cm_display.plot()
        plt.title(f"Confusion Matrix for {mfcc} MFCC")

        # display_label_2 = ['0', '2', '3', '7']
        # acc = [accuracy_per_digit[0], accuracy_per_digit[2], accuracy_per_digit[3], accuracy_per_digit[7]]

        plt.figure()
        plt.bar(display_labels, accuracy_per_digit)
        addlabels(display_labels, accurRound)
        plt.title(f"Accuracy Per Digit with Varied Components")
        plt.xlabel("Digit")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 1)

        plt.show()
        # plt.savefig(f"C:\\Users\\Emily Shao\\Desktop\\Predicting-Digit\\Results\\mfcc_trial_{trial}.png")

    return accuracy_per_digit






# for i in range(8, 11):
trial = 16

mfcc = 13
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
print("")

gmm0 = return_model(train0, 6) #9, 6c = 97
gmm1 = return_model(train1, 6) #10 6c = 96
gmm2 = return_model(train2, 5) #8, 4c = 80 ----> 10, 6c
gmm3 = return_model(train3, 8) #9, 4c = -------> 9, 7c
gmm4 = return_model(train4, 6) #7, 6c = 
gmm5 = return_model(train5, 6) #7, 6c = 
gmm6 = return_model(train6, 5) #10, 6c = 
gmm7 = return_model(train7, 8) #8, 7c = 78
gmm8 = return_model(train8, 6) #10, 6c = 90
gmm9 = return_model(train9, 6) #10, 6c = 91

all_gmm = [gmm0, gmm1, gmm2, gmm3, gmm4, gmm5, gmm6, gmm7, gmm8, gmm9]

print("finished training models")

utterances, true_labels = shuffle_all_test_data_and_labels(all_tests)

guess_labels = run_model_tests(utterances, all_gmm)

accur = (create_confusion_matrix(true_labels, guess_labels, mfcc, trial, True))

print(accur)
print(np.sum(accur) / len(accur))



# accur = [0.94977169, 0.96347032, 0.80365297, 0.89954338, 0.91780822, 0.90410959, 0.9543379,  0.79452055, 0.89041096, 0.93150685]

# print(np.sum(accur) / len(accur))
