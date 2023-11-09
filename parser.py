import numpy as np
import matplotlib.pyplot as plt


train_data = "..Data\\Train_Arabic_Digit.txt"
test_data = "..Data\\Test_Arabic_Digit.txt"


def parse(data_file, speaker=""):

    curFile = open(data_file, 'r')

    start = curFile.readline()
    line = curFile.readline()

    curData = []
    curDigit = []

    while line:

        if line.isspace() or line == "\n":
            curDigitInfo = np.asarray(curDigit)
            curDigitInfo = np.reshape(curDigitInfo, (-1, 13))
            curData.append(curDigitInfo)
            curDigit = []

        else:
            mfccVals = np.asarray(list(map(float, line.split(' '))))
            curDigit.append(mfccVals)
        
        line = curFile.readline()

    if curDigit:
        curDigitInfo = np.asarray(curDigit)
        curDigitInfo = np.reshape(curDigitInfo, (-1, 13))
        curData.append(curDigitInfo)
        curDigit = []
    
    return curData

def parse_train_data():
    return parse(train_data)

def parse_test_data():
    return parse(test_data)

def get_mfcc(dataset, digit, coef):
    start = digit * (len(dataset)//10)
    end = start + (len(dataset)//10)

    mfcc = []

    for i in range(start, end, 1):
        data13 = np.transpose(dataset[i])
        forCoef = data13[coef-1].tolist()
        mfcc += forCoef

    return mfcc








if __name__ == "__main__":
    training = parse_train_data()
    # testing = parse_test_data()
    mfcc01 = get_mfcc(training, 0, 1)
    mfcc02 = get_mfcc(training, 0, 2)
    mfcc03 = get_mfcc(training, 0, 3)

    plt.scatter(mfcc01, mfcc03, alpha=0.5)
    plt.show()

