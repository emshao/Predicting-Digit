import csv
import numpy as np
import configurations as cfg
import matplotlib.pyplot as plt


def plot_MFCC_lines(digit, mfcc):
    path = cfg.data_store['digit'] + f'{digit}' + ".csv"
    data = []

    with open(path,'r') as csv_file:
        file_info = csv.reader(csv_file)

        for row in file_info:
            if not len(row):
                break
            else:
                float_row = [float(e) for e in row]
                data.append(float_row[mfcc-1])

    data = np.array(data)

    plt.plot(np.arange(len(data)), data) #np.abs(data))
    # plt.title(f'Absolute MFCC Magnitudes per Analysis Window for Digit {digit}')
    plt.title(f'MFCCs per Analysis Window for Digit {digit}')
    plt.xlabel("Analysis Window")
    plt.ylabel("Value of MFCC")
    plt.xlim(-2, 45)
    # plt.ylim(-0.5, 10)


def plot_MFCC_cluster(digit, mfcc):
    path = cfg.data_store['mfcc'] + f'{digit}' + ".csv"
    data = []
    counter = 1

    with open(path,'r') as csv_file:
        file_info = csv.reader(csv_file)

        for row in file_info:
            if counter==mfcc:
                float_row = [float(n) for n in row]
                data.append(float_row)
                break
            counter += 1
    
    plt.hist(data)
    plt.title(f'Sample MFCC Histogram for Digit {digit}')
    plt.xlim(-15, 15)
    plt.ylim(0, 16000)


def plot_2MFCC_scatter(digit, mfcc1, mfcc2):
    path = cfg.data_store['digit'] + f'{digit}' + ".csv"
    data_x = []
    data_y = []

    with open(path,'r') as csv_file:
        file_info = csv.reader(csv_file)

        for row in file_info:
            if len(row):
                float_row = [float(e) for e in row]
                data_x.append(float_row[mfcc1-1])
                data_y.append(float_row[mfcc2-1])

    plt.scatter(data_x, data_y)
    plt.title(f'Pair-Wise Scatter of MFCC {mfcc2} vs MFCC {mfcc1} for Digit {digit}')
    plt.xlabel(f'MFCC {mfcc1}')
    plt.ylabel(f'MFCC {mfcc2}')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)

def plot_3MFCC_3D(digit, mfcc1, mfcc2, mfcc3):
    path = cfg.data_store['digit'] + f'{digit}' + ".csv"
    data_x = []
    data_y = []
    data_z = []

    with open(path,'r') as csv_file:
        file_info = csv.reader(csv_file)

        for row in file_info:
            if len(row):
                float_row = [float(e) for e in row]
                data_x.append(float_row[mfcc1-1])
                data_y.append(float_row[mfcc2-1])
                data_z.append(float_row[mfcc3-1])

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(data_x, data_y, data_z)
    plt.xlabel(f'MFCC {mfcc1}')
    plt.ylabel(f'MFCC {mfcc2}')
    plt.gca().set_zlabel(f'MFCC {mfcc3}')

def save_plot(save_path):
    complete_path = cfg.data_save["figures"] + save_path
    plt.savefig(complete_path)


if __name__ == "__main__":
    for d in range(1):
        plt.figure()

        for i in range(1, 13):
            plot_MFCC_lines(d, i)
            # plot_MFCC_cluster(d, i)
            # plot_2MFCC_scatter(d, i, i+1)
            # plot_3MFCC_3D(d, i, i+1, i+2)
 
        plt.grid(True)
        plt.legend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

        # save_path = cfg.data_save['mfcc_lines'] + f'{d}.png'
        # save_path = cfg.data_save['mfcc_histograms'] + f'{d}.png'
        # save_path = cfg.data_save['mfcc_scatters'] + f'{d}.png'

        # save_plot(save_path)

    plt.show()





