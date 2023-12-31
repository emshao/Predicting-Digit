import configurations as cfg
import os.path as pth
import csv
import numpy as np


train_data = cfg.data_paths["train"]
test_data = cfg.data_paths["test"]
digit_data_folder = cfg.data_paths["digits"]
MFCC_data_folder = cfg.data_paths["mfccs"]
test_data_folder = cfg.data_paths["test-digits"]

def get_digit_data(digit, mfcc=13, test=False):
    csv_path = digit_data_folder + f'digit_{digit}.csv'
    if test:
        csv_path = test_data_folder + f'digit_{digit}_test.csv'
    matrix = []

    with open(csv_path, 'r') as open_file:
        for line in open_file:
            if line !='\n':
                matrix.append([float(e) for e in line.split(',')[0:mfcc]])

    matrix = np.vstack(matrix)
    return matrix


def count_analysis_windows(digit, mfcc=13, test=False):
    csv_path = digit_data_folder + f'digit_{digit}.csv'
    if test:
        csv_path = test_data_folder + f'digit_{digit}_test.csv'
    
    count = []
    cur_count = 0

    with open(csv_path, 'r') as open_file:
        for line in open_file:
            if line != '\n':
                cur_count += 1
            else:
                count.append(cur_count)
                cur_count = 0
    
    print(len(count))
    print(count)
    print(sum(count)/len(count))




def get_analysis_windows(data_file, save_path, test=False):
    # make CSV files for all digits

    blocks_per_digit = cfg.data_dims["blocks"]
    if test:
        blocks_per_digit = cfg.data_dims["blocks-test"]
    block_counter = 0

    current_digit = 0
    save_file = None
    first_line = True

    with open(data_file, 'r') as open_file:
        for line in open_file:
            if not first_line:
                if save_file is None:
                    save_file = save_path + f'digit_{current_digit}.csv'
                    if test:
                        save_file = save_path + f'{current_digit}_test.csv'
                    open_csv = open(save_file, 'w', newline='')
                    csv_writer = csv.writer(open_csv)
                
                if line.isspace() and (block_counter+1 >= blocks_per_digit):
                    current_digit += 1
                    save_file = None
                    block_counter = 0

                else:
                    if line.isspace():
                        block_counter += 1
                        csv_writer.writerow('')
                    else: 
                        csv_writer.writerow(line.strip().split(' '))
            else:
                first_line = False

    print("Finished creating all digit CSV files\n")



def get_MFCCs(data_file, save_path, test=False):
    matrix = np.zeros(13).reshape(-1, 1)

    blocks_per_digit = cfg.data_dims["blocks"]
    block_counter = -1

    current_digit = 0
    save_file = save_path + f'MFCC_digit_{current_digit}.csv'
    if test:
        save_file = save_path + f'MFCC_digit_{current_digit}-test.csv'

    with open(data_file, 'r') as open_file:
        for line in open_file:

            if line.isspace() and (block_counter+1 >= blocks_per_digit):
                open_csv = open(save_file, 'w', newline='')
                csv_writer = csv.writer(open_csv)
                for row in matrix:
                    csv_writer.writerow(row)
                

                current_digit += 1
                block_counter = 0
                matrix = np.zeros(13).reshape(-1, 1)
                save_file = save_path + f'MFCC_digit_{current_digit}.csv'
                if test:
                    save_file = save_path + f'MFCC_digit_{current_digit}-test.csv'

            else:
                if line.isspace():
                    block_counter += 1
                else:
                    str_line = line.strip().split(' ')
                    values = np.array([float(n) for n in str_line])
                    matrix = np.hstack((matrix, values.reshape(-1, 1)))
    
    open_csv = open(save_file, 'w', newline='')
    csv_writer = csv.writer(open_csv)
    for row in matrix:
        csv_writer.writerow(row)

    print("Finished creating all MFCC files")


def check_size():
    check = cfg.data_store["mfcc"] + "0_test.csv"
    
    open_csv = open(check, 'r', newline='')
    reader = csv.reader(open_csv)
    for row in reader:
        print(len(row))



if __name__ == "__main__":

    # data_path_name = cfg.data_store["train-digit"] + "0_test.csv"
    # # mfcc_path_name = cfg.data_store["mfcc"] + "0-test.csv"

    # if not pth.isfile(data_path_name):
    #     get_analysis_windows(test_data, cfg.data_store["train-digit"], test=True)
    
    # # if not pth.isfile(mfcc_path_name):
    #     get_MFCCs(train_data)

    # check_size()



    count_analysis_windows(0)