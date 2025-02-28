# 设置项目的默认路径
import os
import sys
from email.policy import default

script_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
project_path = os.path.abspath(os.path.join(script_path, '..'))  # 获取项目路径
sys.path.append(project_path)  # 添加路径到系统路径中

default_dataset_path = os.path.abspath(os.path.join(project_path, 'PreTrainingDataset'))  # 获取默认数据集路径

################################################# 以下是代码的正是部分 #####################################################

import numpy as np
from scipy import signal

# 可视化模块
import matplotlib.pyplot as plt

###################################################### 分割线 ###########################################################

alternative_dataset_dir_dict = {'SYSU': 'F:/PycharmProjects/MyoArmbandDataset/PreTrainingDataset'}

###################################################### 分割线 ###########################################################

number_of_vector_per_example = 52
number_of_canals = 8  # canal 即 channel
number_of_classes = 7
size_non_overlap = 5

def format_data_to_train(vector_to_format: np.ndarray) -> np.ndarray:
    dataset_example_formatted = []
    # 最新版本的python不支持不同维度的数组/列表直接的 bool 逻辑判断，所以用None替代[]
    # 否则会报 'operands could not be broadcast together with shapes (8,52) (0,)' 之类的错
    example = None  # 初始化example变量
    emg_vector = []
    for value in vector_to_format:

        emg_vector.append(value)
        if (len(emg_vector) >= 8):
            if example is None:
                example = emg_vector  # 初始化example变量
            else:
                example = np.vstack((example, emg_vector))

            emg_vector = []
            if (len(example) >= number_of_vector_per_example):
                example = example.transpose()
                dataset_example_formatted.append(example)
                example = example.transpose()
                example = example[size_non_overlap:]
    # Apply the butterworth high pass filter at 2Hz
    dataset_high_pass_filtered = []
    for example in dataset_example_formatted:
        example_filtered = []
        for channel_example in example:
            example_filtered.append(butter_highpass_filter(channel_example, 2, 200))
        dataset_high_pass_filtered.append([example_filtered])
    return np.array(dataset_high_pass_filtered)

def butter_highpass(cutoff, fs, order=3):
    nyq = .5*fs
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=3):
    b, a = butter_highpass(cutoff=cutoff, fs=fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def shift_electrodes(examples: list, labels: list) -> tuple[list, list]:
    index_normal_class = [1, 2, 6, 2]  # The normal activation of the electrodes.
    class_mean = []
    # For the classes that are relatively invariant to the highest canals activation, we get on average for a
    # subject the most active canals for those classes
    for classe in range(3, 7):  # classe =  3, 4, 5, 6
        X_example = []
        Y_example = []
        for k in range(len(examples)):
            X_example.extend(examples[k])
            Y_example.extend(labels[k])

        cwt_add = None  # 初始化cwt_add变量（最新版本的python不支持不同维度的数组/列表直接的 bool 逻辑判断，所以用None替代[]）
        for j in range(len(X_example)):
            if Y_example[j] == classe:
                if cwt_add is None:
                    cwt_add = np.array(X_example[j][0])
                else:
                    cwt_add += np.array(X_example[j][0])
        class_mean.append(np.argmax(np.sum(np.array(cwt_add), axis=0)))

    # We check how many we have to shift for each channels to get back to the normal activation
    new_cwt_emplacement_left = ((np.array(class_mean) - np.array(index_normal_class)) % 10)
    new_cwt_emplacement_right = ((np.array(index_normal_class) - np.array(class_mean)) % 10)

    shifts_array = []
    for valueA, valueB in zip(new_cwt_emplacement_left, new_cwt_emplacement_right):
        if valueA < valueB:
            # We want to shift toward the left (the start of the array)
            orientation = -1
            shifts_array.append(orientation*valueA)
        else:
            # We want to shift toward the right (the end of the array)
            orientation = 1
            shifts_array.append(orientation*valueB)

    # We get the mean amount of shift and round it up to get a discrete number representing how much we have to shift
    # if we consider all the canals
    # Do the shifting only if the absolute mean is greater or equal to 0.5
    final_shifting = np.mean(np.array(shifts_array))
    if abs(final_shifting) >= 0.5:
        final_shifting = int(np.round(final_shifting))
    else:
        final_shifting = 0

    # Build the dataset of the candiate with the circular shift taken into account.
    X_example = []
    Y_example = []
    for k in range(len(examples)):
        sub_ensemble_example = []
        for example in examples[k]:
            sub_ensemble_example.append(np.roll(np.array(example), final_shifting))
        X_example.append(sub_ensemble_example)
        Y_example.append(labels[k])
    return X_example, Y_example

def read_single_data(path: str, subject: str) -> tuple[list,list]:
    '''
    读取单个测试目标的数据
    '''
    print("Reading Data")

    labels = []
    examples = []
    for i in range(number_of_classes * 4):

        datafile = f'{path}/{subject}'+'/training0/classe_%d.dat' % i  # 数据文件地址

        data_read_from_file = np.fromfile(datafile, dtype=np.int16)
        data_read_from_file = np.array(data_read_from_file, dtype=np.float32)

        dataset_example = format_data_to_train(data_read_from_file)
        examples.append(dataset_example)
        labels.append((i % number_of_classes) + np.zeros(dataset_example.shape[0]))

    examples, labels = shift_electrodes(examples, labels)

    print("Finished Reading Data")
    return examples, labels

def read_data(path: str, num_male: int=12, num_female: int=7) -> tuple[list,list]:
    '''
    数据读取函数
    :param num_male: number of male participants
    :param num_female: number of female participants
    '''

    print("Reading Data")
    list_dataset = []
    list_labels = []

    for candidate in range(num_male):
        labels = []
        examples = []
        for i in range(number_of_classes * 4):

            datafile = path+'/Male'+str(candidate)+'/training0/classe_%d.dat' % i  # 数据文件地址
            print(datafile)

            data_read_from_file = np.fromfile(datafile, dtype=np.int16)
            data_read_from_file = np.array(data_read_from_file, dtype=np.float32)

            dataset_example = format_data_to_train(data_read_from_file)
            examples.append(dataset_example)
            labels.append((i % number_of_classes) + np.zeros(dataset_example.shape[0]))
        examples, labels = shift_electrodes(examples, labels)
        list_dataset.append(examples)
        list_labels.append(labels)

    for candidate in range(num_female):
        labels = []
        examples = []
        for i in range(number_of_classes * 4):
            i=0

            datafile = path + '/Female' + str(candidate) + '/training0/classe_%d.dat' % i  # 数据文件地址
            print(datafile)

            data_read_from_file = np.fromfile(datafile, dtype=np.int16)
            data_read_from_file = np.array(data_read_from_file, dtype=np.float32)
            dataset_example = format_data_to_train(data_read_from_file)
            examples.append(dataset_example)
            labels.append((i % number_of_classes) + np.zeros(dataset_example.shape[0]))
        examples, labels = shift_electrodes(examples, labels)
        list_dataset.append(examples)
        list_labels.append(labels)

    print("Finished Reading Data")
    return list_dataset, list_labels

if __name__ == '__main__':
    # example, label = read_data(f'{default_dataset_path}/Female0/training0/classe_0.dat')

    mode = 'single'

    if mode == 'single':

        example, label = read_single_data(default_dataset_path, 'Female0')

        print(len(label))
        print(len(label[0]))

        print(f'The length of example is: {len(example)}; and the length of label is: {len(label)}.')

        print(np.array(example[0]).shape)

        data = np.array(example[0])[:, 0, 0, 0]

        plt.plot(data)
        plt.show(block=True)

    elif mode == 'multiple':

        example, label = read_data(default_dataset_path)

        print(len(label[0]))
        print(len(label[0][0]))

        print(f'The length of example is: {len(example)}; and the length of label is: {len(label)}.')
        print(f'The length of element of example is: {len(example[0])}; and the length of element of  label is: {len(label[0])}.')

        print(np.array(example[0][8]).shape)

        data = np.array(example[0][0])[:,0,0,0]
        print(data)

        plt.plot(data)
        plt.show(block=True)

    else:
        print('Invalid mode!')