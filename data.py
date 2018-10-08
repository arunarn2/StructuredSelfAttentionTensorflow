import re

import numpy as np
import pandas as pd


def batch_iterator(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data(input_file):
    train_data = pd.read_csv(input_file, header=None)
    x_text = []
    values = list(train_data[train_data.columns[0]] - 1)
    n_values = np.max(values) + 1
    y = np.array(np.eye(n_values)[values], int)
    x_data = list(train_data[train_data.columns[1]] + " " + train_data[train_data.columns[2]])
    for idx in range(train_data.shape[0]):
        text = clean_str(re.sub("^\s*(.-)\s*$", "%1", x_data[idx]).replace("\\n", "\n"))
        x_text.append(text)
    return x_text, y
