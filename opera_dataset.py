import pandas as pd
import itertools
import os
import numpy as np

from utils import extract_number


class OperaDataset:
    def __init__(self, base_dir='data/train', max_files=None, remove_conflicting=True):
        self.metadata = pd.read_csv('data/train/metadata.txt')

        with open(os.path.join(base_dir, 'feature_names.txt'), 'r') as f:
            self.feature_names = f.read().splitlines()

        self.__initialize_data(base_dir, max_files, remove_conflicting)

    def __initialize_data(self, base_dir, max_files, remove_conflicting):
        feature_files, label_files, max_files = self.__get_train_test_files(base_dir, max_files)

        data = self.__extract_data(base_dir, feature_files, label_files, max_files, remove_conflicting)

        self.data = pd.DataFrame(data,
                                 columns=np.concatenate([self.feature_names, self.metadata.columns, self.label_names]))

        # convert labels to boolean
        for l in self.label_names:
            self.data[l] = self.data[l].astype('bool')

        def calc_label(row):
            label = ('C' if row.choral else '') + ('F' if row.female else '') + ('M' if row.male else '')

            if label == '':
                label = 'None'

            return label

        # calculate combined label
        self.data['label'] = self.data.apply(lambda row: calc_label(row), axis=1)

    def __get_train_test_files(self, base_dir, max_files):
        files = os.listdir(base_dir)
        files.sort()

        feature_files = [f for f in files if f.endswith('.npy')]
        label_files = [f for f in files if f.endswith('.npz')]

        if max_files is None:
            max_files = len(feature_files)

        return feature_files, label_files, max_files

    def __extract_data(self, base_dir, feature_files, label_files, max_files, remove_conflicting):
        self.label_names = ['choral', 'female', 'male']

        data = None

        for f_file, l_file in itertools.islice(zip(feature_files, label_files), max_files):
            features = np.load(os.path.join(base_dir, f_file))

            index = extract_number(f_file)
            meta = self.metadata[self.metadata.file == index].values.repeat(features.shape[0], 0)

            labels = np.load(os.path.join(base_dir, l_file))

            combined = np.hstack((features, meta))

            for l in self.label_names:
                combined = np.hstack((combined, np.expand_dims(labels[l][:, 0], axis=1)))

            if remove_conflicting:
                conflicting = self.__find_conflicting_indices(labels)
                combined = np.delete(combined, conflicting, 0)

            if data is None:
                data = combined
            else:
                data = np.concatenate([data, combined])

        return data

    def __find_conflicting_indices(self, labels):
        conflicting = []

        for name in self.label_names:
            l = labels[name]

            if l.shape[1] > 1:
                conflicting = np.concatenate((conflicting, np.nonzero(l[:, 0] - l[:, 1])[0]))

        return np.unique(conflicting)

    def split(self, train_split=0.55, validation_split=0.2, seed=42):
        unique_performances = self.data.performance.unique()
        num_performances = len(unique_performances)

        np.random.seed(seed)
        np.random.shuffle(unique_performances)

        train_end_index = int(num_performances * train_split)
        validation_end_index = train_end_index + int(
            num_performances * validation_split)

        self.train_performances = unique_performances[:train_end_index]
        self.validation_performances = unique_performances[train_end_index:validation_end_index]
        self.test_performances = unique_performances[validation_end_index:]

    def generate_train_test_validation(self, label, feature_columns=None, sampling=None):
        if feature_columns is None:
            feature_columns = self.feature_names[:]

        train_data = self.data_for_performances(self.train_performances, label, sampling)
        X_train = train_data[feature_columns]
        y_train = train_data[label]

        validation_data = self.data_for_performances(self.validation_performances, label, sampling)
        X_validation = validation_data[feature_columns]
        y_validation = validation_data[label]

        test_data = self.data_for_performances(self.test_performances, label, sampling)
        X_test = test_data[feature_columns]
        y_test = test_data[label]

        return X_train, y_train, X_validation, y_validation, X_test, y_test

    def generate_folds(self, label, feature_columns=None, k=5, seed=42, sampling=None):
        if feature_columns is None:
            feature_columns = self.feature_names[:]

        # TODO: smarter split. Some performances are longer than others
        unique_performances = np.concatenate([self.train_performances, self.validation_performances])
        num_performances = len(unique_performances)

        data = self.data_for_performances(unique_performances, label, sampling)

        np.random.seed(seed)
        np.random.shuffle(unique_performances)

        prev_start_index = 0
        step = np.math.ceil(num_performances / k)
        cv_indices = []

        for i, end_index in enumerate(np.arange(start=step, stop=num_performances + step, step=step)):
            if end_index > num_performances:
                end_index = num_performances

            performances = np.take(unique_performances, range(prev_start_index, end_index))

            cv_indices.append((data[~data.performance.isin(performances)].index.values.astype(int),
                               data[data.performance.isin(performances)].index.values.astype(int)))

            prev_start_index = end_index

        return data[feature_columns], data[label], cv_indices

    def data_for_performances(self, performances, label, sampling=None):
        data = self.data[self.data.performance.isin(performances)]

        if sampling is None:
            return data

        counts = data[label].value_counts().to_numpy()
        classes = [data[data[label] == False], data[data[label] == True]]

        min_class = counts.argmin()
        max_class = counts.argmax()

        if sampling == 'down':
            classes[max_class] = classes[max_class].sample(counts[min_class])
        elif sampling == 'up':
            classes[min_class] = classes[min_class].sample(counts[max_class], replace=True)

        data = pd.concat(classes)

        # create continuous index
        data.index = np.arange(len(data))

        return data

    def features(self):
        return self.data[self.feature_names]

    def labels(self, label='label'):
        return self.columns([label])

    def columns(self, columns):
        if not isinstance(columns, list):
            columns = [columns]

        return self.data[columns]

    def correlation_matrix(self, target=None):
        columns = self.feature_names[:]

        if target is not None:
            columns += [target]

        corr_matrix = self.data[columns].corr()

        if target is None:
            return corr_matrix
        else:
            return corr_matrix[target].sort_values(ascending=False)
