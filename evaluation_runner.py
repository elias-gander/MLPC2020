import numpy as np
import copy
from sklearn.metrics import accuracy_score


class EvaluationRunner:
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def cross_validate(self, folds, scaler=None, projector=None):
        accuracies = { }

        folds = self.__concatenate_folds(folds, scaler, projector)

        for name, classifier in self.classifiers.items():
            print(f'Evaluating {name}')

            accuracies[name] = self.__cross_validate(folds, classifier)

        return accuracies

    def __concatenate_folds(self, folds, scaler, projector):
        concatenated = []

        for index, validation in enumerate(folds):
            X_train = None
            y_train = None

            X_validation = validation['X']
            y_validation = validation['y']

            for i, fold in enumerate(folds):
                if i == index:
                    continue

                if X_train is None:
                    X_train = fold['X']
                    y_train = fold['y']
                else:
                    X_train = np.concatenate((X_train, fold['X']))
                    y_train = np.concatenate((y_train, fold['y']))

            if scaler is not None:
                s = copy.deepcopy(scaler)
                X_train = s.fit_transform(X_train)
                X_validation = s.transform(X_validation)

            if projector is not None:
                p = copy.deepcopy(projector)
                X_train = p.fit_transform(X_train)
                X_validation = p.transform(X_validation)

            concatenated.append(
                { 'X_train': X_train, 'y_train': y_train, 'X_validation': X_validation, 'y_validation': y_validation })

        return concatenated

    def __cross_validate(self, folds, classifier):
        accuracies = []

        for fold in folds:
            model = copy.deepcopy(classifier)

            model.fit(fold['X_train'], fold['y_train'])

            pred = model.predict(fold['X_validation'])
            accuracy = accuracy_score(fold['y_validation'], pred)

            accuracies.append(accuracy)

        return accuracies
