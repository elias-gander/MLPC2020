import numpy as np
import copy
from sklearn.metrics import accuracy_score


class EvaluationRunner:
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def cross_validate(self, folds):
        accuracies = { }

        for name, classifier in self.classifiers.items():
            print(f'Evaluating {name}')

            accuracies[name] = self.__cross_validate(folds, classifier)

        return accuracies

    def __cross_validate(self, folds, classifier):
        accuracies = []

        for index, test in enumerate(folds):
            model = copy.deepcopy(classifier)

            X = None
            y = None

            for i, fold in enumerate(folds):
                if i == index:
                    continue

                if X is None:
                    X = fold['X']
                    y = fold['y']
                else:
                    X = np.concatenate((X, fold['X']))
                    y = np.concatenate((y, fold['y']))

            model.fit(X, y)

            y_pred = model.predict(test['X'])
            accuracy = accuracy_score(test['y'], y_pred)

            accuracies.append(accuracy)

        return accuracies
