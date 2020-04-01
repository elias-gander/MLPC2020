import numpy as np
import copy
from sklearn.metrics import accuracy_score


class EvaluationRunner:
    def __init__(self, classifier):
        self.classifier = classifier

    def cross_validate(self, folds, verbose=True):
        accuracies = []

        for index, test in enumerate(folds):
            classifier = copy.deepcopy(self.classifier)

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

            classifier.fit(X, y)

            y_pred = classifier.predict(test['X'])
            accuracy = accuracy_score(test['y'], y_pred)

            if verbose:
                print(f'Fold {index + 1} accuracy: {accuracy:.6f}')

            accuracies.append(accuracy)

        return accuracies
