import numpy as np
import copy
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd


class EvaluationRunner:
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def cross_validate(self, folds, scaler=None, projector=None):
        rows = []

        folds = self.__concatenate_folds(folds, scaler, projector)

        for name, classifier in self.classifiers.items():
            print(f'Evaluating {name}')

            rows += self.__cross_validate(folds, classifier, name)

        return pd.DataFrame(rows)

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

    def __cross_validate(self, folds, classifier, name):
        rows = []

        for i, fold in enumerate(folds):
            model = copy.deepcopy(classifier)

            model.fit(fold['X_train'], fold['y_train'])

            row = self.__evaluate(model, fold['X_validation'], fold['y_validation'])
            row['name'] = name
            row['fold'] = i

            rows.append(row)

        return rows

    def __evaluate(self, classifier, X, y):
        y_pred = classifier.predict(X)

        confusion = confusion_matrix(y, y_pred, labels=[0, 1])
        tp = confusion[0, 0]
        fn = confusion[0, 1]
        fp = confusion[1, 0]
        tn = confusion[1, 1]

        accuracy = (tp + tn) / float(tp + tn + fp + fn)

        fnr = fn / float(fn + tp)
        fpr = fp / float(fp + tn)

        probs = classifier.predict_proba(X)
        y_pred = probs[:, 1]

        fprs, tprs, _ = roc_curve(y, y_pred)
        auc_score = auc(fprs, tprs)

        return { 'accuracy': accuracy, 'fpr': fpr, 'fnr': fnr, 'auc_score': auc_score, 'fprs': fprs, 'tprs': tprs }
