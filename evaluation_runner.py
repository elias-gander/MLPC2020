import copy
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd


class EvaluationRunner:
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def cross_validate(self, X, y, cv, scaler=None, projector=None):
        rows = []

        for name, classifier in self.classifiers.items():
            print(f'Evaluating {name}')

            rows += self.__cross_validate(X, y, cv, scaler, projector, classifier, name)

        return pd.DataFrame(rows)

    def __cross_validate(self, X, y, cv, scaler, projector, classifier, name):
        rows = []

        for i, (train_indices, test_indices) in enumerate(cv):
            model = copy.deepcopy(classifier)

            X_train = X[train_indices, :]
            X_test = X[test_indices, :]
            y_train = y[train_indices]
            y_test = y[test_indices]

            if scaler is not None:
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            if projector is not None:
                X_train = projector.fit_transform(X_train)
                X_test = projector.transform(X_test)

            model.fit(X_train, y_train)

            row = self.__evaluate(model, X_test, y_test)
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
