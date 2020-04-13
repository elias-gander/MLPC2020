import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


class GridSearchWrapper:
    def __init__(self, classifier, scaler=None, projector=None):
        pipeline = []

        if scaler is not None:
            pipeline.append(('scaler', scaler))

        if projector is not None:
            pipeline.append(('projector', projector))

        pipeline.append(('classifier', classifier))

        self.pipeline = Pipeline(pipeline)

    def fit(self, X, y, param_grid, cv):
        for key in list(param_grid.keys()):
            param_grid[f'classifier__{key}'] = param_grid.pop(key)

        self.param_grid = param_grid

        grid_search = GridSearchCV(self.pipeline, param_grid=param_grid, cv=cv)
        grid_search.fit(X, y)

        self.results = grid_search.cv_results_

    def report(self, n):
        for i in range(1, n + 1):
            candidates = np.flatnonzero(self.results['rank_test_score'] == i)

            for candidate in candidates:
                print(f'Model with rank: {i}')
                print('Mean validation score: {0:.3f} (std: {1:.3f})'.format(self.results['mean_test_score'][candidate],
                                                                             self.results['std_test_score'][candidate]))
                print(f"Parameters: {self.results['params'][candidate]}\n")

    def visualize(self, param_keys=None):
        if param_keys is None:
            param_keys = self.param_grid.keys()

        for key in param_keys:
            if key not in self.param_grid.keys():
                raise AssertionError(f'Unknown key {key}')

        combinations = list(itertools.combinations(param_keys, 2))
        count = np.sum([len(self.param_grid[key]) for _, key in combinations])

        width = 14
        cols = 4
        rows = np.math.ceil(count / cols)
        fig, axs = plt.subplots(rows, cols, figsize=(width, rows * width / cols))
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        ax_index = 0

        for param1, param2 in combinations:
            for value in self.param_grid[param2]:
                x = []
                y = []
                e = []

                if len(axs.shape) == 1:
                    ax = axs[ax_index]
                else:
                    ax = axs[int(ax_index / cols), int(ax_index % cols)]

                for index, params in enumerate(self.results['params']):
                    if params[param2] == value:
                        x.append(params[param1])
                        y.append(self.results['mean_test_score'][index])
                        e.append(self.results['std_test_score'][index])

                ax.errorbar(x, y, yerr=e, fmt='o')
                ax.set_ylim(0, 1)
                ax.set_xlabel(param1.replace('classifier__', ''))
                ax.set_ylabel('accuracy')
                ax.set_title(f'{param2.replace("classifier__", "")} = {value}')
                ax_index += 1
