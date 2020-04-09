import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from collections import defaultdict


class GridSearchWrapper:
    def __init__(self, classifier, scaler=None, projector=None):
        pipeline = []

        if scaler is not None:
            pipeline.append(('scaler', scaler))

        if projector is not None:
            pipeline.append(('projector', projector))

        pipeline.append(('classifier', classifier))

        self.pipeline = Pipeline(pipeline)

    def fit(self, X, y, param_grid, cv, report_top=3, visualize=True):
        for key in list(param_grid.keys()):
            param_grid[f'classifier__{key}'] = param_grid.pop(key)

        grid_search = GridSearchCV(self.pipeline, param_grid=param_grid, cv=cv)
        grid_search.fit(X, y)

        if report_top > 0:
            self.__report(grid_search.cv_results_, report_top)

        if visualize:
            self.__visualize(grid_search.cv_results_, param_grid)

        return grid_search

    def __report(self, results, n_top):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print(f'Model with rank: {i}')
                print('Mean validation score: {0:.3f} (std: {1:.3f})'.format(results['mean_test_score'][candidate],
                                                                             results['std_test_score'][candidate]))
                print(f"Parameters: {results['params'][candidate]}\n")

    def __visualize(self, results, params):
        x = defaultdict(list)
        y = defaultdict(list)

        param_keys = list(params.keys())

        # TODO: visualize additional params
        for index, params in enumerate(results['params']):
            x[param_keys[0]].append(param_keys[1])
            y[param_keys[0]].append(results['mean_test_score'][index])

        fig, axs = plt.subplots(1, 2, figsize=(15, 7))
        ax = axs[0]
        for index, (key, value) in enumerate(x.items()):
            ax.plot(value, y[key], label=f'{key}')

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels)
        plt.show()
