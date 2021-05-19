import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, DotProduct, \
    RationalQuadratic
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from scipy.stats import loguniform as sp_lguniform
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from skopt.learning.gaussian_process.kernels import RBF
from skorch import NeuralNetClassifier
from hyperopt import hp

from base_ml.models import MLPClassifier, MGMCCheby


class HyperParameterProvider:
    """
    Provide hyperparameters for machine learning models.
    """

    def __init__(self, estimator, parameters=None):
        self.estimator = estimator
        self.parameters = parameters

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        # TODO check if estimator is correct
        self._estimator = value

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        print("Setting hyperparameter space...")
        if value is not None:
            value = value

        elif isinstance(self.estimator, LogisticRegression):
            value = {
                "C": sp_randint(1, 11),
                'penalty': ['elasticnet'],
                'solver': ['saga'],
                'l1_ratio': sp_uniform(0, 1)
            }

        elif isinstance(self.estimator, RandomForestClassifier):
            value = {
                "max_depth": [3, None],
                "max_features": sp_randint(1, 11),
                "min_samples_split": sp_randint(2, 11),
                "bootstrap": [True, False],
                "criterion": ["gini", "entropy"],
                "n_estimators": sp_randint(5, 50)
            }

        elif isinstance(self.estimator, KNeighborsClassifier):
            value = {
                "n_neighbors": sp_randint(3, 100),
                "weights": ["uniform", "distance"]
            }

        elif isinstance(self.estimator, SVC):
            value = {
                'C': sp_lguniform(1e-6, 1e+6),
                'gamma': sp_lguniform(1e-6, 1e+1),
                'degree': sp_randint(1, 8),  # integer valued parameter
                'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
            }

        elif isinstance(self.estimator, DecisionTreeClassifier):
            value = {
                "max_depth": [3, None],
                "max_features": sp_randint(1, 11),
                "min_samples_split": sp_randint(2, 11),
                "criterion": ["gini", "entropy"]
            }

        elif isinstance(self.estimator, GaussianProcessClassifier):
            value = {"kernel": [1 * RBF(), 1 * DotProduct(), 1 * Matern(),
                                1 * RationalQuadratic(), 1 * WhiteKernel()]}

        elif isinstance(self.estimator, AdaBoostClassifier):
            value = {
                'n_estimators': [50, 60, 70, 90, 100],
                'learning_rate': [0.5, 0.8, 1.0, 1.3]
            }

        elif isinstance(self.estimator, GaussianNB):
            value = {'var_smoothing': np.logspace(0, -9, num=100)}

        elif isinstance(self.estimator, LinearDiscriminantAnalysis):
            value = {
                "solver": ['svd', 'lsqr', 'eigen'],
                "shrinkage": np.arange(0, 1, 0.01)
            }

        elif isinstance(self.estimator, NeuralNetClassifier):
            # TODO dynamically adapt hyperparam space based on given layers
            if issubclass(self.estimator.module, MLPClassifier):
                value = {
                    'lr': [1e-1, 1e-2, 1e-3, 1e-4],
                    'optimizer__weight_decay': [0.0, 0.05, 0.5],
                    'module__layer_list': [32, 64, 128],
                    'module__p_list': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
                }

            elif issubclass(self.estimator.module, MGMCCheby):
                value = {
                    "ce_wt": hp.uniform('ce_wt', 0.001, 1000),
                    "frob_wt": hp.uniform('frob_wt', 0.001, 1000),
                    "dirch_wt": hp.uniform('dirch_wt', 0.001, 1000),
                }

        else:
            raise ValueError(f"{self.estimator} not supported.")

        self._parameters = value
