import argparse
import logging
from collections import defaultdict, Counter

import numpy as np
import os
import warnings
import datasets
from pathlib import Path
import tensorflow as tf
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted

warnings.filterwarnings("ignore")

from data_loader import load_mnist_data
from supervised_models import logit, xgb_model, mlp

from vime_self import vime_self
from vime_semi import vime_semi
from vime_utils import perf_metric

datasets.set_caching_enabled(True)
logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # mnist()
    # return

    # Configuration
    data_dir = Path("/data/IBC/stefan_ibc/omop-pkg/external/numpy_datasets")
    dataset_folders = {
        'income': 'income-',
        'car': 'car-20220708-144656',
        'heart': 'heart-20220708-140117',
        'diabetes': 'diabetes-'
    }
    data_dir = data_dir / dataset_folders[args.dataset]

    # Hyper-parameters
    parameters = {
        'vime': {
            'p_m': [0.3],
            'alpha': [2.0],
            'K': [3],
            'beta': [1.0],
            'mlp_hidden_dim': [100],
            'mlp_epochs': [100],
            'mlp_activation': ['relu'],
            'mlp_batch_size': [100],
            'vime_self_batch_size': [128],
            'vime_self_epochs': [10],
            'vime_semi_hidden_dim': [100],
            'vime_semi_batch_size': [128],
            'vime_semi_iterations': [1000]
        }
    }

    metric = 'roc_auc'  # accuracy
    num_shots = [16, 32, 64]  # , 128]  # , 256, 512, 1024, 2048, 4096, 8192, 16384, 50000, 'all']  # ['all']
    seeds = [42, 1024, 0]  # , 1, 32, 45, 655, 186, 126, 836]
    seeded_results = defaultdict(list)
    if metric == 'roc_auc' and args.dataset == 'car':
        # This computes the roc_auc_score for ovr on macro level:
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        metric = 'roc_auc_ovr'
    model = 'vime'
    print(f"Evaluate dataset {args.dataset} with model {model}.")
    for i, seed in enumerate(seeds):
        for num_shot in num_shots:
            with open(data_dir / (args.dataset + '_numshot' + str(num_shot) + '_seed' + str(seed)), 'rb') as f:
                npzfile = np.load(f)
                X_train = npzfile['x_train']
                y_train = npzfile['y_train']
                X_test = npzfile['x_test']
                y_test = npzfile['y_test']
                X_unlab = npzfile['x_unlab']

                folds = 4
                inner_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
                estimator = VimeSemiEstimator(x_unlab=X_unlab)
                # estimator.fit(X_train, y_train)
                # a = estimator.predict(X_test)

                clf = GridSearchCV(estimator=estimator, param_grid=parameters['vime'], cv=inner_cv, scoring=metric)
                clf.fit(X_train, y_train)

                # TODO: Need to extract best parameters manually and retrain model.
                best_params = clf.best_params_
                print(f"Best parameter config: {best_params}")
                # Reset graph to prevent error of vime
                tf.reset_default_graph()
                clf = VimeSemiEstimator(x_unlab=X_unlab, **best_params)
                clf.fit(X_train, y_train)
                p = clf.predict_proba(X_test)[:, 1]
                results = roc_auc_score(y_test, p)
                # print(f"AUC {num_shot}: {str(metric_score)}")

            # results_lr, results_vime_self, results_vime_semi = \
            #    evaluate_model(seed, model, metric, parameters[model], X_train, y_train, X_test, y_test, X_unlab)
            # seeded_results_lr[num_shot] = seeded_results_lr[num_shot] + [results_lr]
            # seeded_results_vime_self[num_shot] = seeded_results_vime_self[num_shot] + [results_vime_self]
            seeded_results[num_shot] = seeded_results[num_shot] + [results]

    for sr in [seeded_results]:
        for k, v in sr.items():
            print(f"Shots {k}: {result_str(v)}")
        print()


def evaluate_model(seed, model, metric, parameters, x_train, y_train, x_test, y_test, x_unlab):
    print(f"\tUse {x_train.shape[0]} train, {x_unlab.shape[0]} unlabeled, and {x_test.shape[0]} test examples.")

    y_test_hat = logit(x_train, y_train, x_test)
    result_lr = perf_metric(metric, y_test, y_test_hat)

    p_m = 0.3
    alpha = 2.0
    K = 3
    beta = 1.0

    # MLP/
    mlp_parameters = dict()
    mlp_parameters['hidden_dim'] = 100
    mlp_parameters['epochs'] = 100
    mlp_parameters['activation'] = 'relu'
    mlp_parameters['batch_size'] = 100

    # Train VIME-Self
    vime_self_parameters = dict()
    vime_self_parameters['batch_size'] = 128
    vime_self_parameters['epochs'] = 10
    vime_self_encoder = vime_self(x_unlab, p_m, alpha, vime_self_parameters)

    # Save encoder
    if not os.path.exists('save_model'):
        os.makedirs('save_model')

    file_name = './save_model/encoder_model.h5'

    vime_self_encoder.save(file_name)

    # Test VIME-Self
    x_train_hat = vime_self_encoder.predict(x_train)
    x_test_hat = vime_self_encoder.predict(x_test)

    y_test_hat = mlp(x_train_hat, y_train, x_test_hat, mlp_parameters)
    result_vime_self = perf_metric(metric, y_test, y_test_hat)

    # Train VIME-Semi
    vime_semi_parameters = dict()
    vime_semi_parameters['hidden_dim'] = 100
    vime_semi_parameters['batch_size'] = 128
    vime_semi_parameters['iterations'] = 1000
    y_test_hat = vime_semi(x_train, y_train, x_unlab, x_test,
                           vime_semi_parameters, p_m, K, beta, file_name)

    # Test VIME
    results_vime_semi = perf_metric(metric, y_test, y_test_hat)

    return result_lr, result_vime_self, results_vime_semi


def mnist():
    # Experimental parameters
    label_no = 1000
    model_sets = ['logit', 'xgboost', 'mlp']

    # Hyper-parameters
    p_m = 0.3
    alpha = 2.0
    K = 3
    beta = 1.0
    label_data_rate = 0.1

    # Metric
    metric = 'acc'

    # Define output
    results = np.zeros([len(model_sets) + 2])

    # Load data
    x_train, y_train, x_unlab, x_test, y_test = load_mnist_data(label_data_rate)

    # Use subset of labeled data
    x_train = x_train[:label_no, :]
    y_train = y_train[:label_no, :]

    # Logistic regression
    y_test_hat = logit(x_train, y_train, x_test)
    results[0] = perf_metric(metric, y_test, y_test_hat)

    # XGBoost
    # y_test_hat = xgb_model(x_train, y_train, x_test)
    # results[1] = perf_metric(metric, y_test, y_test_hat)
    results[1] = 0

    # MLP
    mlp_parameters = dict()
    mlp_parameters['hidden_dim'] = 100
    mlp_parameters['epochs'] = 100
    mlp_parameters['activation'] = 'relu'
    mlp_parameters['batch_size'] = 100

    # y_test_hat = mlp(x_train, y_train, x_test, mlp_parameters)
    # results[2] = perf_metric(metric, y_test, y_test_hat)
    results[2] = 0

    # Train VIME-Self
    vime_self_parameters = dict()
    vime_self_parameters['batch_size'] = 128
    vime_self_parameters['epochs'] = 10
    vime_self_encoder = vime_self(x_unlab, p_m, alpha, vime_self_parameters)

    # Save encoder
    if not os.path.exists('save_model'):
        os.makedirs('save_model')

    file_name = './save_model/encoder_model.h5'

    vime_self_encoder.save(file_name)

    # Test VIME-Self
    x_train_hat = vime_self_encoder.predict(x_train)
    x_test_hat = vime_self_encoder.predict(x_test)

    y_test_hat = mlp(x_train_hat, y_train, x_test_hat, mlp_parameters)
    results[3] = perf_metric(metric, y_test, y_test_hat)

    print('VIME-Self Performance: ' + str(results[3]))

    # Train VIME-Semi
    vime_semi_parameters = dict()
    vime_semi_parameters['hidden_dim'] = 100
    vime_semi_parameters['batch_size'] = 128
    vime_semi_parameters['iterations'] = 1000
    y_test_hat = vime_semi(x_train, y_train, x_unlab, x_test, vime_semi_parameters, p_m, K, beta, file_name)

    # Test VIME
    results[4] = perf_metric(metric, y_test, y_test_hat)

    print('VIME Performance: ' + str(results[4]))

    # Report performance
    for m_it in range(len(model_sets)):
        model_name = model_sets[m_it]

        print('Supervised Performance, Model Name: ' + model_name +
              ', Performance: ' + str(results[m_it]))


def result_str(scores):
    if len(scores) > 1:
        return f"{np.mean(scores):.2f} ({np.std(scores):.2f})"
    else:
        return f"{scores[0] * 100:.2f}"


def parse_args():
    parser = argparse.ArgumentParser(description="Create note dataset from cohort.")
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--dataset",
        type=str
    )

    args = parser.parse_args()

    return args


class VimeSemiEstimator(BaseEstimator):
    """ A template estimator to be used as a reference implementation.
    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    Examples
    --------
    """

    def __init__(
            self,
            x_unlab=np.array([[]]),
            p_m=0.3,
            alpha=2.0,
            K=3,
            beta=1.0,
            mlp_hidden_dim=100,
            mlp_epochs=100,
            mlp_activation='relu',
            mlp_batch_size=100,
            vime_self_batch_size=128,
            vime_self_epochs=10,
            vime_semi_hidden_dim=100,
            vime_semi_batch_size=128,
            vime_semi_iterations=1000
    ):
        self.x_unlab = x_unlab
        self.x_train = None
        self.y_train = None

        self.p_m = p_m
        self.alpha = alpha
        self.K = K
        self.beta = beta

        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_epochs = mlp_epochs
        self.mlp_activation = mlp_activation
        self.mlp_batch_size = mlp_batch_size

        self.vime_self_batch_size = vime_self_batch_size
        self.vime_self_epochs = vime_self_epochs

        self.vime_semi_hidden_dim = vime_semi_hidden_dim
        self.vime_semi_batch_size = vime_semi_batch_size
        self.vime_semi_iterations = vime_semi_iterations

    def fit(self, X, y):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True)
        self.x_train = X
        self.y_train = y
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict_proba(self, X):
        print(f"Predict proba {X.shape}")

        # Train VIME-Self
        vime_self_parameters = {
            'batch_size': self.vime_self_batch_size,
            'epochs': self.vime_self_epochs
        }
        vime_self_encoder = vime_self(self.x_unlab, self.p_m, self.alpha, vime_self_parameters)

        # Save encoder
        if not os.path.exists('save_model'):
            os.makedirs('save_model')
        file_name = './save_model/encoder_model.h5'
        vime_self_encoder.save(file_name)

        # Test VIME-Self
        # x_train_hat = vime_self_encoder.predict(x_train)
        # x_test_hat = vime_self_encoder.predict(x_test)
        # y_test_hat = mlp(x_train_hat, y_train, x_test_hat, mlp_parameters)
        # results[3] = perf_metric(metric, y_test, y_test_hat)

        # Convert labels into 1-hot encoding
        def convert_labels_into_1_hot(a):
            inflated = np.zeros((a.size, a.max() + 1))
            inflated[np.arange(a.size), a] = 1
            return inflated

        # Train VIME-Semi
        vime_semi_parameters = {
            'hidden_dim': self.vime_semi_hidden_dim,
            'batch_size': self.vime_semi_batch_size,
            'iterations': self.vime_semi_iterations
        }
        y_train = convert_labels_into_1_hot(self.y_train)
        y_test_hat = vime_semi(self.x_train, y_train, self.x_unlab, X, vime_semi_parameters,
                               self.p_m, self.K, self.beta, file_name)

        return y_test_hat

    def predict(self, X):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        y_test_hat = self.predict_proba(X)
        return np.argmax(y_test_hat, axis=1)


if __name__ == '__main__':
    main()
