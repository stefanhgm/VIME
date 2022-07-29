import argparse
import datetime
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
import datasets
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, GridSearchCV, train_test_split, KFold
import tensorflow as tf
import random


from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y

from vime_self import vime_self
from vime_semi import vime_semi

datasets.set_caching_enabled(True)
logger = logging.getLogger(__name__)

# TODO: This is all copied over from evaluate_external except lines marked with TODO


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # TODO: Configuration
    data_dir = Path("/data/IBC/stefan_ibc/omop-pkg/external/numpy_datasets")
    dataset_folders = {
        'income': 'income-20220712-114316',
        'car': 'car-20220708-144656',
        'heart': 'heart-20220708-140117',
        'diabetes': 'diabetes-20220712-112505'
    }
    data_dir = data_dir / dataset_folders[args.dataset]

    # TODO: Hyper-parameters
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
    models = ['vime']
    # models = ['output_datasets']
    ts = datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")
    metric = 'roc_auc'  # accuracy
    num_shots = [4, 8, 16]  # , 32]  # , 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 50000, 'all']  # ['all']
    seeds = [42, 1024, 0, 1, 32, 45, 655, 186, 126, 836]
    seeded_results = defaultdict(list)
    if metric == 'roc_auc' and args.dataset == 'car':
        # This computes the roc_auc_score for ovr on macro level:
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        metric = 'roc_auc_ovr'
    for model in models:
        print(f"Evaluate dataset {args.dataset} with model {model}.")
        for i, seed in enumerate(seeds):
            # TODO: No need to create datasets
            # Tried to make vime reproducible, but still not possible
            # https://stackoverflow.com/questions/57246526/how-do-i-get-reproducible-results-with-keras
            # tf.set_random_seed(seed)
            # tf.compat.v1.set_random_seed(seed)
            # os.environ['PYTHONHASHSEED'] = str(seed)
            # np.random.seed(seed)
            # random.seed(seed)

            # session_conf = tf.compat.v1.ConfigProto(
            #     intra_op_parallelism_threads=1,
            #     inter_op_parallelism_threads=1
            # )
            # sess = tf.compat.v1.Session(
            #     graph=tf.compat.v1.get_default_graph(),
            #     config=session_conf
            # )
            # tf.compat.v1.keras.backend.set_session(sess)

            for num_shot in num_shots:
                # TODO: Read dataset
                with open(data_dir / (args.dataset + '_numshot' + str(num_shot) + '_seed' + str(seed)), 'rb') as f:
                    npzfile = np.load(f)
                    X_train = npzfile['x_train']
                    y_train = npzfile['y_train']
                    X_test = npzfile['x_test']
                    y_test = npzfile['y_test']
                    X_unlab = npzfile['x_unlab']

                    results = evaluate_model(seed, model, metric, parameters[model],
                                             X_train, y_train, np.array([]), np.array([]), X_test, y_test, X_unlab)
                    seeded_results[num_shot] = seeded_results[num_shot] + [results]

        for k, v in seeded_results.items():
            print(f"Shots {k}: {result_str(v)}")
    print()


def evaluate_model(seed, model, metric, parameters, X_train, y_train, X_valid, y_valid, X_test, y_test, X_unlab):
    print(f"\tUse {X_train.shape[0]} train, {X_valid.shape[0]} valid, and {X_test.shape[0]} test examples.")

    def compute_metric(clf_in, X, y):
        if metric == 'roc_auc':
            p = clf_in.predict_proba(X)[:, 1]
            metric_score = roc_auc_score(y, p)
        elif metric == 'roc_auc_ovr':
            p = clf_in.predict_proba(X)
            metric_score = roc_auc_score(y, p, multi_class='ovr', average='macro')
        elif metric == 'accuracy':
            p = np.argmax(clf_in.predict_proba(X), axis=1)
            metric_score = np.sum(p == np.array(y)) / p.shape[0]
        else:
            raise ValueError("Undefined metric.")
        return metric_score

    # Do a 4-fold cross validation on the training set for parameter tuning
    folds = min(Counter(y_train).values()) if min(Counter(y_train).values()) < 4 else 4  # If less than 4 examples
    if folds < 4:
        print(f"Manually reduced folds to {folds} since this is maximum number of labeled examples.")

    if folds > 1:
        inner_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    else:
        print(f"Warning: Increased folds from {folds} to 2 (even though not enough labels) and use simple KFold.")
        folds = 2
        inner_cv = KFold(n_splits=folds, shuffle=True, random_state=seed)

    estimator = VimeSemiEstimator(x_unlab=X_unlab)
    clf = GridSearchCV(estimator=estimator, param_grid=parameters, cv=inner_cv, scoring=metric)  #, verbose=4)
    clf.fit(X_train, y_train)
    # TODO: Manually extract parameters and retrain after reset to prevent vime error
    best_params = clf.best_params_
    print(f"Best parameter config: {best_params}")
    # Reset graph to prevent error of vime
    # tf.reset_default_graph()
    clf = VimeSemiEstimator(x_unlab=X_unlab, **best_params)
    clf.fit(X_train, y_train)

    score_test = compute_metric(clf, X_test, y_test)

    return score_test


def result_str(scores):
    if len(scores) > 1:
        return f"{np.mean(scores):.2f} ({np.std(scores):.2f})"
    else:
        return f"{scores[0] * 100:.2f}"


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
        # Identify as regressor
        self._estimator_type = 'regressor'

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

    def decision_function(self, X):
        # Leads to use predict_proba as fallback
        raise NotImplementedError

    def predict_proba(self, X):
        # print(f"Predict proba {X.shape}")

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
            inflated = np.zeros((a.size, a.max() + 1), dtype=int)
            inflated[np.arange(a.size), a] = 1
            return inflated

        # Train VIME-Semi
        vime_semi_parameters = {
            'hidden_dim': self.vime_semi_hidden_dim,
            'batch_size': self.vime_semi_batch_size,
            'iterations': self.vime_semi_iterations
        }
        y_train = convert_labels_into_1_hot(self.y_train)
        tf.compat.v1.reset_default_graph()
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


if __name__ == '__main__':
    main()
