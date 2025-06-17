import copy
import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from ..utils import noise_level_estimator as noise_est
from .clean_dt import DecisionTreeLevelWise as custom_tree

class RandomForestClassifier2Step(object):
    # initializer
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        kappa: str = None,  # either "1nn" or "lasso" or "mean_var" or "no_es"
        max_features: str = None,  # rn: either None or "sqrt"
        random_state=7,
        apply_es=False,
        es_offset=0,
        disable_pb=True,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.estimators_ = []
        self.max_features = max_features
        self.kappa = kappa
        self.apply_es = apply_es
        self.es_offset = es_offset
        self.disable_pb = disable_pb

        # Noise Related Quantities for algo inspection
        # when f_train is given
        self.mean_var_true_train_noise = np.nan

        # Estimates
        self.mean_estimated_train_noise = (
            np.nan
        )  # this is the estimated noise / kappa (before sampling)

        # Set the random seed for reproducibility
        np.random.seed(random_state)

    # private function to make bootstrap samples
    def __make_bootstraps(self, data):  # f is corresponding error to data
        # initialize output dictionary & unique value count
        bootstrap_sets_dict = {}
        # get sample size
        bootstrap_size = data.shape[0]
        # get list of row indexes
        data_indices = [i for i in range(bootstrap_size)]
        # loop through the required number of bootstraps
        for bootstrap_id in range(self.n_estimators):
            # obtain boostrap samples with replacement
            bootstrap_indices = np.random.choice(
                data_indices, replace=True, size=bootstrap_size
            )
            bootstrap_sample = data[bootstrap_indices, :]

            # obtain out-of-bag samples for the current b
            out_of_bag_indices = list(set(data_indices) - set(bootstrap_indices))
            out_of_bag_sample = np.array([])
            if out_of_bag_indices:
                out_of_bag_sample = data[out_of_bag_indices, :]
            # store results
            bootstrap_sets_dict["boot_" + str(bootstrap_id)] = {
                "boot": bootstrap_sample,
                "test": out_of_bag_sample,
            }
        return bootstrap_sets_dict

    # train the ensemble
    def fit(self, X_train, y_train, f_train=None):
        # Import here to avoid circular import
        from infrastructure.model_builder import build_post_pruned_dt_clf

        # Package the input data for bootstrapping
        training_data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
        # make bootstrap samples
        bootstrap_sets_dict = self.__make_bootstraps(data=training_data)
        if self.apply_es:
            noise_estimator = noise_est.Estimator(X_train, y_train)
            if self.kappa == "1nn":
                self.mean_estimated_train_noise = noise_estimator.estimate_1NN()
            elif self.kappa == "true_noise_level":
                self.mean_estimated_train_noise = np.mean(f_train * (1 - f_train))
            elif self.kappa == "no_es":
                # self.max_depth = 999999999
                self.max_depth = None
            else:
                raise ValueError("Kappa not recognized")

            # Get ES Depth
            tree = custom_tree(
                max_depth=None,
                min_samples_split=2,
                kappa=self.mean_estimated_train_noise,
                max_features=self.max_features,
            )
            tree.fit(X_train, y_train)
            es_depth = tree.get_depth() + self.es_offset
            if self.apply_es:
                self.max_depth = es_depth

        for bootstrap_sample_id in tqdm(
            bootstrap_sets_dict, desc="Training Trees", disable=self.disable_pb
        ):  # bootstrap_sets_dict: #
            X_bootstrap = bootstrap_sets_dict[bootstrap_sample_id]["boot"][:, :-1]
            y_bootstrap = bootstrap_sets_dict[bootstrap_sample_id]["boot"][
                :, -1
            ].reshape(-1)

            bootstrap_es_dt_classifier = build_post_pruned_dt_clf(
                X_train=X_bootstrap,
                y_train=y_bootstrap,
                random_state=7,
                max_depth=self.max_depth,
                n_cv_alpha=5,
                full_alpha_range=False,
                max_features=self.max_features,
            )
            # bootstrap_es_dt = DecisionTreeClassifier(
            #     max_depth=self.max_depth,
            #     min_samples_split=2,
            #     max_features=self.max_features,
            #     random_state=7,  # does not matter at this point!
            # )
            # ccp_alphas = bootstrap_es_dt.cost_complexity_pruning_path(
            #     X_bootstrap, y_bootstrap
            # ).ccp_alphas
            # # in ccp_alphas make negative elements to zero (prevent number underflow)
            # ccp_alphas[ccp_alphas < 0] = 0
            # # fit a decision tree classifier to the current sample
            # bootstrap_es_dt_classifier = GridSearchCV(
            #     bootstrap_es_dt,
            #     param_grid={"ccp_alpha": ccp_alphas},
            #     cv=5,
            # )
            # bootstrap_es_dt_classifier.fit(X_bootstrap, y_bootstrap)
            # bootstrap_es_dt_classifier = bootstrap_es_dt_classifier.best_estimator_
            # append the fitted model
            self.estimators_.append(bootstrap_es_dt_classifier)

    # predict from the ensemble
    def predict(self, X, max_depth_per_tree=None, n_trees=None):
        if n_trees is None:
            n_trees = self.n_estimators
        # check we've fit the ensemble
        if not self.estimators_:
            print("You must train the ensemble before making predictions!")
            return None
        if max_depth_per_tree is None:
            max_depth_per_tree = np.inf

        # loop through each fitted model
        predictions = []
        for model in self.estimators_[:n_trees]:
            # if model.stopping_index is None:
            #     breakpoint()
            # make predictions on the input X
            y_pred_labels_single_tree = model.predict(X)

            # append predictions to storage list
            predictions.append(y_pred_labels_single_tree.reshape(-1, 1))
        # compute the ensemble prediction
        y_pred_labels_ensemble = np.round(
            np.mean(np.concatenate(predictions, axis=1), axis=1)
        ).astype(int)
        # return the prediction
        return y_pred_labels_ensemble

    def predict_proba(
        self, X, max_depth_per_tree=None, n_trees=None
    ):  # TODO: Adjust to kappa and depth
        if n_trees is None:
            n_trees = self.n_estimators
        # check we've fit the ensemble
        if not self.estimators_:
            print("You must train the ensemble before making predictions!")
            return None
        if max_depth_per_tree is None:
            max_depth_per_tree = np.inf
        # loop through each fitted model
        predictions = []
        for model in self.estimators_[:n_trees]:
            # if model.stopping_index is None:
            #     breakpoint()
            # make predictions on the input X
            y_pred_labels_single_tree = model.predict(X)

            # append predictions to storage list
            predictions.append(y_pred_labels_single_tree.reshape(-1, 1))
        # compute the ensemble prediction
        y_pred_labels_ensemble = np.mean(np.concatenate(predictions, axis=1), axis=1)
        # conc = column stack, axis=1 je zeile mean nehmen/über columns
        y_pred_prob_ensemble = np.column_stack(
            (1 - y_pred_labels_ensemble, y_pred_labels_ensemble)
        )
        # return the prediction
        return y_pred_prob_ensemble
