import copy
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from .clean_dt import DecisionTreeLevelWise as ESDecisionTreeClassifier
from ..utils import noise_level_estimator as noise_est

class RandomForestClassifier(object):
    # initializer
    def __init__(
        self,
        n_estimators=50,
        max_depth=None,
        kappa: str = None,  # either "1nn" or "mean_var" or "no_es"
        max_features: str = "sqrt",  # rn: either None or "sqrt"
        es_offset=0,
        estimate_noise_before_sampling: bool = True,
        random_state=7,
        disable_pb=True,
        two_step=False,
        ues_time_track = False,
        rf_train_mse = False
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.estimators_ = []
        self.stopping_indices_ = []
        self.max_features = max_features
        self.estimate_noise_before_sampling = estimate_noise_before_sampling
        self.kappa = kappa
        self.es_offset = es_offset
        self.disable_pb = disable_pb
        self.two_step = two_step
        self.rf_train_mse = rf_train_mse
        self.ues_time_track = ues_time_track
        # Noise Related Quantities for algo inspection
        # when f_train is given
        self.mean_var_true_train_noise = np.nan
        self.mean_var_true_bootstrap_noises = (
            []
        )  # this is true noise within the bootstrap samples

        # Estimates
        self.mean_estimated_train_noise = (
            np.nan
        )  # this is the estimated noise / kappa (before sampling)
        self.estimated_kappa_train_bootstrap = (
            []
        )  # these are the estimated bootstrap noises

        # Track individual tree characteristics
        self.tree_fit_durations_ = []
        # self.tree_fit_duration = np.nan
        self.random_state = random_state
        np.random.seed(self.random_state)

    # private function to make bootstrap samples
    def __make_bootstraps(self, data, f=None):  # f is corresponding error to data
        # initialize output dictionary & unique value count
        bootstrap_sets_dict = {}
        f_bootstrap_samples = {}
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
            if f is not None:
                f_bootstrap_samples["boot_" + str(bootstrap_id)] = f[bootstrap_indices]
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
        return bootstrap_sets_dict, f_bootstrap_samples

    # train the ensemble
    def fit(self, X_train, y_train, f_train=None):
        bootstrap_noise_start_time = time.time()
        # Package the input data for bootstrapping
        training_data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
        # make bootstrap samples
        bootstrap_sets_dict, bootstrap_f_dict = self.__make_bootstraps(
            data=training_data, f=f_train
        )

        # iterate through each bootstrap sample & fit a model
        # Get Noise Estimates for whole dataset for model analysis and eventually fitting
        if (
            f_train is not None and self.kappa != "no_es"
        ):  # and self.estimate_noise_before_sampling == True: # TODO Adjust here
            self.mean_var_true_train_noise = np.mean(f_train * (1 - f_train))
            # True Noise Level is always calculated

        if self.estimate_noise_before_sampling == True:
            noise_estimator = noise_est.Estimator(X_train, y_train)
            if self.kappa == "1nn":
                self.mean_estimated_train_noise = noise_estimator.estimate_1NN()
            # elif self.kappa == "lasso":
            #     self.mean_estimated_train_noise = noise_estimator.estimate_LASSO()
            elif self.kappa == "mean_var":
                self.mean_estimated_train_noise = self.mean_var_true_train_noise
            elif self.kappa == "no_es":
                self.mean_estimated_train_noise = np.nan
            else:
                raise ValueError("Kappa not recognized")

        bootstrap_noise_duration = time.time() - bootstrap_noise_start_time

        if self.ues_time_track:
            ues_tree = ESDecisionTreeClassifier(
                max_depth=None,
                min_samples_split=2,
                kappa=self.mean_estimated_train_noise,
                max_features=self.max_features,
            )
            ues_tree.fit(X_train, y_train)
            ues_depth = ues_tree.get_depth()
            if self.es_offset is not None:
                ues_depth = ues_depth + self.es_offset

        for bootstrap_sample_id in tqdm(
            bootstrap_sets_dict, desc="Training Trees", disable=self.disable_pb
        ):  # bootstrap_sets_dict: #
            X_bootstrap = bootstrap_sets_dict[bootstrap_sample_id]["boot"][:, :-1]
            y_bootstrap = bootstrap_sets_dict[bootstrap_sample_id]["boot"][
                :, -1
            ].reshape(-1)
            # Get Bootstrap true noise for analytical purposes
            if bootstrap_f_dict and self.kappa != "no_es":
                f_bootstrap = bootstrap_f_dict[bootstrap_sample_id]
                self.mean_var_true_bootstrap_noises.append(
                    np.mean(f_bootstrap * (1 - f_bootstrap))
                )
            else:
                self.mean_var_true_bootstrap_noises.append(np.nan)

            if (
                self.estimate_noise_before_sampling == True
            ):  # initiate same DT for every bootstrap sample
                if self.ues_time_track:
                    bootstrap_es_dt_classifier = DecisionTreeClassifier(
                        max_depth=ues_depth,
                        min_samples_split=2,
                        max_features=self.max_features,
                        random_state=self.random_state,
                    )
                elif self.kappa == "no_es":
                    bootstrap_es_dt_classifier = DecisionTreeClassifier(
                        max_depth=None,
                        min_samples_split=2,
                        max_features=self.max_features,
                        random_state=self.random_state,
                    )
                else:
                    bootstrap_es_dt_classifier = ESDecisionTreeClassifier(
                        max_depth=self.max_depth,
                        min_samples_split=2,
                        es_offset=self.es_offset,
                        kappa=self.mean_estimated_train_noise,
                        max_features=self.max_features,
                        random_state=self.random_state,
                        rf_train_mse=self.rf_train_mse,
                    )
                if bootstrap_f_dict:
                    self.estimated_kappa_train_bootstrap.append(
                        np.mean(f_bootstrap * (1 - f_bootstrap))
                    )  

            else:
                pass
                ## Set Kappa for ES on bootstrap samples
                # noise_estimator = noise_est.Estimator(X_bootstrap, y_bootstrap)
                # if self.kappa == "1nn":
                #     mean_estimated_boot_noise = noise_estimator.estimate_1NN()
                # elif self.kappa == "lasso":
                #     mean_estimated_boot_noise = noise_estimator.estimate_LASSO()
                # elif self.kappa == "mean_var":
                #     mean_estimated_boot_noise = self.mean_var_true_bootstrap_noises[-1]
                # elif self.kappa == "no_es":
                #     mean_estimated_boot_noise = np.nan
                # else:
                #     raise ValueError("Kappa not recognized")

                # self.estimated_kappa_train_bootstrap.append(mean_estimated_boot_noise)

                # bootstrap_es_dt_classifier = ESDecisionTreeClassifier(
                #     max_depth=self.max_depth,
                #     min_samples_split=2,
                #     kappa=self.estimated_kappa_train_bootstrap[-1],
                #     max_features=self.max_features,
                #     random_state=self.random_state,
                #     es_offset=self.es_offset,
                # )

            # fit a decision tree classifier to the current sample
            start_time = time.time()
            if self.rf_train_mse:
                bootstrap_es_dt_classifier.fit(
                    X_bootstrap,
                    y_bootstrap,
                    X_whole_rf=X_train,
                    y_whole_rf=y_train,
                )
            else:
                bootstrap_es_dt_classifier.fit(
                    X_bootstrap,
                    y_bootstrap,
                )
            end_time = time.time()
            # es_offset_depth = bootstrap_es_dt_classifier.get_depth() + self.es_offset
            # train scikit learn decision tree with same depth
            # if self.kappa == "no_es":
            #     bootstrap_sklearn_dt_es = DecisionTreeClassifier(max_depth=None, random_state=self.random_state, max_features=self.max_features)
            # elif self.two_step:
            #     bootstrap_sklearn_dt_es = model_builder.build_post_pruned_dt_clf(
            #         X_train=X_bootstrap, y_train=y_bootstrap, random_state=self.random_state, max_depth=es_offset_depth, n_cv_alpha=5, full_alpha_range=False, max_features=self.max_features)
            # else:
            #     bootstrap_sklearn_dt_es = DecisionTreeClassifier(max_depth=es_offset_depth, random_state=self.random_state, max_features=self.max_features)

            
            # bootstrap_sklearn_dt_es.fit(X_bootstrap, y_bootstrap)
            

            fit_duration = end_time - start_time
            self.tree_fit_durations_.append(fit_duration)

            # append the fitted model
            # self.estimators_.append(bootstrap_sklearn_dt_es)
            self.estimators_.append(bootstrap_es_dt_classifier)
        self.fit_duration = bootstrap_noise_duration + sum(self.tree_fit_durations_)

    # predict from the ensemble
    def predict(self, X, n_trees=None):
        if n_trees is None:
            n_trees = self.n_estimators
        # check we've fit the ensemble
        if not self.estimators_:
            print("You must train the ensemble before making predictions!")
            return None

        # loop through each fitted model
        predictions = []
        for model in self.estimators_[:n_trees]:
            # if model.stopping_index is None:
            #     breakpoint()
            # make predictions on the input X
            y_pred_labels_single_tree = model.predict(X)
            # append predictions to storage list
            predictions.append(y_pred_labels_single_tree.reshape(-1, 1))
        # regression-style averaging (of binary values) then rounding
        y_pred_labels_ensemble = np.round(
            np.mean(np.concatenate(predictions, axis=1), axis=1)
        ).astype(int)
        
        return y_pred_labels_ensemble

    def predict_proba(
        self, X, n_trees=None
    ):  # TODO: Adjust to kappa and depth
        if n_trees is None:
            n_trees = self.n_estimators
        # check we've fit the ensemble
        if not self.estimators_:
            print("You must train the ensemble before making predictions!")
            return None
        # loop through each fitted model
        predictions = []
        for model in self.estimators_[:n_trees]:
            # make predictions on the input X
            y_pred_prob_single_tree = model.predict_proba(X)[:, 1]
            # append predictions to storage list
            predictions.append(
                y_pred_prob_single_tree.reshape(-1, 1)
            )  # each entry is a column
        # compute the ensemble prediction
        y_pred_prob_ensemble = np.mean(
            np.concatenate(predictions, axis=1), axis=1
        )  # conc = column stack, axis=1 je zeile mean nehmen/über columns
        y_pred_prob_ensemble = np.column_stack(
            (1 - y_pred_prob_ensemble, y_pred_prob_ensemble)
        )
        # return the prediction
        return y_pred_prob_ensemble

    def get_depths(self):
        """
        Get the depths of the trees
        """
        return [tree.get_depth() for tree in self.estimators_]
    
    
    def get_mean_depth(self):
        """
        Get the mean depth of the trees
        """
        return np.round(np.mean([tree.get_depth() for tree in self.estimators_]), decimals=1)
    
    def get_sd_depth(self):
        """
        Get the standard deviation of the depths of the trees
        """
        return np.round(np.std([tree.get_depth() for tree in self.estimators_]), decimals=2)
    
    def get_median_depth(self):
        """
        Get the median depth of the trees
        """
        return int(np.median([tree.get_depth() for tree in self.estimators_]))

    def get_mean_n_leaves(self):
        """
        Get the mean number of leaf nodes
        """
        return np.round(np.mean([tree.get_n_leaves() for tree in self.estimators_]), decimals=1)
    
    def get_median_n_leaves(self):
        """
        Get the median number of leaf nodes
        """
        return int(np.median([tree.get_n_leaves() for tree in self.estimators_]))
    
    def get_sd_n_leaves(self):
        """
        Get the standard deviation of the number of leaf nodes
        """
        return np.round(np.std([tree.get_n_leaves() for tree in self.estimators_]), decimals=2)

    def get_mean_fit_duration(self):
        """
        Get the mean fit duration of the trees
        """
        return np.round(np.mean(self.tree_fit_durations_), decimals=6)
    
    def get_median_fit_duration(self):
        """
        Get the median fit duration of the trees
        """
        return np.round(np.median(self.tree_fit_durations_), decimals=6)
    
    def get_mean_noise_level(self):
        """
        Get the mean noise level of the trees
        """
        return np.mean(self.estimated_kappa_train_bootstrap)
    
    def get_sd_noise_level(self):
        """
        Get the standard deviation of the noise level of the trees
        """
        return np.std(self.estimated_kappa_train_bootstrap)
    
    
    
