import os
import sys
import time

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as scikit_rf

# Remove this line:
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Replace infrastructure imports with direct imports:
from ..algorithms.mseRF import mseRF
from ..algorithms.clean_dt import DecisionTreeLevelWise as custom_tree

from . import noise_level_estimator as noise_est
import numpy as np

from ..algorithms.ScikitRFTwoStep import RandomForestClassifier2Step
from ..algorithms import EsGlobalRF as ESGlobalRF

def build_post_pruned_dt_clf(
    X_train, y_train, random_state, max_depth=None, n_cv_alpha=5, full_alpha_range=False, max_features=None
):
    # Create base classifier with common parameters
    base_params = {
        "max_depth": max_depth,
        "min_samples_split": 2,
        "max_features": max_features,
        "random_state": random_state,
    }

    # Get path only once
    path = DecisionTreeClassifier(**base_params).cost_complexity_pruning_path(
        X_train, y_train
    )
    ccp_alphas = path.ccp_alphas

    # Remove last alpha as it typically results in an empty tree if full_alpha_range is False
    if not full_alpha_range:
        # dt = DecisionTreeClassifier(ccp_alpha=ccp_alphas[-1])
        # dt.fit(X_train, y_train)
        # dt.get_depth()
        ccp_alphas = ccp_alphas[:-1]

    # Only keep positive alphas if any exist, numerical underflow
    if len(ccp_alphas) > 0:
        ccp_alphas = ccp_alphas[ccp_alphas > 0]

    # If no valid alphas found, return unpruned tree
    if len(ccp_alphas) == 0:
        # esel = DecisionTreeClassifier(**base_params).fit(X_train, y_train)
        return DecisionTreeClassifier(**base_params).fit(X_train, y_train)

    # Grid search with the valid alphas
    grid_search = GridSearchCV(
        DecisionTreeClassifier(**base_params),
        param_grid={"ccp_alpha": ccp_alphas},
        cv=n_cv_alpha,
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def build_rf_clf(
    X_train,
    y_train,
    algorithm,
    kappa,
    n_estimators,
    max_features,
    estimate_noise_before_sampling,
    random_state,
    f_train=None,
    es_offset = None,
    two_step = False,
    rf_train_mse = False,
    vote_probability = True,
    ues_time_track = False,
):
    fit_duration = 0.0  # Initialize fit_duration
    if algorithm == "MD_scikit":
        rf = scikit_rf(
            n_estimators=n_estimators,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            max_features=max_features,  # either a number, None or "sqrt", 
            bootstrap=True,
            oob_score=False,
            ccp_alpha=0.0,
            max_samples=None,
            n_jobs=None,
            random_state=random_state,
        )
        start_time_rf = time.time()
        rf.fit(X_train, y_train)
        end_time_rf = time.time()
        fit_duration = end_time_rf - start_time_rf

    elif algorithm == "UES":
        if kappa == "mean_var":
            noise_est_time_start = time.time()
            mean_estimated_train_noise = np.mean(f_train * (1 - f_train))
            noise_est_time_end = time.time()
            noise_est_time = noise_est_time_end - noise_est_time_start
        elif kappa == "1nn":
            noise_estimator = noise_est.Estimator(X_train, y_train)
            noise_est_time_start = time.time()
            mean_estimated_train_noise = noise_estimator.estimate_1NN()
            noise_est_time_end = time.time()
            noise_est_time = noise_est_time_end - noise_est_time_start
        else:
            raise ValueError(f"Invalid kappa: {kappa}")

        tree = custom_tree(
            max_depth=None,
            min_samples_split=2,
            kappa=mean_estimated_train_noise,
            max_features=max_features,
            random_state=random_state,
        )
        tree.fit(X_train, y_train)
        es_depth = tree.get_depth()
        if es_offset is not None:
            es_depth = es_depth + es_offset
        es_depth = max(es_depth, 1)
        scikit_tree = DecisionTreeClassifier(
            max_depth=es_depth,
            min_samples_split=2,
            max_features=max_features,
            random_state=random_state,
        )
        single_tree_start_time = time.time()
        scikit_tree.fit(X_train, y_train)
        single_tree_end_time = time.time()
        single_tree_fit_duration = single_tree_end_time - single_tree_start_time

        rf = scikit_rf(
            n_estimators=n_estimators,
            criterion="gini",
            max_depth=es_depth,
            min_samples_split=2,
            max_features=max_features,
            bootstrap=True,
            oob_score=False,
            ccp_alpha=0.0,
            max_samples=None,
            n_jobs=None,
            random_state=random_state,
        )
        start_time_rf = time.time()
        rf.fit(X_train, y_train)
        end_time_rf = time.time()
        rf_fit_duration = end_time_rf - start_time_rf
        fit_duration = rf_fit_duration + single_tree_fit_duration + noise_est_time
    elif algorithm == "MD_custom":
        rf = ESGlobalRF.RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            max_features=max_features,
            estimate_noise_before_sampling=estimate_noise_before_sampling,
            kappa="no_es",
            random_state=random_state,
        )
        start_time_rf = time.time()
        rf.fit(X_train, y_train)
        end_time_rf = time.time()
        fit_duration = end_time_rf - start_time_rf
    elif algorithm == "IES":
        rf = ESGlobalRF.RandomForestClassifier(
            n_estimators=n_estimators,
            kappa=kappa,
            max_features=max_features,
            es_offset=es_offset,
            estimate_noise_before_sampling=estimate_noise_before_sampling,
            max_depth=None,
            random_state=random_state,
            ues_time_track=ues_time_track,
            rf_train_mse=rf_train_mse,
            two_step=two_step,
        )
        start_time_rf = time.time()
        rf.fit(X_train, y_train, f_train=f_train)
        end_time_rf = time.time()
        fit_duration = end_time_rf - start_time_rf
    elif algorithm == "CCP":
        rf = RandomForestClassifier2Step(
            n_estimators=n_estimators,
            max_features=max_features,
            apply_es=False,
            kappa="no_es",
            random_state=random_state,
        )
        start_time_rf = time.time() 
        rf.fit(X_train, y_train, f_train=f_train)
        end_time_rf = time.time()
        fit_duration = end_time_rf - start_time_rf
    elif algorithm == "UGES":
        rf = mseRF(
            n_estimators=n_estimators,
            random_state=random_state,
            vote_probability=vote_probability,
            es_offset=es_offset,
            max_features=max_features,
        )
        start_time_rf = time.time()
        rf = rf.fit(X_train, y_train, f_train=f_train)
        fit_duration = time.time() - start_time_rf
    elif algorithm == "MD_scikit_factor":
        rf = scikit_rf(
            n_estimators=n_estimators,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            max_features=max_features,  # This will be the sqrt_factor value
            bootstrap=True,
            oob_score=False,
            ccp_alpha=0.0,
            max_samples=None,
            n_jobs=None,
            random_state=random_state,
        )
        start_time_rf = time.time()
        rf.fit(X_train, y_train)
        end_time_rf = time.time()
        fit_duration = end_time_rf - start_time_rf
    elif algorithm == "MD_custom_factor":
        rf = ESGlobalRF.RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            max_features=max_features,  # This will be the sqrt_factor value
            estimate_noise_before_sampling=estimate_noise_before_sampling,
            kappa="no_es",
            random_state=random_state,
        )
        start_time_rf = time.time()
        rf.fit(X_train, y_train)
        end_time_rf = time.time()
        fit_duration = end_time_rf - start_time_rf
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return rf, fit_duration
