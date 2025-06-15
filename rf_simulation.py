# Paths for default/test run
ALG_PATH = "experiments/experiment_configs/algorithm_configs/es_rf_simulation_study/rf/new_structure/UES_RF_star_time.yaml"
DGP_PATH = "experiments/experiment_configs/simulated_data_configs/standard/circular_feature_dim_2_n_samples_2000_bernoulli_p_0.8.yaml"
ENV_PATH = "experiments/experiment_configs/env_setting_configs/rf_experiments/test.yaml"
# EXPERIMENT_NAME = "test"  # name of folder

import argparse
import csv
import os
import sys

from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    log_loss,
)
from tqdm import tqdm


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import infrastructure.config_reader as config_reader
import infrastructure.dgp.data_generation as data_generation
from infrastructure.model_builder import build_rf_clf
import infrastructure.noise_level_estimator as noise_est
from infrastructure.algorithms.EsGlobalRF import RandomForestClassifier as EsGlobalRF
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


# for mc_iteration in range(MC_ITERATIONS):
def run_mc_iteration(mc_iteration, dgp_config, algo_config, env_config):
    # print(f"MC iteration {mc_iteration}")

    # Set Hyperparameters for Experiment
    # RANDOM_SEED is fixed for estimators and iterating for data_generation
    RANDOM_STATE = env_config.get("random_state")  # for model building and data
    N_ESTIMATORS = env_config.get("n_estimators")

    data_random_state = RANDOM_STATE + mc_iteration
    # Generate data
    X_train, X_test, y_train, y_test, f_train, f_test = (
        data_generation.generate_X_y_f_classification(
            random_state=data_random_state,
            n_ticks_per_ax_meshgrid=dgp_config.get(
                "n_ticks_per_ax_meshgrid"
            ),  # for 2 dim cases, n_samples
            dgp_name=dgp_config.get("dgp_name"),  # indicating which DGP to use
            bernoulli_p=dgp_config.get("bernoulli_p"),  # for 2 dim cases
            n_samples=dgp_config.get(
                "n_samples"
            ),  # size of dateset (train + test), in additive cases!
            feature_dim=dgp_config.get("feature_dim"),
        )
    )

    rf, ensemble_fit_duration = build_rf_clf(
        X_train=X_train,
        y_train=y_train,
        f_train=f_train,  # No true function values for empirical data
        algorithm=algo_config.get("algorithm"),
        max_features=algo_config.get("max_features"),
        es_offset=algo_config.get("es_offset"),
        rf_train_mse=algo_config.get("rf_train_mse"),
        kappa=algo_config.get("kappa"),
        n_estimators=N_ESTIMATORS,
        ues_time_track=algo_config.get("ues_time_track"),
        vote_probability=algo_config.get("vote_probability", False),
        estimate_noise_before_sampling=algo_config.get(
            "estimate_noise_before_sampling"
        ),
        random_state=RANDOM_STATE,
    )

    train_accuracies_per_n_trees = []
    test_accuracies_per_n_trees = []
    for n_estimators in range(1, N_ESTIMATORS + 1):
        # Collect predictions from the first n_estimators trees for train and test sets
        y_pred_label_train = (
            np.mean(
                np.stack(
                    [
                        tree.predict_proba(X_train)[:, 1]
                        for tree in rf.estimators_[:n_estimators]
                    ]
                ),
                axis=0,
            )
            >= 0.5
        )
        y_pred_label_test = (
            np.mean(
                np.stack(
                    [
                        tree.predict_proba(X_test)[:, 1]
                        for tree in rf.estimators_[:n_estimators]
                    ]
                ),
                axis=0,
            )
            >= 0.5
        )

        # Calculate accuracy for train and test sets
        train_accuracy = np.mean(y_pred_label_train == y_train)
        test_accuracy = np.mean(y_pred_label_test == y_test)
        # Append accuracies to the lists
        train_accuracies_per_n_trees.append(train_accuracy)
        test_accuracies_per_n_trees.append(test_accuracy)

    train_accuracy = train_accuracies_per_n_trees[-1]
    test_accuracy = test_accuracies_per_n_trees[-1]

    # Track Performance of individual trees and Ensemble
    train_accuracies = [
        np.mean(estimator.predict(X_train) == y_train) for estimator in rf.estimators_
    ]
    mean_train_accuracy = np.mean(train_accuracies)
    median_train_accuracy = np.median(train_accuracies)
    sd_train_accuracy = np.std(train_accuracies)

    test_accuracies = [
        np.mean(estimator.predict(X_test) == y_test) for estimator in rf.estimators_
    ]
    mean_test_accuracy = np.mean(test_accuracies)
    median_test_accuracy = np.median(test_accuracies)
    sd_test_accuracy = np.std(test_accuracies)

    # Track Tree Structure
    mean_tree_depth = np.mean([tree.get_depth() for tree in rf.estimators_])
    sd_tree_depth = np.std([tree.get_depth() for tree in rf.estimators_])
    median_tree_depth = np.median([tree.get_depth() for tree in rf.estimators_])
    mean_n_leaves = np.mean([tree.get_n_leaves() for tree in rf.estimators_])
    median_n_leaves = np.median([tree.get_n_leaves() for tree in rf.estimators_])
    sd_n_leaves = np.std([tree.get_n_leaves() for tree in rf.estimators_])

    # track fitting time
    if (
        algo_config.get("algorithm") == "MD_custom"
        or algo_config.get("algorithm") == "IES"
    ):
        mean_tree_fit = rf.get_mean_fit_duration()
        median_tree_fit = rf.get_median_fit_duration()
    else:
        mean_tree_fit = np.nan
        median_tree_fit = np.nan

    if algo_config.get("algorithm") == "IES" and algo_config.get("kappa") == "1nn":
        # Track Noise Related Stuff
        noise_estimator = noise_est.Estimator(X_train, y_train)
        noise_est_time_start = time.time()
        _ = noise_estimator.estimate_1NN()
        noise_est_time_end = time.time()
        noise_est_time = noise_est_time_end - noise_est_time_start
    else:
        noise_est_time = np.nan

    # Track Noise Related Stuff
    true_whole_set_noise_level = np.mean(f_train * (1 - f_train))

    if algo_config.get("algorithm") == "IES" and algo_config.get("kappa") == "mean_var":
        # Noise Estimation related Stuff
        # In case of before sampling
        bootstrap_mean_noise_level = rf.get_mean_noise_level()
        bootstrap_sd_noise_level = rf.get_sd_noise_level()
        (
            whole_set_noise_true_vs_estimate,
            avg_true_bootstap_vs_estimate,
        ) = (np.nan, np.nan)
    else:
        bootstrap_mean_noise_level = np.nan
        bootstrap_sd_noise_level = np.nan
        (
            whole_set_noise_true_vs_estimate,
            avg_true_bootstap_vs_estimate,
        ) = (np.nan, np.nan)

    # Calculate final predictions for full ensemble
    y_pred_proba_train = np.mean(
        np.stack([tree.predict_proba(X_train)[:, 1] for tree in rf.estimators_]), axis=0
    )
    y_pred_label_train = y_pred_proba_train >= 0.5

    y_pred_proba_test = np.mean(
        np.stack([tree.predict_proba(X_test)[:, 1] for tree in rf.estimators_]), axis=0
    )
    y_pred_label_test = y_pred_proba_test >= 0.5

    # Calculate additional metrics
    test_f1 = f1_score(y_test, y_pred_label_test)

    # Ensure probabilities are clipped to avoid log(0)
    y_pred_proba_test_clipped = np.clip(y_pred_proba_test, 1e-15, 1 - 1e-15)
    test_log_loss = log_loss(y_test, y_pred_proba_test_clipped)

    test_mcc = matthews_corrcoef(y_test, y_pred_label_test)

    return (
        train_accuracy,
        test_accuracy,
        ensemble_fit_duration,
        mean_tree_depth,
        sd_tree_depth,
        median_tree_depth,
        mean_n_leaves,
        sd_n_leaves,
        median_n_leaves,
        mean_tree_fit,
        median_tree_fit,
        sd_train_accuracy,
        mean_train_accuracy,
        median_train_accuracy,
        sd_test_accuracy,
        mean_test_accuracy,
        median_test_accuracy,
        true_whole_set_noise_level,
        whole_set_noise_true_vs_estimate,
        avg_true_bootstap_vs_estimate,
        train_accuracies_per_n_trees,
        test_accuracies_per_n_trees,
        test_f1,
        test_log_loss,
        test_mcc,
        noise_est_time,
        bootstrap_mean_noise_level,
        bootstrap_sd_noise_level,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm_config",
        "-alg_cfg",
        type=str,
        default=ALG_PATH,
        help="Path to a yaml file containing the hyperparameters for the algorithm",
        required=False,
    )
    parser.add_argument(
        "--dgp_config",
        "-dgp_cfg",
        type=str,
        default=DGP_PATH,
        help="Path to a yaml file containing the hyperparameters for the DGP",
        required=False,
    )
    parser.add_argument(
        "--env_config",
        "-env_cfg",
        type=str,
        default=ENV_PATH,
        help="Path to yaml where random state, n_estimators, mc_iterations are stored",
        required=False,
    )
    args = parser.parse_args()
    env_config = config_reader.load_config(args.env_config)
    # Set Up data with config file
    dgp_config = config_reader.load_config(args.dgp_config)
    algo_config = config_reader.load_config(args.algorithm_config)
    # Get algo_config filename at the end of path
    algo_config_filename = os.path.basename(args.algorithm_config)

    # Set Hyperparameters for Experiment (here for documentation)
    # Has to be set outside and inside of function call
    MC_ITERATIONS = env_config.get("mc_iter")
    N_ESTIMATORS = env_config.get("n_estimators")
    RANDOM_STATE = env_config.get("random_state")

    # Initialize lists to store all values for mean and median calculation
    all_train_accuracies = []
    all_test_accuracies = []
    all_ensemble_fit_durations = []
    all_mean_tree_fit_durations = []
    all_median_tree_fit_durations = []
    all_avg_depths = []
    all_sd_depths = []
    all_median_depths = []
    all_avg_n_leaves = []
    all_sd_n_leaves = []
    all_median_n_leaves = []
    all_sd_train_accuracies = []
    all_mean_train_accuracies = []
    all_median_train_accuracies = []
    all_sd_test_accuracies = []
    all_mean_test_accuracies = []
    all_median_test_accuracies = []
    all_true_noise_levels = []
    all_noise_true_vs_estimates = []
    all_avg_noise_b_true_vs_estimates = []
    all_train_accuracies_per_n_trees = []
    all_test_accuracies_per_n_trees = []
    all_test_f1 = []
    all_test_log_loss = []
    all_test_mcc = []
    all_noise_est_times = []
    bootstrap_mean_noise_levels = []
    bootstrap_sd_noise_levels = []

    # Run MC iterations in parallel
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_mc_iteration, mc_iteration, dgp_config, algo_config, env_config
            )
            for mc_iteration in range(MC_ITERATIONS)
        ]

        for future in tqdm(
            as_completed(futures), total=MC_ITERATIONS, desc="MC Iterations"
        ):
            (
                train_accuracy,
                test_accuracy,
                ensemble_fit_duration,
                mean_tree_depth,
                sd_tree_depth,
                median_tree_depth,
                mean_n_leaves,
                sd_n_leaves,
                median_n_leaves,
                mean_tree_fit,
                median_tree_fit,
                sd_train_accuracy,
                mean_train_accuracy,
                median_train_accuracy,
                sd_test_accuracy,
                mean_test_accuracy,
                median_test_accuracy,
                true_whole_set_noise_level,
                whole_set_noise_true_vs_estimate,
                avg_true_bootstap_vs_estimate,
                train_accuracies_per_n_trees,
                test_accuracies_per_n_trees,
                test_f1,
                test_log_loss,
                test_mcc,
                noise_est_time,
                bootstrap_mean_noise_level,
                bootstrap_sd_noise_level,
            ) = future.result()

            # Store values for mean and median calculation
            all_train_accuracies.append(train_accuracy)
            all_test_accuracies.append(test_accuracy)
            all_ensemble_fit_durations.append(ensemble_fit_duration)
            all_mean_tree_fit_durations.append(mean_tree_fit)
            all_median_tree_fit_durations.append(median_tree_fit)
            all_avg_depths.append(mean_tree_depth)
            all_sd_depths.append(sd_tree_depth)
            all_median_depths.append(median_tree_depth)
            all_avg_n_leaves.append(mean_n_leaves)
            all_sd_n_leaves.append(sd_n_leaves)
            all_median_n_leaves.append(median_n_leaves)
            all_sd_train_accuracies.append(sd_train_accuracy)
            all_mean_train_accuracies.append(mean_train_accuracy)
            all_median_train_accuracies.append(median_train_accuracy)
            all_sd_test_accuracies.append(sd_test_accuracy)
            all_mean_test_accuracies.append(mean_test_accuracy)
            all_median_test_accuracies.append(median_test_accuracy)
            all_true_noise_levels.append(true_whole_set_noise_level)
            all_noise_true_vs_estimates.append(whole_set_noise_true_vs_estimate)
            all_avg_noise_b_true_vs_estimates.append(avg_true_bootstap_vs_estimate)
            all_train_accuracies_per_n_trees.append(train_accuracies_per_n_trees)
            all_test_accuracies_per_n_trees.append(test_accuracies_per_n_trees)
            all_test_f1.append(test_f1)
            all_test_log_loss.append(test_log_loss)
            all_test_mcc.append(test_mcc)
            all_noise_est_times.append(noise_est_time)
            bootstrap_mean_noise_levels.append(bootstrap_mean_noise_level)
            bootstrap_sd_noise_levels.append(bootstrap_sd_noise_level)

    # Calculate means and medians
    metrics = {
        "train_acc (mean)": np.mean(all_train_accuracies),
        "train_acc (median)": np.median(all_train_accuracies),
        "test_acc (mean)": np.mean(all_test_accuracies),
        "test_acc (median)": np.median(all_test_accuracies),
        # "fit_duration (mean)": np.mean(all_ensemble_fit_durations),
        # "fit_duration (median)": np.median(all_ensemble_fit_durations),
        "mean_tree_fit_duration (mean)": np.mean(all_mean_tree_fit_durations),
        "mean_tree_fit_duration (median)": np.median(all_mean_tree_fit_durations),
        "median_tree_fit_duration (mean)": np.mean(all_median_tree_fit_durations),
        "median_tree_fit_duration (median)": np.median(all_median_tree_fit_durations),
        "mean_n_leaves (mean)": np.mean(all_avg_n_leaves),
        "mean_n_leaves (median)": np.median(all_avg_n_leaves),
        "sd_n_leaves (mean)": np.mean(all_sd_n_leaves),
        "sd_n_leaves (median)": np.median(all_sd_n_leaves),
        "median_n_leaves (mean)": np.mean(all_median_n_leaves),
        "median_n_leaves (median)": np.median(all_median_n_leaves),
        "mean_depth (mean)": np.mean(all_avg_depths),
        "mean_depth (median)": np.median(all_avg_depths),
        "sd_depth (mean)": np.mean(all_sd_depths),
        "sd_depth (median)": np.median(all_sd_depths),
        "median_depth (mean)": np.mean(all_median_depths),
        "median_depth (median)": np.median(all_median_depths),
        "sd_train_acc (mean)": np.mean(all_sd_train_accuracies),
        "sd_train_acc (median)": np.median(all_sd_train_accuracies),
        "mean_train_acc (mean)": np.mean(all_mean_train_accuracies),
        "mean_train_acc (median)": np.median(all_mean_train_accuracies),
        "median_train_acc (mean)": np.mean(all_median_train_accuracies),
        "median_train_acc (median)": np.median(all_median_train_accuracies),
        "sd_test_acc (mean)": np.mean(all_sd_test_accuracies),
        "sd_test_acc (median)": np.median(all_sd_test_accuracies),
        "mean_test_acc (mean)": np.mean(all_mean_test_accuracies),
        "mean_test_acc (median)": np.median(all_mean_test_accuracies),
        "median_test_acc (mean)": np.mean(all_median_test_accuracies),
        "median_test_acc (median)": np.median(all_median_test_accuracies),
        "noise_level (mean)": np.mean(all_true_noise_levels),
        "noise_level (median)": np.median(all_true_noise_levels),
        "noise_true_vs_estimate (mean)": np.mean(all_noise_true_vs_estimates),
        "noise_true_vs_estimate (median)": np.median(all_noise_true_vs_estimates),
        "avg_noise_b_true_vs_estimate (mean)": np.mean(
            all_avg_noise_b_true_vs_estimates
        ),
        "avg_noise_b_true_vs_estimate (median)": np.median(
            all_avg_noise_b_true_vs_estimates
        ),
        # New metrics - only test scores
        "f1_test (mean)": np.mean(all_test_f1),
        "f1_test (median)": np.median(all_test_f1),
        "log_loss_test (mean)": np.mean(all_test_log_loss),
        "log_loss_test (median)": np.median(all_test_log_loss),
        "mcc_test (mean)": np.mean(all_test_mcc),
        "mcc_test (median)": np.median(all_test_mcc),
        "noise_est_time (mean)": np.mean(all_noise_est_times),
        "noise_est_time (median)": np.median(all_noise_est_times),
    }

    # Calculate mean accuracies per n_trees across all MC iterations
    mc_train_ensemble_accuracies_per_n_trees = np.mean(
        all_train_accuracies_per_n_trees, axis=0
    )
    mc_test_ensemble_accuracies_per_n_trees = np.mean(
        all_test_accuracies_per_n_trees, axis=0
    )

    # If you also want medians (optional)
    mc_train_ensemble_accuracies_per_n_trees_median = np.median(
        all_train_accuracies_per_n_trees, axis=0
    )
    mc_test_ensemble_accuracies_per_n_trees_median = np.median(
        all_test_accuracies_per_n_trees, axis=0
    )

    # Store results in a table
    train_accuracies_per_n_trees_dict = {
        f"train_acc_{n_trees + 1}_trees (mean)": accuracy
        for n_trees, accuracy in enumerate(mc_train_ensemble_accuracies_per_n_trees)
    }
    train_accuracies_per_n_trees_dict.update(
        {
            f"train_acc_{n_trees + 1}_trees (median)": accuracy
            for n_trees, accuracy in enumerate(
                mc_train_ensemble_accuracies_per_n_trees_median
            )
        }
    )

    test_accuracies_per_n_trees_dict = {
        f"test_acc_{n_trees + 1}_trees (mean)": accuracy
        for n_trees, accuracy in enumerate(mc_test_ensemble_accuracies_per_n_trees)
    }
    test_accuracies_per_n_trees_dict.update(
        {
            f"test_acc_{n_trees + 1}_trees (median)": accuracy
            for n_trees, accuracy in enumerate(
                mc_test_ensemble_accuracies_per_n_trees_median
            )
        }
    )

    results = {
        "algo_config": algo_config_filename,
        "dgp_config_folder": args.dgp_config.split("/")[-2],
        "algorithm_name": algo_config.get(
            "algorithm_name", algo_config.get("algorithm")
        ),
        "algorithm": (
            algo_config.get("algorithm") + "*"
            if algo_config.get("kappa") == "mean_var"
            else algo_config.get("algorithm")
        ),
        "method": ("RF" if algo_config.get("max_features") == "sqrt" else "Bagg"),
        "dataset": dgp_config.get("dgp_name"),
        "feature_dim": dgp_config.get("feature_dim"),
        "n_samples": dgp_config.get("n_samples"),
        "mc_iterations": MC_ITERATIONS,
        "n_estimators": N_ESTIMATORS,
        "max_depth": algo_config.get("max_depth"),
        # "depth_type": (
        #     "same" if algo_config.get("algorithm")[:6] == "scikit" else "individual"
        # ),
        "max_features": (algo_config.get("max_features")),
        "es_offset": algo_config.get("es_offset"),
        "estimate_noise_before_sampling": algo_config.get(
            "estimate_noise_before_sampling", True
        ),
        "rf_random_state": RANDOM_STATE,
        "kappa": algo_config.get("kappa"),  # TODO: not true for Scikit es star
        "ensemble_fit_duration (mean)": np.mean(all_ensemble_fit_durations),
        "ensemble_fit_duration (median)": np.median(all_ensemble_fit_durations),
        "bootstrap_mean_noise_level (mean)": (
            np.mean(bootstrap_mean_noise_levels)
            if bootstrap_mean_noise_levels
            else np.nan
        ),
        "bootstrap_sd_noise_level (median)": (
            np.mean(bootstrap_sd_noise_levels) if bootstrap_sd_noise_levels else np.nan
        ),
    }  # algo config auch rein!

    results.update(metrics)
    results.update(train_accuracies_per_n_trees_dict)
    results.update(test_accuracies_per_n_trees_dict)
    results = {
        key: (
            value
            if "fit_duration" or "noise_level" in key
            else f"{round(value, 2)}" if isinstance(value, float) else value
        )
        for key, value in results.items()
    }

    ## Store Results
    # Check if directory "experiment_results" exists, if not create it
    folder_name = "_".join(
        [
            f"{key}_{value}" if key != "dgp_name" else f"{value}"
            for key, value in dgp_config.items()
        ]
    )  # e.g. "additive_model_feature_dim_5_n_samples_1000"
    path_name = (
        "experiments/experiment_raw_results/rf_es_simulation_study/"
        # + args.experiment_name
        # + "/"
        + dgp_config.get("dgp_name")
        + "/"
        + folder_name
    )  # e.g. "experiments/experiment_raw_results/rf_es_simulation_study/additive_model_feature_dim_5_n_samples_1000"
    if not os.path.exists(path_name):
        os.makedirs(path_name)

    file_name = (
        "_".join(
            [
                f"{key}_{value}" if key != "algorithm" else f"{value}"
                for key, value in algo_config.items()
            ]
        )
        + ".csv"
    )  # configurations as filename

    file_path = os.path.join(path_name, file_name)
    # Save results in a csv file
    with open(file_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results.keys())
        # Write the header (keys of the dictionary)
        writer.writeheader()
        # Write the row (values of the dictionary)
        writer.writerow(results)


if __name__ == "__main__":
    main()