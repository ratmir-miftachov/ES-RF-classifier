import pandas as pd
import numpy as np
import math
from openml import datasets
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, log_loss
from joblib import Parallel, delayed
import os
import sys


from model_builder import build_rf_clf

# Random seed for reproducibility
RANDOM_SEED = 7
N_ESTIMATORS = 40
N_ITERATIONS = 96  # Number of Monte Carlo iterations
N_JOBS = -1  # für alle Kerne -1

# Define dataset names instead of IDs
DATASETS = [
    "Banknote",
    "SA Heart",
    "Pima Indians",
    "Haberman",
    "Ozone",
    "Spam",
    "Wisc. Breast Cancer",
]


def load_dataset(dataset_name):
    if dataset_name == "Pima Indians":
        dataset = datasets.get_dataset(43582)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        X = X.to_numpy()
        y = y.to_numpy()
    elif dataset_name == "Haberman":
        dataset = datasets.get_dataset(43)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        X = X.astype(float).to_numpy()
        y = (y.astype(int) == 2).astype(int).to_numpy()
    elif dataset_name == "Ozone":
        dataset = fetch_openml(data_id=1487, as_frame=True)
        X = dataset.data.to_numpy()
        y = (dataset.target == "1").astype(int).to_numpy()
    elif dataset_name == "SA Heart":
        dataset = datasets.get_dataset(1498)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        X = X.astype(float).to_numpy()
        y = (y == "2").astype(int).to_numpy()
    elif dataset_name == "Spam":
        dataset = datasets.get_dataset(44)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        X = X.astype(float).to_numpy()
        y = y.astype(int).to_numpy()
    elif dataset_name == "Wisc. Breast Cancer":
        dataset = datasets.get_dataset(15)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        X = X.to_numpy()
        y = (y == "malignant").astype(int).to_numpy()
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]
    elif dataset_name == "Banknote":
        dataset = datasets.get_dataset(43466)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        X = X.to_numpy()
        y = y.to_numpy()
    # Check if y contains only 0 and 1
    # Verify that y only contains 0 and 1
    unique_values = np.unique(y)
    if not np.array_equal(unique_values, np.array([0, 1])):
        raise ValueError(
            f"Dataset {dataset_name} contains labels other than 0 and 1: {unique_values}"
        )
    return X, y


# Define algorithm configurations
ALGORITHM_CONFIGS = [
    {
        "algorithm": "FMSE",
        "vote_probability": False,
        "algorithm_name": "FMSE",
        "es_offset": 0,
    },
    # {
    #     "algorithm_name": "ICCP",
    #     "algorithm": "CCP",
    #     "kappa": "no_es",
    #     "max_features": "sqrt",
    # },
    {
        "algorithm": "UES",
        "kappa": "1nn",
        "max_features": "sqrt",
        "estimate_noise_before_sampling": True,
        "es_offset": 0,
    },
    {
        "algorithm_name": "IGES",
        "algorithm": "IES",
        "kappa": "1nn",
        "max_features": "sqrt",
        "estimate_noise_before_sampling": True,
        "es_offset": 0,
        "rf_train_mse": True,
    },
    {
        "algorithm_name": "IGES_2",
        "algorithm": "IES",
        "kappa": "1nn",
        "max_features": "sqrt_factor",  # This will be calculated dynamically
        "estimate_noise_before_sampling": True,
        "es_offset": 0,
        "rf_train_mse": True,
    },
    {
        "algorithm_name": "ILES",
        "algorithm": "IES",
        "kappa": "1nn",
        "max_features": "sqrt",
        "estimate_noise_before_sampling": True,
        "es_offset": 0,
    },
    {
        "algorithm_name": "ILES_2",
        "algorithm": "IES", 
        "kappa": "1nn",
        "max_features": "sqrt_factor",  # This will be calculated dynamically
        "estimate_noise_before_sampling": True,
        "es_offset": 0,
    },
    {
        "algorithm": "MD_scikit",
        "max_features": "sqrt",
    },
    {
        "algorithm": "MD_custom",
        "kappa": "no_es",
        "max_features": "sqrt",
        "estimate_noise_before_sampling": True,
        "es_offset": 0,
    }
]


def run_single_dataset(dataset_info, seed):
    """Run a single experiment with the specified random seed.

    Args:
        dataset_info: Tuple containing (dataset_name, X, y)
        seed: Random seed for the experiment
    """
    dataset_name, X, y = dataset_info

    # Convert to binary classification if needed
    if len(np.unique(y)) > 2:
        # For multiclass, convert to binary by taking most frequent class vs others
        most_frequent_class = np.argmax(np.bincount(y.astype(int)))
        y = (y == most_frequent_class).astype(int)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=seed
    )

    results = []

    # Track dataset characteristics
    dataset_results = {
        "dataset": dataset_name,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "class_1_fraction": np.mean(y),
    }

    # Run each algorithm configuration
    for algo_config in ALGORITHM_CONFIGS:
        result = dataset_results.copy()
        result.update(algo_config)  # Add algorithm configuration to result

        # Handle dynamic max_features calculation
        max_features = algo_config.get("max_features")
        if max_features == "sqrt_factor": #sqrt(d)*factor
            max_features = math.ceil(math.sqrt(X_train.shape[1]) * 2)

        # Build the random forest classifier
        rf, ensemble_fit_duration = build_rf_clf(
            X_train=X_train,
            y_train=y_train,
            f_train=None,  # No true function values for empirical data
            algorithm=algo_config.get("algorithm"),
            max_features=max_features,
            es_offset=algo_config.get("es_offset"),
            rf_train_mse=algo_config.get("rf_train_mse"),
            kappa=algo_config.get("kappa"),
            n_estimators=N_ESTIMATORS,
            vote_probability=algo_config.get("vote_probability"),
            estimate_noise_before_sampling=algo_config.get(
                "estimate_noise_before_sampling"
            ),
            random_state=seed,
        )

        # Get predictions
        y_pred_train = rf.predict(X_train)
        y_pred_test = rf.predict(X_test)

        # For log loss, we need probability estimates
        try:
            y_pred_prob_test = rf.predict_proba(X_test)[:, 1]
        except:
            # Some RF implementations might not support predict_proba
            y_pred_prob_test = y_pred_test

        # Calculate metrics
        #train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_mcc = matthews_corrcoef(y_test, y_pred_test)
        #test_f1 = f1_score(y_test, y_pred_test, average="binary")

        # Calculate log loss if probabilities are available
        #if np.array_equal(y_pred_prob_test, y_pred_test):
        #    log_loss_test = np.nan
        #else:
        #    log_loss_test = log_loss(y_test, y_pred_prob_test)

        # Track tree structure metrics
        try:
            # Calculate mean tree metrics
            mean_tree_depth = np.mean([tree.get_depth() for tree in rf.estimators_])
            #sd_tree_depth = np.std([tree.get_depth() for tree in rf.estimators_])
            median_tree_depth = np.median([tree.get_depth() for tree in rf.estimators_])
            mean_n_leaves = np.mean([tree.get_n_leaves() for tree in rf.estimators_])
            median_n_leaves = np.median(
                [tree.get_n_leaves() for tree in rf.estimators_]
            )
            #sd_n_leaves = np.std([tree.get_n_leaves() for tree in rf.estimators_])

            # Track fitting time
            if (
                algo_config.get("algorithm") == "MD_custom"
                or algo_config.get("algorithm") == "IES"
                or algo_config.get("algorithm") == "IES*"
            ):
                mean_tree_fit_duration = rf.get_mean_fit_duration()
                median_tree_fit_duration = rf.get_median_fit_duration()
            else:
                mean_tree_fit_duration = np.nan
                median_tree_fit_duration = np.nan
        except:
            # Fallback if tree structure metrics are not available
            mean_tree_depth = median_tree_depth = np.nan
            mean_n_leaves = median_n_leaves = np.nan
            mean_tree_fit_duration = median_tree_fit_duration = np.nan

        # Compile results
        result.update(
            {
                "algorithm": algo_config.get("algorithm"),
                "algorithm_name": algo_config.get(
                    "algorithm_name", algo_config.get("algorithm")
                ),
                #"train_acc": train_acc,
                "test_acc": test_acc,
                "mcc_test": test_mcc,
                #"f1_test": test_f1,
                #"log_loss_test": log_loss_test,
                "mean_depth": mean_tree_depth,
                #"median_depth": median_tree_depth,
                "mean_n_leaves": mean_n_leaves,
                #"median_n_leaves": median_n_leaves,
                #"ensemble_fit_duration": ensemble_fit_duration,
                "mean_tree_fit_duration": mean_tree_fit_duration,
                #"median_tree_fit_duration": median_tree_fit_duration,
                #"seed": seed,
                #"n_estimators": N_ESTIMATORS,
            }
        )

        results.append(result)

    return results


def main():
    # Create output directory if it doesn't exist
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    # Run each dataset in parallel across multiple iterations
    for dataset_name in DATASETS:
        print(f"Processing dataset: {dataset_name}")

        # Load the dataset
        X, y = load_dataset(dataset_name)

        # Run Monte Carlo iterations in parallel
        dataset_results = Parallel(n_jobs=N_JOBS, verbose=1)(
            delayed(run_single_dataset)((dataset_name, X, y), seed)
            for seed in range(N_ITERATIONS)
        )

        # Flatten results
        dataset_results = [item for sublist in dataset_results for item in sublist]
        all_results.extend(dataset_results)

    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)

    # Save raw results

    # Calculate mean and median aggregates
    agg_results = []

    for dataset in df_results["dataset"].unique():
        for algorithm in df_results["algorithm_name"].unique():
            dataset_algo_df = df_results[
                (df_results["dataset"] == dataset)
                & (df_results["algorithm_name"] == algorithm)
            ]

            if dataset_algo_df.empty:
                continue

            # Get first row but only keep non-metric data
            base_row = dataset_algo_df.iloc[0]

            # Create a new result with only non-metric data
            aggregate_result = {
                "dataset": base_row["dataset"],
                #"n_samples": base_row["n_samples"],
                #"n_features": base_row["n_features"],
                #"class_1_fraction": base_row["class_1_fraction"],
                #"algorithm": base_row["algorithm"],
                "algorithm_name": base_row["algorithm_name"],
                #"kappa": base_row.get("kappa", None),
                #"max_features": base_row.get("max_features", None),
                #"estimate_noise_before_sampling": base_row.get(
                #    "estimate_noise_before_sampling", False
                #),
                #"es_offset": base_row.get("es_offset", 0),
                #"n_estimators": N_ESTIMATORS,
                #"n_iterations": N_ITERATIONS,
            }

            # Calculate mean and median for each metric
            metrics = [
                #"train_acc",
                "test_acc",
                #"mcc_test",
                #"f1_test",
                #"log_loss_test",
                "mean_depth",
                #"median_depth",
                #"sd_depth",
                "mean_n_leaves",
                #"median_n_leaves",
                #"sd_n_leaves",
                "ensemble_fit_duration",
                #"mean_tree_fit_duration",
                #"median_tree_fit_duration",
            ]

            for metric in metrics:
                if metric in dataset_algo_df.columns:
                    values = dataset_algo_df[metric].dropna()
                    if not values.empty:
                        #aggregate_result[f"{metric} (mean)"] = values.mean()
                        aggregate_result[f"{metric} (median)"] = values.median()

            agg_results.append(aggregate_result)

    # Convert to DataFrame
    df_agg_results = pd.DataFrame(agg_results)

    # Round different columns to different decimal places
    # Round depth and n_leaves columns to 1 decimal place
    depth_cols = [col for col in df_agg_results.columns if "depth" in col]
    leaves_cols = [col for col in df_agg_results.columns if "leaves" in col]
    if depth_cols or leaves_cols:
        df_agg_results[depth_cols + leaves_cols] = df_agg_results[depth_cols + leaves_cols].round(1)

    # Round fit_time/duration columns to 3 decimal places
    time_cols = [col for col in df_agg_results.columns if "duration" in col or "fit_time" in col]
    if time_cols:
        df_agg_results[time_cols] = df_agg_results[time_cols].round(3)

    # Round all other numeric columns to 2 decimal places
    numeric_cols = df_agg_results.select_dtypes(include=[np.number]).columns
    other_cols = [col for col in numeric_cols if col not in depth_cols + leaves_cols + time_cols]
    if other_cols:
        df_agg_results[other_cols] = df_agg_results[other_cols].round(2)

    # Save aggregated results
    output_path = os.path.join(output_dir, "rf_empirical_study.csv")
    df_agg_results.to_csv(output_path, index=False)
    print(f"Aggregated results saved to {output_path}")


if __name__ == "__main__":
    main()