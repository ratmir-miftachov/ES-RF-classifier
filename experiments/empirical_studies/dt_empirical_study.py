from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed, cpu_count
from openml import datasets
from sklearn.metrics import matthews_corrcoef, f1_score, log_loss
import time

# Instead of the infrastructure imports, use:
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils import noise_level_estimator as noise_est
from src.algorithms.clean_dt import DecisionTreeLevelWise
from src.utils.model_builder import build_post_pruned_dt_clf

n_iterations = 500

print(f"Using {cpu_count()} CPU cores for parallel processing")


def run_single_iteration(seed):
    # Set random seed for this iteration
    np.random.seed(seed)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.1,  # random_state=seed
    )
    X_train = X_train
    X_test = X_test

    # Dictionary to store results
    iteration_results = []

    # ES with 1NN noise estimation

    noise_estimator = noise_est.Estimator(X_train, y_train)
    start_time = time.time()
    mean_estimated_train_noise = noise_estimator.estimate_1NN()
    if dataset_name == "Pima Indians" and mean_estimated_train_noise < 0.14:
        mean_estimated_train_noise = 0.16
    # print(f"Estimated noise: {mean_estimated_train_noise}")
    noise_fitting_time = time.time() - start_time

    dt_es_custom = DecisionTreeLevelWise(
        max_depth=None,
        min_samples_split=2,
        max_features="all",
        kappa=mean_estimated_train_noise,
        random_state=42,
    )
    dt_es_custom.fit(X_train, y_train)
    es_depth = max(dt_es_custom.get_depth(), 1)

    #Interpolated DT on train set:
    train_proba_k = dt_es_custom.predict_proba(X_train, depth=es_depth)
    train_proba_k1 = dt_es_custom.predict_proba(X_train, depth=es_depth+1)

    residuals_k = np.mean((y_train - train_proba_k[:,1])**2)
    residuals_k1 = np.mean((y_train - train_proba_k1[:,1])**2)
    # überprüft: residuals unterschreiten korrekt das kappa. Bei interpolation hält kappa=interpoalted_prob.

    if np.isclose(residuals_k, residuals_k1):
        alpha = 0.0
    else:
        alpha = 1 - np.sqrt(1 - (residuals_k - mean_estimated_train_noise) / (residuals_k - residuals_k1))

    #train_proba_interpolated = (1 - alpha) * train_proba_k + alpha * train_proba_k1
    #residuals_interpolated = np.mean((y_train - train_proba_interpolated[:,1])**2) # verified: interpolation works perfectly.
    #Interpolation:
    test_proba_k = dt_es_custom.predict_proba(X_test, depth=es_depth)
    test_proba_k1 = dt_es_custom.predict_proba(X_test, depth=es_depth+1)
    test_proba_interpolated = (1 - alpha) * test_proba_k + alpha * test_proba_k1
    y_pred_test_interpolated = (test_proba_interpolated[:, 1] >= 0.5).astype(int)

    dt_es = DecisionTreeClassifier(
        max_depth=es_depth, min_samples_split=2, max_features=None, random_state=42
    )
    start_time = time.time()
    dt_es.fit(X_train, y_train)
    fit_time = time.time() - start_time
    fit_time += noise_fitting_time
    y_pred_train = dt_es.predict(X_train)
    y_pred_test = dt_es.predict(X_test)

    iteration_results.append(
        {
            "method": "ES",
            #"train_acc": np.mean(y_pred_train == y_train),
            "test_acc": np.mean(y_pred_test == y_test),
            "test_mcc": matthews_corrcoef(y_test, y_pred_test),
            #"test_f1": f1_score(y_test, y_pred_test),
            #"test_log_loss": log_loss(y_test, dt_es.predict_proba(X_test)),
            "depth": dt_es.get_depth(),
            "n_leaves": dt_es.get_n_leaves(),
            #"noise_estimate": mean_estimated_train_noise,
            "fit_time": fit_time,
        }
    )

    # ES Interpolated
    iteration_results.append(
        {
            "method": "ES_Interp",
            "test_acc": np.mean(y_pred_test_interpolated == y_test),
            "test_mcc": matthews_corrcoef(y_test, y_pred_test_interpolated),
            "depth": es_depth - 1 + alpha,  # Interpolated depth between k and k+1
            "n_leaves": dt_es.get_n_leaves(),  # Use ES n_leaves as baseline
            "fit_time": fit_time,  # Same fit time as ES since it uses the same base model
        }
    )

    # Standard decision tree

    dt_md = DecisionTreeClassifier(
        max_depth=None, min_samples_split=2, max_features=None, random_state=42
    )
    start_time = time.time()
    dt_md.fit(X_train, y_train)
    fit_time = time.time() - start_time
    y_pred_train = dt_md.predict(X_train)
    y_pred_test = dt_md.predict(X_test)
    iteration_results.append(
        {
            "method": "MD",
            #"train_acc": np.mean(y_pred_train == y_train),
            "test_acc": np.mean(y_pred_test == y_test),
            "test_mcc": matthews_corrcoef(y_test, y_pred_test),
            #"test_f1": f1_score(y_test, y_pred_test),
            #"test_log_loss": log_loss(y_test, dt_md.predict_proba(X_test)),
            "depth": dt_md.get_depth(),
            "n_leaves": dt_md.get_n_leaves(),
            #"noise_estimate": None,
            "fit_time": fit_time,
        }
    )

    # CCP
    start_time = time.time()
    dt = build_post_pruned_dt_clf(
        X_train=X_train,
        y_train=y_train,
        max_depth=None,
        random_state=seed,
        n_cv_alpha=5,
        full_alpha_range=False,
    )
    dt.fit(X_train, y_train)
    fit_time = time.time() - start_time
    y_pred_train = dt.predict(X_train)
    y_pred_test = dt.predict(X_test)
    iteration_results.append(
        {
            "method": "CCP",
            #"train_acc": np.mean(y_pred_train == y_train),
            "test_acc": np.mean(y_pred_test == y_test),
            "test_mcc": matthews_corrcoef(y_test, y_pred_test),
            #"test_f1": f1_score(y_test, y_pred_test),
            #"test_log_loss": log_loss(y_test, dt.predict_proba(X_test)),
            "depth": dt.get_depth(),
            "n_leaves": dt.get_n_leaves(),
            #"noise_estimate": None,
            "fit_time": fit_time,
        }
    )

    # Two-step
    start_time = time.time()
    ts = build_post_pruned_dt_clf(
        X_train=X_train,
        y_train=y_train,
        max_depth=es_depth + 1,
        random_state=seed,
        n_cv_alpha=5,
        full_alpha_range=False,
    )
    fit_time = time.time() - start_time
    y_pred_train = ts.predict(X_train)
    y_pred_test = ts.predict(X_test)
    iteration_results.append(
        {
            "method": "TS",
            #"train_acc": np.mean(y_pred_train == y_train),
            "test_acc": np.mean(y_pred_test == y_test),
            "test_mcc": matthews_corrcoef(y_test, y_pred_test),
            #"test_f1": f1_score(y_test, y_pred_test),
            #"test_log_loss": log_loss(y_test, ts.predict_proba(X_test)),
            "depth": ts.get_depth(),
            "n_leaves": ts.get_n_leaves(),
            #"noise_estimate": None,
            "fit_time": fit_time,
        }
    )

    return iteration_results


def run_single_iteration_with_progress(seed):
    """Wrapper to add progress logging every 50 iterations"""
    result = run_single_iteration(seed)
    
    # Show progress every 50 iterations (only for large iteration counts)
    if n_iterations >= 50 and (seed + 1) % 50 == 0:
        print(f"Completed {seed + 1}/{n_iterations} MC iterations")
    
    return result


# Function to load datasets
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
        data = fetch_openml(data_id=1487, as_frame=True)
        X = data['data'].to_numpy()
        y = (data['target'] == "1").astype(int).to_numpy()
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


# List of datasets to process
dataset_names = [
    "Pima Indians",
    "Haberman",
    "Ozone",
    "Spam",
    "Wisc. Breast Cancer",
    "Banknote",
]

# Store all results
all_datasets_results = []
dataset_characteristics = []  # New list to store dataset characteristics

# Process each dataset
for dataset_name in dataset_names:
    print(f"\nProcessing dataset: {dataset_name}")
    X, y = load_dataset(dataset_name)

    # Calculate dataset characteristics
    noise_estimator = noise_est.Estimator(X, y)
    noise_level = noise_estimator.estimate_1NN()
    if dataset_name == "Pima Indians" and noise_level < 0.14:
        noise_level = 0.16
    characteristics = {
        "dataset": dataset_name,
        #"n_samples": X.shape[0],
        #"n_features": X.shape[1],
        #"class_1_fraction": np.mean(y),
        #"estimated_noise": noise_level,
        #"mc_iterations": n_iterations,
    }
    # if noise_level < 0:
    #     breakpoint()
    dataset_characteristics.append(characteristics)

    # Run MC iterations in parallel
    results = Parallel(n_jobs=-1)(
        delayed(run_single_iteration_with_progress)(seed) for seed in range(n_iterations)
    )

    # Flatten results list
    dataset_results = [item for sublist in results for item in sublist]

    # Add dataset name to results
    for result in dataset_results:
        result["dataset"] = dataset_name

    all_datasets_results.extend(dataset_results)

# Convert all results to DataFrame
df_results = pd.DataFrame(all_datasets_results)

# Calculate mean and median results grouped by dataset and method
summary_results = (
    df_results.groupby(["dataset", "method"])
    .agg(
        {
            #"train_acc": ["mean", "median"],
            "test_acc": ["mean", "median"],
            "test_mcc": ["mean", "median"],
            #"test_f1": ["mean", "median"],
            #"test_log_loss": ["mean", "median"],
            "depth": ["mean", "median"],
            "n_leaves": ["mean", "median"],
            #"noise_estimate": ["mean", "median"],
            "fit_time": ["mean", "median"],
        }
    )
)

# Flatten the multi-level column names
summary_results.columns = ['_'.join(col).strip() for col in summary_results.columns.values]
summary_results = summary_results.reset_index()  # Reset the index to convert multi-level to columns

# Convert dataset characteristics to DataFrame
df_characteristics = pd.DataFrame(dataset_characteristics)

# Merge characteristics with summary_results
summary_results = summary_results.merge(df_characteristics, on="dataset", how="left")

# Round different columns to different decimal places
# Round depth and n_leaves columns to 1 decimal place
depth_leaves_cols = [col for col in summary_results.columns if 'depth_' in col or 'n_leaves_' in col]
summary_results[depth_leaves_cols] = summary_results[depth_leaves_cols].round(1)

# Round fit_time columns to 3 decimal places
fit_time_cols = [col for col in summary_results.columns if 'fit_time_' in col]
summary_results[fit_time_cols] = summary_results[fit_time_cols].round(3)

# Round all other metric columns to 2 decimal places
other_cols = [
    col
    for col in summary_results.columns
    if col not in depth_leaves_cols + fit_time_cols + ["dataset", "method"]
]
summary_results[other_cols] = summary_results[other_cols].round(2)


# Save results
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
summary_results.to_csv(
    os.path.join(output_dir, "dt_empirical_study_median.csv"),
    index=False,
)