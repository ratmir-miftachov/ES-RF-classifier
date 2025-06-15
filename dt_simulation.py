import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed, cpu_count
from sklearn.metrics import matthews_corrcoef #, f1_score, log_loss
import time

# Local imports
import noise_level_estimator as noise_est
from clean_dt import DecisionTreeLevelWise
from model_builder import build_post_pruned_dt_clf
import data_generation

n_iterations = 33

print(f"Using {cpu_count()} CPU cores for parallel processing")


def generate_simulated_data(dgp_name, n_samples, feature_dim, bernoulli_p, random_state=42):
    """
    Generate simulated classification data using the data_generation module
    
    Returns X_train, X_test, y_train, y_test, f_train, f_test
    """
    return data_generation.generate_X_y_f_classification(
        dgp_name=dgp_name,
        bernoulli_p=bernoulli_p,
        n_samples=n_samples,
        feature_dim=feature_dim,
        random_state=random_state,
        n_ticks_per_ax_meshgrid=None,  # Use random sampling instead if None
    )


def run_single_iteration(seed, dgp_config):
    """Run a single Monte Carlo iteration for all methods"""
    # Generate data (already split by data_generation)
    X_train, X_test, y_train, y_test, f_train, f_test = generate_simulated_data(
        dgp_name=dgp_config['dgp_name'],
        n_samples=dgp_config['n_samples'],
        feature_dim=dgp_config['feature_dim'],
        bernoulli_p=dgp_config.get('bernoulli_p', 0.8),
        random_state=seed
    )
    
    # Dictionary to store results
    iteration_results = []
    
    # ES with TRUE noise level 
    # For binary classification: noise = E[f(X) * (1 - f(X))]
    true_noise_level = np.mean(f_train * (1 - f_train))
    mean_estimated_train_noise = true_noise_level
    noise_fitting_time = 0.0  # No time needed for oracle noise level
    
    # Uncomment below to use 1NN estimation instead:
    # noise_estimator = noise_est.Estimator(X_train, y_train)
    # start_time = time.time()
    # mean_estimated_train_noise = noise_estimator.estimate_1NN()
    # noise_fitting_time = time.time() - start_time
    
    dt_es_custom = DecisionTreeLevelWise(
        max_depth=None,
        min_samples_split=2,
        max_features="all",
        kappa=mean_estimated_train_noise,
        random_state=42,
    )
    dt_es_custom.fit(X_train, y_train)
    es_depth = max(dt_es_custom.get_depth(), 1)
    
    # Interpolated DT on train set:
    train_proba_k = dt_es_custom.predict_proba(X_train, depth=es_depth)
    train_proba_k1 = dt_es_custom.predict_proba(X_train, depth=es_depth+1)
    
    residuals_k = np.mean((y_train - train_proba_k[:,1])**2)
    residuals_k1 = np.mean((y_train - train_proba_k1[:,1])**2)
    
    if np.isclose(residuals_k, residuals_k1):
        alpha = 0.0
    else:
        alpha = 1 - np.sqrt(1 - (residuals_k - mean_estimated_train_noise) / (residuals_k - residuals_k1))
    
    # Interpolation:
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
    y_pred_test = dt_es.predict(X_test)
    
    iteration_results.append({
        "method": "ES",
        "test_acc": np.mean(y_pred_test == y_test),
        "test_mcc": matthews_corrcoef(y_test, y_pred_test),
        #"test_f1": f1_score(y_test, y_pred_test),
        #"test_log_loss": log_loss(y_test, dt_es.predict_proba(X_test)),
        "depth": dt_es.get_depth(),
        "n_leaves": dt_es.get_n_leaves(),
        #"noise_estimate": mean_estimated_train_noise,
        "fit_time": fit_time,
    })
    
    # ES Interpolated
    iteration_results.append({
        "method": "ES_Interp",
        "test_acc": np.mean(y_pred_test_interpolated == y_test),
        "test_mcc": matthews_corrcoef(y_test, y_pred_test_interpolated),
        #"test_f1": f1_score(y_test, y_pred_test_interpolated),
        #"test_log_loss": log_loss(y_test, test_proba_interpolated),
        "depth": es_depth - 1 + alpha,  # Interpolated depth
        "n_leaves": dt_es.get_n_leaves(),
        #"noise_estimate": mean_estimated_train_noise,
        "fit_time": fit_time,
    })
    
    # Standard decision tree (Maximum Depth)
    dt_md = DecisionTreeClassifier(
        max_depth=None, min_samples_split=2, max_features=None, random_state=42
    )
    start_time = time.time()
    dt_md.fit(X_train, y_train)
    fit_time = time.time() - start_time
    y_pred_test = dt_md.predict(X_test)
    
    iteration_results.append({
        "method": "MD",
        "test_acc": np.mean(y_pred_test == y_test),
        "test_mcc": matthews_corrcoef(y_test, y_pred_test),
        #"test_f1": f1_score(y_test, y_pred_test),
        #"test_log_loss": log_loss(y_test, dt_md.predict_proba(X_test)),
        "depth": dt_md.get_depth(),
        "n_leaves": dt_md.get_n_leaves(),
        #"noise_estimate": None,
        "fit_time": fit_time,
    })
    
    # CCP (Cost Complexity Pruning)
    start_time = time.time()
    dt_ccp = build_post_pruned_dt_clf(
        X_train=X_train,
        y_train=y_train,
        max_depth=None,
        random_state=seed,
        n_cv_alpha=5,
        full_alpha_range=False,
    )
    fit_time = time.time() - start_time
    y_pred_test = dt_ccp.predict(X_test)
    
    iteration_results.append({
        "method": "CCP",
        "test_acc": np.mean(y_pred_test == y_test),
        "test_mcc": matthews_corrcoef(y_test, y_pred_test),
        #"test_f1": f1_score(y_test, y_pred_test),
        #"test_log_loss": log_loss(y_test, dt_ccp.predict_proba(X_test)),
        "depth": dt_ccp.get_depth(),
        "n_leaves": dt_ccp.get_n_leaves(),
        #"noise_estimate": None,
        "fit_time": fit_time,
    })
    
    # Two-step (TS)
    start_time = time.time()
    dt_ts = build_post_pruned_dt_clf(
        X_train=X_train,
        y_train=y_train,
        max_depth=es_depth + 1,
        random_state=seed,
        n_cv_alpha=5,
        full_alpha_range=False,
    )
    fit_time = time.time() - start_time
    fit_time += noise_fitting_time  # Add noise estimation time
    y_pred_test = dt_ts.predict(X_test)
    
    iteration_results.append({
        "method": "TS",
        "test_acc": np.mean(y_pred_test == y_test),
        "test_mcc": matthews_corrcoef(y_test, y_pred_test),
        #"test_f1": f1_score(y_test, y_pred_test),
        #"test_log_loss": log_loss(y_test, dt_ts.predict_proba(X_test)),
        "depth": dt_ts.get_depth(),
        "n_leaves": dt_ts.get_n_leaves(),
        #"noise_estimate": mean_estimated_train_noise,
        "fit_time": fit_time,
    })
    
    return iteration_results


def run_single_iteration_with_progress(seed, dgp_config):
    """Wrapper to add progress logging every 50 iterations"""
    result = run_single_iteration(seed, dgp_config)
    
    # Show progress every 50 iterations (only for large iteration counts)
    if n_iterations >= 50 and (seed + 1) % 50 == 0:
        print(f"Completed {seed + 1}/{n_iterations} MC iterations")
    
    return result


# Mapping from DGP names to display names (in desired order)
dgp_display_names = {
    "hierarchical-interaction_sparse_jump": "Add. H.I. Jump",
    "additive_model_I": "Add. Het.",
    "additive_sparse_jump": "Add. Jump", 
    "additive_sparse_smooth": "Add. Smooth",
    "circular": "Circular",
    "smooth_signal": "Circular Smooth",
    "rectangular": "Rectangular",
    "sine_cosine": "Sine Cosine",
}

# List of DGP configurations to process (in desired order)
dgp_configs = [
    # Additive Models (in specified order)
    {
        "dgp_name": "hierarchical-interaction_sparse_jump",  # Add. H.I. Jump
        "n_samples": 2000,
        "feature_dim": 30,
        "bernoulli_p": 0.8,  # Not used for additive models
    },
    {
        "dgp_name": "additive_model_I",  # Add. Het.
        "n_samples": 2000,
        "feature_dim": 30,
        "bernoulli_p": 0.8,  # Not used for additive models
    },
    {
        "dgp_name": "additive_sparse_jump",  # Add. Jump
        "n_samples": 2000,
        "feature_dim": 30,
        "bernoulli_p": 0.8,  # Not used for additive models
    },
    {
        "dgp_name": "additive_sparse_smooth",  # Add. Smooth
        "n_samples": 2000,
        "feature_dim": 30,
        "bernoulli_p": 0.8,  # Not used for additive models
    },
    # 2D Cases (in specified order)
    {
        "dgp_name": "circular",
        "n_samples": 2000,
        "feature_dim": 2,
        "bernoulli_p": 0.8,
    },
    {
        "dgp_name": "smooth_signal",  # Circular Smooth
        "n_samples": 2000,
        "feature_dim": 2,
        "bernoulli_p": 0.8,  # Not used for smooth_signal
    },
    {
        "dgp_name": "rectangular",
        "n_samples": 2000,
        "feature_dim": 2,
        "bernoulli_p": 0.8,
    },
    {
        "dgp_name": "sine_cosine",
        "n_samples": 2000,
        "feature_dim": 2,
        "bernoulli_p": 0.8,  # Not used for sine_cosine
    },
]

# Store all results
all_results = []
dgp_characteristics = []

# Process each DGP configuration
for dgp_config in dgp_configs:
    dgp_name = dgp_config['dgp_name']  # Use clean DGP name without dimension/parameter suffixes
    print(f"\nProcessing DGP: {dgp_name}")
    
    # Calculate DGP characteristics (commented out to match empirical study)
    #characteristics = {
    #    "dataset": dgp_name,
    #    "dgp_name": dgp_config['dgp_name'],
    #    "feature_dim": dgp_config['feature_dim'],
    #    "n_samples": dgp_config['n_samples'],
    #    "bernoulli_p": dgp_config['bernoulli_p'],
    #}
    #dgp_characteristics.append(characteristics)
    
    # Run MC iterations in parallel
    results = Parallel(n_jobs=-1)(
        delayed(run_single_iteration_with_progress)(seed, dgp_config) 
        for seed in range(n_iterations)
    )
    
    # Flatten results list
    dgp_results = [item for sublist in results for item in sublist]
    
    # Add dataset name to results (renamed from dgp to match empirical study)
    display_name = dgp_display_names.get(dgp_name, dgp_name)  # Use display name or fallback to dgp_name
    for result in dgp_results:
        result["dataset"] = display_name
        #result.update(dgp_config)  # Commented out extra config info
    
    all_results.extend(dgp_results)

# Convert all results to DataFrame
df_results = pd.DataFrame(all_results)

# Calculate median results grouped by dataset and method (using median for robustness)
median_results = (
    df_results.groupby(["dataset", "method"])
    .agg({
        "test_acc": "median",
        "test_mcc": "median", 
        #"test_f1": "median",
        #"test_log_loss": "median",
        "depth": "median",
        "n_leaves": "median",
        #"noise_estimate": "median",
        "fit_time": "median",
    })
    .rename(columns={
        "test_acc": "test_acc_median",
        "test_mcc": "test_mcc_median",
        #"test_f1": "test_f1_median", 
        #"test_log_loss": "test_log_loss_median",
        "depth": "depth_median",
        "n_leaves": "n_leaves_median",
        #"noise_estimate": "noise_estimate_median",
        "fit_time": "fit_time_median",
    })
    .reset_index()
)

# Commented out characteristics merging to match empirical study structure
#df_characteristics = pd.DataFrame(dgp_characteristics)
#median_results = median_results.merge(df_characteristics, on="dataset", how="left")

# Round different columns to different decimal places 
median_results[["depth_median", "n_leaves_median"]] = median_results[
    ["depth_median", "n_leaves_median"]
].round(1)

median_results["fit_time_median"] = median_results["fit_time_median"].round(3)

# Round other columns to 2 decimal places 
numeric_cols = ["test_acc_median", "test_mcc_median"]
median_results[numeric_cols] = median_results[numeric_cols].round(2)

# Save results
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
median_results.to_csv(
    os.path.join(output_dir, "dt_simulation.csv"),
    index=False,
)

