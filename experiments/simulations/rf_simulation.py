import pandas as pd
import numpy as np
import os
import sys
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed, cpu_count
from sklearn.metrics import matthews_corrcoef
import time

# Local imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils import noise_level_estimator as noise_est
from src.algorithms.EsGlobalRF import RandomForestClassifier as EsGlobalRF
from src.utils.model_builder import build_rf_clf
from src.utils import data_generation

n_iterations = 10
bernoulli_p_global = 0.8  # Global probability parameter used for all DGPs

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
    """Run a single Monte Carlo iteration for all RF methods"""
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
    
    # Algorithm configurations matching rf_empirical_study.py
    rf_configs = [
        {
            "method": "UES",
            "algorithm": "UES",
            "max_features": "sqrt",
            "vote_probability": False,
            "es_offset": 0,
        },
        {
            "method": "UES_d",
            "algorithm": "UES",
            "max_features": None,  # mtry=d (all features)
            "vote_probability": False,
            "es_offset": 0,
        },
        {
            "method": "UES_1",
            "algorithm": "UES",
            "max_features": 1,  # mtry=1 (single feature)
            "vote_probability": False,
            "es_offset": 0,
        },
        {
            "method": "IES",
            "algorithm": "IES",
            "kappa": "mean_var",  # Use true noise level instead of 1nn
            "max_features": "sqrt",
            "estimate_noise_before_sampling": True,
            "es_offset": 0,
            "rf_train_mse": True,
        },
        {
            "method": "IES_d",
            "algorithm": "IES",
            "kappa": "mean_var",  # Use true noise level instead of 1nn
            "max_features": None,  # mtry=d (all features)
            "estimate_noise_before_sampling": True,
            "es_offset": 0,
            "rf_train_mse": True,
        },
        {
            "method": "IES_1",
            "algorithm": "IES",
            "kappa": "mean_var",  # Use true noise level instead of 1nn
            "max_features": 1,  # mtry=1 (single feature)
            "estimate_noise_before_sampling": True,
            "es_offset": 0,
            "rf_train_mse": True,
        },
        # ILES variants (commented out)
        # {
        #     "method": "ILES",
        #     "algorithm": "IES",
        #     "kappa": "mean_var",  # Use true noise level instead of 1nn
        #     "max_features": "sqrt",
        #     "estimate_noise_before_sampling": True,
        #     "es_offset": 0,
        # },
        # {
        #     "method": "ILES_d",
        #     "algorithm": "IES",
        #     "kappa": "mean_var",  # Use true noise level instead of 1nn
        #     "max_features": None,  # mtry=d (all features)
        #     "estimate_noise_before_sampling": True,
        #     "es_offset": 0,
        # },
        # {
        #     "method": "ILES_1",
        #     "algorithm": "IES",
        #     "kappa": "mean_var",  # Use true noise level instead of 1nn
        #     "max_features": 1,  # mtry=1 (single feature)
        #     "estimate_noise_before_sampling": True,
        #     "es_offset": 0,
        # },
        {
            "method": "MD_scikit",
            "algorithm": "MD_scikit",
            "max_features": "sqrt",
        },
        {
            "method": "MD_scikit_d",
            "algorithm": "MD_scikit",
            "max_features": None,  # mtry=d (all features)
        },
        {
            "method": "MD_scikit_1",
            "algorithm": "MD_scikit",
            "max_features": 1,  # mtry=1 (single feature)
        },
    ]
    
    # Run each RF configuration
    for config in rf_configs:
        start_time = time.time()
        
        # Build RF using model_builder (same as empirical study)
        rf, ensemble_fit_duration = build_rf_clf(
            X_train=X_train,
            y_train=y_train,
            f_train=f_train,  # Provide true probabilities for true noise level 
            algorithm=config.get("algorithm"),
            max_features=config.get("max_features"),
            es_offset=config.get("es_offset"),
            rf_train_mse=config.get("rf_train_mse"),
            kappa=config.get("kappa"),
            n_estimators=2,  # number of trees in the forest
            vote_probability=config.get("vote_probability"),
            estimate_noise_before_sampling=config.get("estimate_noise_before_sampling"),
            random_state=seed,
        )
        
        fit_time = time.time() - start_time
        y_pred_test = rf.predict(X_test)
        
        # Calculate ensemble statistics
        try:
            tree_depths = [tree.get_depth() for tree in rf.estimators_]
            tree_leaves = [tree.get_n_leaves() for tree in rf.estimators_]
            mean_depth = np.mean(tree_depths)
            mean_leaves = np.mean(tree_leaves)
        except:
            # Fallback if tree structure metrics are not available
            mean_depth = np.nan
            mean_leaves = np.nan
        
        iteration_results.append({
            "method": config["method"],
            "test_acc": np.mean(y_pred_test == y_test),
            "test_mcc": matthews_corrcoef(y_test, y_pred_test),
            "depth": mean_depth,
            "n_leaves": mean_leaves,
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
        "bernoulli_p": bernoulli_p_global,  # Not used for additive models
    },
    {
        "dgp_name": "additive_model_I",  # Add. Het.
        "n_samples": 2000,
        "feature_dim": 30,
        "bernoulli_p": bernoulli_p_global,  # Not used for additive models
    },
    {
        "dgp_name": "additive_sparse_jump",  # Add. Jump
        "n_samples": 2000,
        "feature_dim": 30,
        "bernoulli_p": bernoulli_p_global,  # Not used for additive models
    },
    {
        "dgp_name": "additive_sparse_smooth",  # Add. Smooth
        "n_samples": 2000,
        "feature_dim": 30,
        "bernoulli_p": bernoulli_p_global,  # Not used for additive models
    },
    # 2D Cases (in specified order)
    {
        "dgp_name": "circular",
        "n_samples": 2000,
        "feature_dim": 2,
        "bernoulli_p": bernoulli_p_global,
    },
    {
        "dgp_name": "smooth_signal",  # Circular Smooth
        "n_samples": 2000,
        "feature_dim": 2,
        "bernoulli_p": bernoulli_p_global,  # Not used for smooth_signal
    },
    {
        "dgp_name": "rectangular",
        "n_samples": 2000,
        "feature_dim": 2,
        "bernoulli_p": bernoulli_p_global,
    },
    {
        "dgp_name": "sine_cosine",
        "n_samples": 2000,
        "feature_dim": 2,
        "bernoulli_p": bernoulli_p_global,  # Not used for sine_cosine
    },
]

# Store all results
all_results = []

# Process each DGP configuration
for dgp_config in dgp_configs:
    dgp_name = dgp_config['dgp_name']
    print(f"\nProcessing DGP: {dgp_name}")
    
    # Run MC iterations in parallel
    results = Parallel(n_jobs=-1)(
        delayed(run_single_iteration_with_progress)(seed, dgp_config) 
        for seed in range(n_iterations)
    )
    
    # Flatten results list
    dgp_results = [item for sublist in results for item in sublist]
    
    # Add dataset name to results
    display_name = dgp_display_names.get(dgp_name, dgp_name)
    for result in dgp_results:
        result["dataset"] = display_name
    
    all_results.extend(dgp_results)

# Convert all results to DataFrame
df_results = pd.DataFrame(all_results)

# Calculate median results grouped by dataset and method
median_results = (
    df_results.groupby(["dataset", "method"])
    .agg({
        "test_acc": "median",
        "test_mcc": "median", 
        "depth": "median",
        "n_leaves": "median",
        "fit_time": "median",
    })
    .rename(columns={
        "test_acc": "test_acc_median",
        "test_mcc": "test_mcc_median",
        "depth": "depth_median",
        "n_leaves": "n_leaves_median",
        "fit_time": "fit_time_median",
    })
    .reset_index()
)

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
    os.path.join(output_dir, f"rf_simulation_d2_p_{bernoulli_p_global}.csv"),
    index=False,
)

