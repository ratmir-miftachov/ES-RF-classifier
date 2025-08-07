import pandas as pd
import numpy as np
import time
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed, cpu_count
from sklearn.metrics import matthews_corrcoef

# Local imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils import data_generation
from src.algorithms.mseRF import mseRF

n_iterations = 10
bernoulli_p_global = 0.8  # Global probability parameter used for all DGPs
n_estimators = 50  # Number of trees in each RF

print(f"Using {cpu_count()} CPU cores for parallel processing")

def generate_simulated_data(dgp_name, n_samples, feature_dim, bernoulli_p, random_state=42):
    """
    Generate simulated classification data using the same method as rf_simulation.py
    """
    return data_generation.generate_X_y_f_classification(
        dgp_name=dgp_name,
        bernoulli_p=bernoulli_p,
        n_samples=n_samples,
        feature_dim=feature_dim,
        random_state=random_state,
        n_ticks_per_ax_meshgrid=None,
    )


def run_single_iteration(seed, dgp_config, mtry_spec):
    """Run a single Monte Carlo iteration for sklearn vs UES comparison"""
    # Generate data (already split by data_generation)
    X_train, X_test, y_train, y_test, f_train, f_test = generate_simulated_data(
        dgp_name=dgp_config['dgp_name'],
        n_samples=dgp_config['n_samples'],
        feature_dim=dgp_config['feature_dim'],
        bernoulli_p=dgp_config.get('bernoulli_p', 0.8),
        random_state=seed
    )
    
    # Calculate mtry value based on specification
    feature_dim = dgp_config['feature_dim']
    if mtry_spec == "1":
        max_features = 1
    elif mtry_spec == "sqrt":
        max_features = "sqrt"
    elif mtry_spec == "d":
        max_features = feature_dim
    else:
        raise ValueError(f"Unknown mtry_spec: {mtry_spec}")
    
    iteration_results = []
    
    # 1. sklearn with unlimited depth (baseline)
    start_time = time.time()
    sklearn_unlimited = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,  # Unlimited depth
        max_features=max_features,
        random_state=seed,
    )
    sklearn_unlimited.fit(X_train, y_train)
    sklearn_unlimited_time = time.time() - start_time
    
    y_pred_test = sklearn_unlimited.predict(X_test)
    unlimited_depths = [tree.get_depth() for tree in sklearn_unlimited.estimators_]
    unlimited_mean_depth = np.mean(unlimited_depths)
    unlimited_leaves = [tree.get_n_leaves() for tree in sklearn_unlimited.estimators_]
    unlimited_mean_leaves = np.mean(unlimited_leaves)
    
    iteration_results.append({
        "method": "sklearn_unlimited",
        "test_acc": np.mean(y_pred_test == y_test),
        "test_mcc": matthews_corrcoef(y_test, y_pred_test),
        "depth": unlimited_mean_depth,
        "n_leaves": unlimited_mean_leaves,
        "fit_time": sklearn_unlimited_time,
    })
    
    # 2. UES to find optimal depth
    start_time = time.time()
    ues = mseRF(
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=seed
    )
    ues_rf = ues.fit(X_train, y_train, f_train=f_train)
    ues_discovery_time = time.time() - start_time
    
    # Extract optimal depth found by UES
    optimal_depth = ues_rf.max_depth
    y_pred_test = ues_rf.predict(X_test)
    
    iteration_results.append({
        "method": "UES_discovery",
        "test_acc": np.mean(y_pred_test == y_test),
        "test_mcc": matthews_corrcoef(y_test, y_pred_test),
        "depth": optimal_depth,
        "n_leaves": np.nan,  # UES doesn't expose this easily
        "fit_time": ues_discovery_time,
        "optimal_depth": optimal_depth,
    })
    
    # 3. sklearn with UES-determined optimal depth
    start_time = time.time()
    sklearn_optimal = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=optimal_depth,  # Use UES-discovered depth
        max_features=max_features,
        random_state=seed,
    )
    sklearn_optimal.fit(X_train, y_train)
    sklearn_optimal_time = time.time() - start_time
    
    y_pred_test = sklearn_optimal.predict(X_test)
    optimal_leaves = [tree.get_n_leaves() for tree in sklearn_optimal.estimators_]
    optimal_mean_leaves = np.mean(optimal_leaves)
    
    iteration_results.append({
        "method": "sklearn_optimal",
        "test_acc": np.mean(y_pred_test == y_test),
        "test_mcc": matthews_corrcoef(y_test, y_pred_test),
        "depth": optimal_depth,
        "n_leaves": optimal_mean_leaves,
        "fit_time": sklearn_optimal_time,
        "optimal_depth": optimal_depth,
    })
    

    
    return iteration_results


def run_single_iteration_with_progress(seed, dgp_config, mtry_spec):
    """Wrapper to add progress logging every 5 iterations"""
    result = run_single_iteration(seed, dgp_config, mtry_spec)
    
    # Show progress every 5 iterations
    if (seed + 1) % 5 == 0:
        print(f"Completed {seed + 1}/{n_iterations} MC iterations")
    
    return result


# Mapping from DGP names to display names (same as rf_simulation.py)
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

# Different sample sizes to test
sample_sizes = [2000, 5000, 10000, 20000, 50000]

# Base DGP configurations
base_dgps = [
    {
        "dgp_name": "hierarchical-interaction_sparse_jump",
        "feature_dim": 100,
        "bernoulli_p": bernoulli_p_global,
    },
    # {
    #     "dgp_name": "additive_model_I",
    #     "feature_dim": 30,
    #     "bernoulli_p": bernoulli_p_global,
    # },
]

# Generate all combinations of DGPs and sample sizes
dgp_configs = []
for base_dgp in base_dgps:
    for n_samples in sample_sizes:
        config = base_dgp.copy()
        config["n_samples"] = n_samples
        dgp_configs.append(config)

# Additional DGPs can be added to base_dgps list above if needed

# Different mtry specifications to test
mtry_specs = ["1", "sqrt", "d"]

# Store all results
all_results = []

# Process each mtry specification
for mtry_spec in mtry_specs:
    print(f"\n{'='*80}")
    print(f"PROCESSING MTRY SPECIFICATION: {mtry_spec}")
    print("="*80)
    
    # Process each DGP configuration for this mtry spec
    for dgp_config in dgp_configs:
        dgp_name = dgp_config['dgp_name']
        n_samples = dgp_config['n_samples']
        print(f"\nProcessing DGP: {dgp_name}, n_samples: {n_samples}, mtry: {mtry_spec}")
        
        # Run MC iterations in parallel
        results = Parallel(n_jobs=-1)(
            delayed(run_single_iteration_with_progress)(seed, dgp_config, mtry_spec) 
            for seed in range(n_iterations)
        )
        
        # Flatten results list
        dgp_results = [item for sublist in results for item in sublist]
        
        # Add dataset name, sample size, and mtry spec to results
        display_name = dgp_display_names.get(dgp_name, dgp_name)
        for result in dgp_results:
            result["dataset"] = display_name
            result["n_samples"] = n_samples
            result["mtry_spec"] = mtry_spec
        
        all_results.extend(dgp_results)

# Convert all results to DataFrame
df_results = pd.DataFrame(all_results)

# Calculate median results grouped by dataset, n_samples, mtry_spec, and method
median_results = (
    df_results.groupby(["dataset", "n_samples", "mtry_spec", "method"])
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

    # Filter out UES_discovery from CSV output and add ratios
    csv_results = median_results[median_results["method"] != "UES_discovery"].copy()

# Calculate ratios between sklearn_optimal and sklearn_unlimited for each sample size and mtry spec
optimal_results = csv_results[csv_results["method"] == "sklearn_optimal"].set_index(["dataset", "n_samples", "mtry_spec"])
unlimited_results = csv_results[csv_results["method"] == "sklearn_unlimited"].set_index(["dataset", "n_samples", "mtry_spec"])

# Create ratio rows
ratio_rows = []
for (dataset, n_samples, mtry_spec) in optimal_results.index:
    if (dataset, n_samples, mtry_spec) in unlimited_results.index:
        opt = optimal_results.loc[(dataset, n_samples, mtry_spec)]
        unl = unlimited_results.loc[(dataset, n_samples, mtry_spec)]
        
        ratio_row = {
            "dataset": dataset,
            "n_samples": n_samples,
            "mtry_spec": mtry_spec,
            "method": "sklearn_unlimited_vs_optimal_ratio",
            "test_acc_median": unl["test_acc_median"] / opt["test_acc_median"],
            "test_mcc_median": unl["test_mcc_median"] / opt["test_mcc_median"],
            "depth_median": unl["depth_median"] / opt["depth_median"],
            "n_leaves_median": unl["n_leaves_median"] / opt["n_leaves_median"],
            "fit_time_median": unl["fit_time_median"] / opt["fit_time_median"],
        }
        ratio_rows.append(ratio_row)

# Add ratio rows to csv_results
ratio_df = pd.DataFrame(ratio_rows)
csv_results = pd.concat([csv_results, ratio_df], ignore_index=True)

# Round different columns to different decimal places 
csv_results[["depth_median", "n_leaves_median"]] = csv_results[
    ["depth_median", "n_leaves_median"]
].round(1)

csv_results["fit_time_median"] = csv_results["fit_time_median"].round(3)

# Round other columns to 2 decimal places 
numeric_cols = ["test_acc_median", "test_mcc_median"]
csv_results[numeric_cols] = csv_results[numeric_cols].round(2)

# Display summary
print(f"\n" + "="*80)
    print("SKLEARN vs UES RUNTIME COMPARISON - SUMMARY")
print("="*80)

# Summary by method across all datasets
method_summary = (
    df_results.groupby("method")
    .agg({
        "test_acc": "mean",
        "depth": "mean", 
        "fit_time": "mean",
    })
    .round(3)
)
print("\nMean performance across all datasets:")
print(method_summary.to_string())

# Save results (same structure as rf_simulation.py)
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
csv_results.to_csv(
    os.path.join(output_dir, f"sklearn_depth_sample_size_mtry_comparison_p_{bernoulli_p_global}.csv"),
    index=False,
)

print(f"\nResults saved to:")
print(f"- {output_dir}/sklearn_depth_sample_size_mtry_comparison_p_{bernoulli_p_global}.csv")