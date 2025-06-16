# Clear all variables from the global namespace
for name in list(globals()):
    if not name.startswith('_'):
        del globals()[name]

# Import required libraries and modules
import numpy as np
import os

import regression_tree as rt
import importlib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, KFold
import CARTRM as RM # for local ES. (not used in the paper)
import semiglobal_runner as semi_runner
import data_generation
from joblib import Parallel, delayed
import noise_level_estimator as noise_est
import es_mp
import pandas as pd

# Reload modules to ensure changes are applied
importlib.reload(data_generation)
importlib.reload(semi_runner)
importlib.reload(es_mp)
importlib.reload(rt)

#np.random.seed(42)

def methods_stopping(X_train, y_train, X_test, y_test, stopping, noise_level, noise, k_cv, true_signal, noise_test, true_signal_test, crit):
    """
       Execute the specified stopping method on decision trees using various criteria.

       Parameters:
       X_train, y_train (np.ndarray): Training data features and targets.
       X_test, y_test (np.ndarray): Test data features and targets.
       stopping (str): The type of stopping criterion ('local', 'global', 'pruning', 'deep', 'semi').
       noise_level (float): Noise level used in the simulation.
       noise (np.ndarray): Array of noise values.
       k_cv (KFold): Cross-validation generator.
       true_signal (np.ndarray): Array of the true signals for training set.
       noise_test (np.ndarray): Array of noise values for the test set.
       true_signal_test (np.ndarray): Array of the true signals for test set.
       crit (str): Criterion used for decision tree regression ('sigma', 'NN', 'epsilon').

       Returns:
       float: Mean squared error for the test predictions depending on the stopping method.
       """

    if crit == 'sigma':
        kappa = noise_level
    elif crit == 'NN':
        estimator = noise_est.Estimator(X_train, y_train)
        kappa = estimator.estimate(method='1NN')
        print('kappa is:', kappa)
    elif crit == 'epsilon':
        kappa = np.mean(noise**2)


    if stopping == 'local':
        return local_ES(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                        noise_level=noise_level, noise=noise, true_signal_test=true_signal_test, crit=crit)
    elif stopping == 'global':
        return global_ES(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                         noise_level=noise_level, noise=noise, k_cv=k_cv,
                         true_signal=true_signal, noise_test=noise_test, signal_test=true_signal_test, kappa=kappa)
    elif stopping == 'pruning':
        return pruning_tree(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, k_cv=k_cv, true_signal_test=true_signal_test)
    elif stopping == 'deep':
        return deep_tree(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, true_signal_test=true_signal_test)
    elif stopping == 'semi':
        return semi_ES(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                       noise_level=noise_level, noise=noise, true_signal=true_signal,
                       noise_test=noise_test, true_signal_test=true_signal_test, kappa=kappa)

def local_ES(X_train, y_train, X_test, y_test, noise_level, noise, true_signal_test, crit):

    if crit == 'sigma':
        tree_local = RM.DecisionTreeRegressor(max_depth=40, loss='mse', global_es=False, min_samples_split=2, sigma2=noise_level)
        tree_local.train(X_train, y_train)
    elif crit == 'NN':
        tree_local = RM.DecisionTreeRegressor(max_depth=40, loss='mse', global_es=False, min_samples_split=2, sigma_est_method='1NN')
        tree_local.train(X_train, y_train)
    elif crit == 'epsilon':
        tree_local = RM.DecisionTreeRegressor(max_depth=40, loss='mse', global_es=False, min_samples_split=2, noise_vector = noise)
        tree_local.train(X_train, y_train)

    predictions_test = tree_local.predict(X_test)
    mse_local = np.mean((predictions_test - true_signal_test)**2)
    return mse_local

def semi_ES(X_train, y_train, X_test, y_test, noise_level, noise, true_signal, noise_test, true_signal_test, kappa):
    tree_semi_runner = semi_runner.DecisionTreeRegressor(max_depth=40, max_iter=1000, loss='mse', global_es=True,
                                                         min_samples_split=1, kappa=kappa)
    tree_semi_runner.train(X_train, y_train)
    stop_tau = tree_semi_runner.stopping_iteration
    predictions_test = tree_semi_runner.predict(X_test, iteration=stop_tau)
    mse_semi = np.mean((predictions_test - true_signal_test)**2)

    mse_list = []
    for iter in range(1, 990):
        predictions = tree_semi_runner.predict(X_test, iteration=iter)
        mse = np.mean((predictions - true_signal_test) ** 2)
        mse_list.append(mse)

    stopping_oracle = np.argmin(mse_list) + 1
    semi_oracle = np.min(mse_list)

    return mse_semi, semi_oracle, stopping_oracle

def global_ES(X_train, y_train, X_test, y_test, noise_level, noise, k_cv, true_signal, noise_test, signal_test, kappa):


    regression_tree = rt.RegressionTree(design=X_train, response=y_train, min_samples_split=1, true_signal=true_signal,
                                        true_noise_vector=noise)
    regression_tree.iterate(max_depth=30)
    early_stopping_iteration = regression_tree.get_discrepancy_stop(critical_value=kappa)

    # Balanced oracle:
    balanced_oracle_iteration = regression_tree.get_balanced_oracle() + 1
    prediction_balanced_oracle = regression_tree.predict(X_test, depth=balanced_oracle_iteration)
    mse_global_balanced_oracle = np.mean((prediction_balanced_oracle - signal_test) ** 2)
    # Global ES prediction
    prediction_global_k1 = regression_tree.predict(X_test, depth=early_stopping_iteration)
    mse_global = np.mean((prediction_global_k1 - signal_test) ** 2)
    # Interpolation:
    if early_stopping_iteration == 0:
        mse_global_inter = mse_global
        print('No Interpolation done.')
    else:
        prediction_global_k = regression_tree.predict(X_test, depth=early_stopping_iteration - 1)
        residuals = regression_tree.residuals
        r2_k1 = residuals[early_stopping_iteration]
        r2_k = residuals[early_stopping_iteration - 1]
        alpha = 1 - np.sqrt(1 - (r2_k - kappa) / (r2_k - r2_k1))
        predictions_interpolated = (1 - alpha) * prediction_global_k + alpha * prediction_global_k1
        mse_global_inter = np.mean((predictions_interpolated - signal_test) ** 2)

    # 2-Step:
    m = early_stopping_iteration + 1
    _, tree_two_step, filtered_alpha_sequence = es_mp.esmp(m, X_train, y_train, 0.01, k_cv)
    prediction_two_step = tree_two_step.predict(X_test)
    mse_two_step = np.mean((prediction_two_step - signal_test) ** 2)

    # Oracle on test set for interpolated global and global
    mse_global_list = []
    mse_global_interpolated_list = []
    residuals = regression_tree.residuals
    max_possible_depth = len(residuals)

    for iter in range(1, max_possible_depth):
        predictions_global_k1 = regression_tree.predict(X_test, depth=iter)
        predictions_global_k = regression_tree.predict(X_test, depth=iter - 1)
        r2_k1_test = regression_tree.residuals[iter]
        r2_k_test = regression_tree.residuals[iter - 1]
        alpha_test = 1 - np.sqrt(1 - (r2_k_test - kappa) / (r2_k_test - r2_k1_test))
        predictions_interpolated = (1 - alpha_test) * predictions_global_k + alpha_test * predictions_global_k1
        mse_global_interpolated_list.append(np.mean((predictions_interpolated - signal_test) ** 2))

        # Empirical MSE on test set
        mse_global_temp = np.mean((predictions_global_k1 - signal_test) ** 2)
        mse_global_list.append(mse_global_temp)

    global_stopping_iteration_oracle = np.argmin(mse_global_list) + 1
    oracle_tree = rt.RegressionTree(design=X_train, response=y_train, min_samples_split=1, true_signal=true_signal,
                                        true_noise_vector=noise)
    oracle_tree.iterate(max_depth=global_stopping_iteration_oracle + 1)
    oracle_leaf_count_global = oracle_tree.get_n_leaves()

    # Interpolated oracle:
    mse_oracle_global_interpolated = np.nanmin(mse_global_interpolated_list)
    # Global Oracle:
    mse_global_min = np.min(mse_global_list)
    # Take the min of both:
    mse_oracle_early_stopping = min(mse_oracle_global_interpolated, mse_global_min)

    # 2 Step Oracle:
    trees = []
    mse_scores_oracle = []
    for ccp_alpha in filtered_alpha_sequence:
        dtree_alpha = DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha,
                                            max_depth=m)  # oracle global ES iteration.
        dtree_alpha.fit(X_train, y_train)
        trees.append(dtree_alpha)
        predictions_test = dtree_alpha.predict(X_test)
        mse = np.mean((predictions_test - signal_test) ** 2)
        mse_scores_oracle.append(mse)

    mse_oracle_two_step = np.min(mse_scores_oracle)
    
    oracle_alpha_index = np.argmin(mse_scores_oracle)
    oracle_alpha = filtered_alpha_sequence[oracle_alpha_index]
    tree_oracle_alpha = DecisionTreeRegressor(ccp_alpha=oracle_alpha)
    tree_oracle_alpha.fit(X_train, y_train)
    two_step_oracle_iteration = tree_oracle_alpha.get_n_leaves()


    return mse_global, mse_two_step, mse_global_balanced_oracle, mse_global_inter, mse_oracle_early_stopping, mse_oracle_two_step, oracle_leaf_count_global, two_step_oracle_iteration

def deep_tree(X_train, y_train, X_test, y_test, true_signal_test):

    regression_tree = rt.RegressionTree(design=X_train, response=y_train, min_samples_split=1)
    regression_tree.iterate(max_depth=40)
    max_possible_depth = len(regression_tree.residuals)
    predictions_test = regression_tree.predict(X_test, depth=max_possible_depth)
    mse_deep = np.mean((predictions_test - true_signal_test) ** 2)

    return mse_deep

def pruning_tree(X_train, y_train, X_test, y_test, k_cv, true_signal_test):
    """
    Apply cost complexity pruning to a decision tree and evaluate its performance.

    Parameters:
    X_train, y_train, X_test, y_test (np.ndarray): Training and testing data features and targets.
    k_cv (KFold): Cross-validation generator for determining the best pruning level.
    true_signal_test (np.ndarray): True signals for the test data.

    Returns:
    Tuple[float, float]: MSE after pruning and MSE using the oracle selected pruning level.
    """
    # Pruning on training set:
    tree_pruning = DecisionTreeRegressor(max_depth=40)
    tree_pruning.fit(X_train, y_train)
    path = tree_pruning.cost_complexity_pruning_path(X_train, y_train)
    alpha_sequence, impurities = path.ccp_alphas[1:-1], path.impurities[1:-1]
    threshold = 0.01
    filtered_alpha_sequence = np.array([alpha_sequence[0]])
    filtered_impurities = np.array([impurities[0]])  # Include the first impurity
    for u in range(1, len(alpha_sequence)):
        impurity_change = impurities[u] - impurities[u - 1]
        if impurity_change >= threshold:
            filtered_alpha_sequence = np.append(filtered_alpha_sequence, alpha_sequence[u])
            filtered_impurities = np.append(filtered_impurities, impurities[u])
    trees = []
    mse_scores = []
    for ccp_alpha in filtered_alpha_sequence:
        dtree_alpha = DecisionTreeRegressor(ccp_alpha=ccp_alpha)
        dtree_alpha.fit(X_train, y_train)
        trees.append(dtree_alpha)
        #To get the pruning oracel:
        predictions_test = dtree_alpha.predict(X_test)
        mse = np.mean((predictions_test - true_signal_test) ** 2)
        mse_scores.append(mse)

    # Select the tree with the smallest MSE
    mse_oracle_pruning = np.min(mse_scores)

    parameters = {'ccp_alpha': filtered_alpha_sequence.tolist()}
    gsearch = GridSearchCV(DecisionTreeRegressor(), parameters, cv=k_cv, scoring='neg_mean_squared_error')
    gsearch.fit(X_train, y_train)
    clf = gsearch.best_estimator_

    predictions_test_pruning = clf.predict(X_test)
    mse_pruning = np.mean((predictions_test_pruning - true_signal_test) ** 2)
    pruning_n_leaves_oracle = clf.get_n_leaves()

    return mse_pruning, mse_oracle_pruning, pruning_n_leaves_oracle

def run_simulation(dgp, M=300, n_points=63, noise_level=1, crit='sigma'): # crit = 'sigma', 'NN', 'epsilon'
    """
       Run simulation for a specified data generating process (DGP) with multiple configurations.

       Parameters:
       dgp (str): Name of the data generating process.
       M (int): Number of Monte Carlo simulation runs.
       n_points (int): Number of data points per run.
       noise_level (float): Standard deviation of the noise added to the data.
       crit (str): Criterion used for noise estimation ('sigma', 'NN', 'epsilon').

       Returns:
       np.ndarray: Array of results from different stopping criteria applied in decision trees.
       """
    if crit == None:
        print('Please specify the kappa threshold')

    all_X_train, all_y_train, all_noise_train, all_f, all_f_test = [], [], [], [], []
    all_X_test, all_y_test, all_noise_test = [], [], []

    for _ in range(M):
        # X deterministic:
        if dgp in ['sine_cosine', 'rectangular', 'circular', 'smooth_signal']:
            n_train = 1000
            d = 5
            X_train = np.random.uniform(0, 1, size=(n_train, d))
            X_test = np.random.uniform(0, 1, size=(n_train, d))


        elif dgp == 'additive_smooth' or dgp == 'additive_step' or 'additive_linear' or 'additive_hills':
            n_train = 1000
            n_test = 1000
            d = 30
            X_train = np.random.uniform(-2.5, 2.5, size=(n_train, d))
            X_test = np.random.uniform(-2.5, 2.5, size=(n_test, d))
        # X random:
        elif dgp == 'breiman84':
            n_train = 500
            n_test = 500
            X1_train = np.random.choice([-1, 1], size=n_train, p=[0.5, 0.5])
            X2_10_train = np.random.choice([-1, 0, 1], size=(n_train, 9), p=[1 / 3, 1 / 3, 1 / 3])
            X_train = np.column_stack((X1_train, X2_10_train))

            # Generate on test set:
            X1_test = np.random.choice([-1, 1], size=n_test, p=[0.5, 0.5])
            X2_10_test = np.random.choice([-1, 0, 1], size=(n_test, 9), p=[1 / 3, 1 / 3, 1 / 3])
            X_test = np.column_stack((X1_test, X2_10_test))
        # X random:
        elif dgp == 'friedman1':
            n_train = 500
            n_test = 500
            d = 10
            X_train = np.random.uniform(0, 1, size=(n_train, d))
            X_test = np.random.uniform(0, 1, size=(n_test, d))

        y_train, noise_train = data_generation.generate_data_from_X(X_train, noise_level, dgp_name=dgp, n_points=n_points, add_noise=True)
        y_test, noise_test = data_generation.generate_data_from_X(X_test, noise_level, dgp_name=dgp, n_points=n_points, add_noise=True) # TODO: That is why n_test=n_train required
        f, nuisance = data_generation.generate_data_from_X(X_train, noise_level, dgp_name=dgp, n_points=n_points, add_noise=False)
        f_test, nuisance = data_generation.generate_data_from_X(X_test, noise_level, dgp_name=dgp, n_points=n_points, add_noise=False) # TODO: That is why n_test=n_train required

        all_f.append(f)
        all_f_test.append(f_test)

        all_X_train.append(X_train)
        all_y_train.append(y_train)
        all_noise_train.append(noise_train)

        all_X_test.append(X_test)
        all_y_test.append(y_test)
        all_noise_test.append(noise_test)

    def monte_carlo(stopping_method, k_cv):

        mspe_list = []
        additional_metric_list = []
        additional_metric2_list = []
        additional_metric3_list = []
        additional_metric4_list = []
        additional_metric5_list = []
        additional_metric6_list = []
        additional_metric7_list = []


        for i in range(M):
            print(dgp, stopping_method, i)
            k_cv_iter = k_cv[i]

            results = methods_stopping(all_X_train[i], all_y_train[i], all_X_test[i], all_y_test[i],
                                               stopping=stopping_method, noise_level=noise_level,
                                               noise=all_noise_train[i], k_cv=k_cv_iter,
                                       true_signal=all_f[i], noise_test=all_noise_test[i],
                                       true_signal_test = all_f_test[i], crit=crit)
            if not isinstance(results, tuple):
                results = (results,)
            mspe = results[0]
            mspe_list.append(mspe)

            # Handle the additional metric if it exists
            if len(results) > 1:
                additional_metric_list.append(results[1])
            if len(results) > 2:
                additional_metric2_list.append(results[2])
            if len(results) > 3:
                additional_metric3_list.append(results[3])
            if len(results) > 4:
                additional_metric4_list.append(results[4])
            if len(results) > 5:
                additional_metric5_list.append(results[5])
            if len(results) > 6:
                additional_metric6_list.append(results[6])
            if len(results) > 7:
                additional_metric7_list.append(results[7])

        mean_mspe = np.array(mspe_list)
        mean_additional_metric = np.array(additional_metric_list) if additional_metric_list else None
        mean_additional_metric2 = np.array(additional_metric2_list) if additional_metric2_list else None
        mean_additional_metric3 = np.array(additional_metric3_list) if additional_metric3_list else None
        mean_additional_metric4 = np.array(additional_metric4_list) if additional_metric4_list else None
        mean_additional_metric5 = np.array(additional_metric5_list) if additional_metric5_list else None
        mean_additional_metric6 = np.array(additional_metric6_list) if additional_metric6_list else None
        mean_additional_metric7 = np.array(additional_metric7_list) if additional_metric7_list else None

        return mean_mspe, mean_additional_metric, mean_additional_metric2, mean_additional_metric3, mean_additional_metric4, mean_additional_metric5, mean_additional_metric6, mean_additional_metric7

    cv_splits = [KFold(n_splits=5, shuffle=True, random_state=42 + i) for i in range(M)]

    mspe_local_mean = monte_carlo('local', k_cv=cv_splits)[0]
    mspe_global_mean, mspe_global_pruning_mean, mspe_global_oracle, mspe_global_inter, mspe_oracle_inter, mspe_two_step_oracle, global_iter_oracle, two_step_iter_oracle  = monte_carlo(stopping_method='global', k_cv=cv_splits)
    # global ES, 2Step, ES Oracle, ES (interpolated), ES Oracle (interpolated), 2Step Oracle
    mspe_pruning_mean, mspe_pruning_min, pruning_iter_oracle = monte_carlo('pruning', k_cv=cv_splits)[0:3]
    mspe_deep_mean = monte_carlo('deep', k_cv=cv_splits)[0]
    mspe_semi_mean, mspe_semi_oracle, semi_iter_oracle = monte_carlo('semi', k_cv=cv_splits)[0:3]

    return np.column_stack((mspe_deep_mean, mspe_local_mean, mspe_pruning_mean,
                            mspe_global_pruning_mean, mspe_global_mean, mspe_global_inter,
                            mspe_global_oracle, mspe_semi_mean, mspe_oracle_inter, mspe_pruning_min, mspe_two_step_oracle, mspe_semi_oracle,
                            global_iter_oracle, two_step_iter_oracle,  pruning_iter_oracle, semi_iter_oracle
                            ))

def run_simulation_wrapper(dgp_name):

    if dgp_name == 'smooth_signal':
        return run_simulation(dgp_name, noise_level=1)
    elif dgp_name == 'breiman84':
        return run_simulation(dgp_name, noise_level=1)
    return run_simulation(dgp_name)

def create_and_save_boxplot(data, dgp_name, fig_dir):

    # Create a boxplot for the given data
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(data, patch_artist=True, labels = ['Pruning', 'Global','Global Int', 'Two-Step', 'Semi'])

    # Define custom colors
    colors = ['#c6dbef', '#d3d3d3', '#fee08b', '#90ee90', '#dda0dd']

    # Set colors for each box
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_linewidth(1.5)  # Set the border thickness
    # Making whiskers, caps, and medians thicker and grey
    for whisker in bp['whiskers']:
        whisker.set_linewidth(1.5)
    for cap in bp['caps']:
        cap.set_linewidth(1.5)
    for median in bp['medians']:
        median.set_linewidth(1.5)

    # Set y-axis limits
    plt.axhline(y=1, color='black', linestyle='--', linewidth=1.5)
    plt.ylim(0, 1)
    # Enable gridlines
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=18)  # Adjust the size as needed
    plt.tight_layout()  # Adjust layout
    plt.savefig(os.path.join(fig_dir, f'{dgp_name}_boxplot.png'), bbox_inches='tight', dpi=300)

    plt.close()  # Close the plot to free memory

def create_and_save_boxplot_ratio_easy(data, fig_dir, name):

    # Create a boxplot for the given data
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(data, patch_artist=True, labels = ['Rectangular', 'Circular', 'Sine cosine', 'Elliptical'])

    # Define custom colors
    colors = ['lightblue', 'lightyellow', 'violet', 'lightpink']

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_linewidth(1.5)
    for whisker in bp['whiskers']:
        whisker.set_linewidth(1.5)
    for cap in bp['caps']:
        cap.set_linewidth(1.5)
    for median in bp['medians']:
        median.set_linewidth(1.5)

    plt.axhline(y=1, color='black', linestyle='--', linewidth=1.5)
    plt.ylim(0, 2)
    plt.grid(True)

    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'ratio_boxplot_{name}.png'), bbox_inches='tight', dpi=300)
    plt.close()

def create_and_save_boxplot_ratio_highdim(data, fig_dir, name):

    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(data, patch_artist=True, labels = ['Additive smooth', 'Additive step', 'Additive linear', 'Additive hills'])
    colors = ['lightblue', 'lightyellow', 'violet', 'lightpink']

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_linewidth(1.5)
    for whisker in bp['whiskers']:
        whisker.set_linewidth(1.5)
    for cap in bp['caps']:
        cap.set_linewidth(1.5)
    for median in bp['medians']:
        median.set_linewidth(1.5)

    plt.axhline(y=1, color='black', linestyle='--', linewidth=1.5)
    plt.ylim(0, 2)
    plt.grid(True)

    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'ratio_boxplot_additive_{name}.png'), bbox_inches='tight', dpi=300)
    plt.close()

def simple_latex_table(arr1, arr2, column_names, row_names, name_threshold, table_dir=None):
    # Start the LaTeX table with minimal formatting
    latex_str = "\\begin{tabular}{l" + "c" * len(column_names) + "}\n\\hline\n"

    # Add column headers
    latex_str += " & " + " & ".join(column_names) + " \\\\ \\hline\n"

    # Add each row with its corresponding row name
    for name, row1, row2 in zip(row_names, arr1, arr2):
        # Formatting rows with data from both arrays
        row_str = name + " & "
        row_str += " & ".join(f"{x1[0]:.2f} ({x2[0]:.2f})" for x1, x2 in zip(row1, row2))  # Adjusted to format each element correctly with brackets
        row_str += " \\\\\n"
        latex_str += row_str

    # Close the table
    latex_str += "\\hline\n\\end{tabular}"

    if table_dir is not None:
        with open(os.path.join(table_dir, f"table_{name_threshold}.txt"), "w") as f:
            f.write(latex_str)

    return latex_str

def main():

    dgps = ['rectangular', 'circular', 'sine_cosine', 'smooth_signal','additive_smooth', 'additive_step', 'additive_linear', 'additive_hills']

    results = Parallel(n_jobs=48)(delayed(run_simulation_wrapper)(dgp) for dgp in dgps)
    results = np.round(results, 6)
    ratio_array1 = []
    ratio_array2 = []
    ratio_array3 = []

    errors_oracle_array4 = []
    errors_array5 = []
    oracle_iters_array = []
    #fig_dir = '/Users/ratmir/PycharmProjects/RandomForest'
    fig_dir = '/home/RDC/miftachr/H:/miftachr/ES_CART/boxplots_sigma'

    # Process each set of results and save a boxplot
    for dgp_name, result in zip(dgps, results):
        print(dgp_name)
        # Select the 9th col as the interpolated global min:
        interpolated_oracle = result[:, 8].reshape(-1, 1)
        # Select the 10th col as the pruning oracle:
        pruning_oracle = result[:, 9].reshape(-1, 1)
        # Select the 11th col as the two-step oracle:
        two_step_oracle = result[:, 10].reshape(-1, 1)
        # Select the 12th col as the semi oracle:
        semi_oracle = result[:, 11].reshape(-1, 1)

        results_no_oracles = result[:, :-8] # Remove the four oracle columns
        iter_oracles = result[:, -4:]

        global_iter_oracle = iter_oracles[:, np.newaxis, 0]
        two_step_iter_oracle = iter_oracles[:, np.newaxis, 1]
        pruning_iter_oracle = iter_oracles[:, np.newaxis, 2]
        semi_iter_oracle = iter_oracles[:, np.newaxis, 3]

        # Relative efficiencies for each estimator:
        deep_error = results_no_oracles[:, np.newaxis, 0]
        local_error = results_no_oracles[:, np.newaxis, 1]

        global_error = results_no_oracles[:, np.newaxis, 4]
        inter_error = results_no_oracles[:, np.newaxis, 5]
        semi_error = results_no_oracles[:, np.newaxis, 7]
        pruning_error = results_no_oracles[:, np.newaxis, 2]
        two_step_error = results_no_oracles[:, np.newaxis, 3]

        relative_global = np.sqrt(interpolated_oracle/results_no_oracles[:, np.newaxis, 4])
        relative_inter = np.sqrt(interpolated_oracle/results_no_oracles[:, np.newaxis, 5])
        relative_semi = np.sqrt(semi_oracle/results_no_oracles[:, np.newaxis, 7])
        # Local not ready yet:
        relative_pruning = np.sqrt(pruning_oracle/results_no_oracles[:, np.newaxis, 2])
        relative_two_step = np.sqrt(two_step_oracle/results_no_oracles[:, np.newaxis, 3])

        #Filter due to numerics:
        mask_inter = relative_inter > 1.0001
        filtered_relative_inter = relative_inter[~mask_inter]
        mask_pruning = relative_pruning > 1.0001
        filtered_relative_pruning = relative_pruning[~mask_pruning]

        mask_semi = relative_semi > 1.0001
        filtered_relative_semi = relative_semi[~mask_semi]


        relative_stack = [filtered_relative_pruning[:, np.newaxis],
                          relative_global, filtered_relative_inter[:, np.newaxis], relative_two_step,
                        filtered_relative_semi[:, np.newaxis]]

        flattened_stack = [arr.flatten() for arr in relative_stack]

        # Create and save the boxplot for each DGP
        create_and_save_boxplot(flattened_stack, dgp_name, fig_dir)

        oracle_errors = [pruning_oracle, interpolated_oracle, interpolated_oracle, two_step_oracle, semi_oracle,semi_oracle,semi_oracle]
        errors_oracle_array4.append(np.median(np.sqrt(oracle_errors), axis=1))

        errors = [pruning_error, global_error,
                  inter_error, two_step_error, semi_error, deep_error, local_error]
        errors_array5.append(np.median(np.sqrt(errors), axis=1))

        oracle_iters = [pruning_iter_oracle, global_iter_oracle, global_iter_oracle, two_step_iter_oracle, semi_iter_oracle]
        oracle_iters_array.append(np.median(oracle_iters, axis=1))


        # Compute the ratios between the oracles
        ratio_pruning_inter = np.sqrt(pruning_oracle/ interpolated_oracle)
        ratio_array1.append(ratio_pruning_inter)

        ratio_pruning_semi = np.sqrt(pruning_oracle/ semi_oracle)
        ratio_array2.append(ratio_pruning_semi)

        ratio_interpolated_semi = np.sqrt(interpolated_oracle/ semi_oracle)
        ratio_array3.append(ratio_interpolated_semi)


    create_and_save_boxplot_ratio_highdim(np.hstack(ratio_array1)[:, 4:8], fig_dir, name='ratio1')
    create_and_save_boxplot_ratio_easy(np.hstack(ratio_array1)[:, 0:4], fig_dir, name='ratio1')

    create_and_save_boxplot_ratio_highdim(np.hstack(ratio_array2)[:, 4:8], fig_dir, name='ratio2')
    create_and_save_boxplot_ratio_easy(np.hstack(ratio_array2)[:, 0:4], fig_dir, name='ratio2')

    create_and_save_boxplot_ratio_highdim(np.hstack(ratio_array3)[:, 4:8], fig_dir, name='ratio3')
    create_and_save_boxplot_ratio_easy(np.hstack(ratio_array3)[:, 0:4], fig_dir, name='ratio3')

    # Additional table for the absolute errors:
    column_names = ["Pruning", "Global", "Global Int", "Two-Step", "Semi", "Deep", "Local"]
    row_names = ['Rectangular', 'Circular', 'Sine Cosine', 'Elliptical','Additive Smooth', 'Additive Step', 'Additive Linear', 'Additive Hills']
    print(simple_latex_table(errors_array5, errors_oracle_array4, column_names, row_names, name_threshold='error_and_oracle_oracle',
                             table_dir=fig_dir))

    # Print the oracle n_leaves
    oracle_leaves = np.vstack([arr.flatten() for arr in oracle_iters_array])
    columns = ["Pruning", "Global", "Global Int", "Two-Step", "Semi"]
    oracle_leaves_df = pd.DataFrame(oracle_leaves, columns=columns, index=row_names)
    print(oracle_leaves_df)

if __name__ == "__main__":
    main()
