import pandas as pd
import numpy as np

def create_mtry_comparison_latex_table():
    """
    Create LaTeX table for sklearn vs UGES mtry comparison results.
    Table consists of three subtables, one for each mtry specification.
    """
    # Read the CSV file
    df = pd.read_csv('../../results/sklearn_depth_sample_size_mtry_comparison_p_0.8.csv')
    
    # Define sample sizes in order
    sample_sizes = [2000, 5000, 10000, 20000, 50000]
    
    # Define mtry specifications
    mtry_specs = ['1', 'sqrt', 'd']
    mtry_labels = {
        '1': 'mtry = 1',
        'sqrt': 'mtry = √d', 
        'd': 'mtry = d'
    }
    
    # Initialize LaTeX string
    latex_str = r"""\begin{table}[htbp]
\centering
\caption{Runtime Comparison: sklearn vs UGES across different mtry specifications}
\label{tab:mtry_comparison}
\begin{tabular}{c|c|c|c}
\hline
Sample Size & MD-RF & UGES & MD-RF/UGES Ratio \\
\hline
"""
    
    # Create each subtable
    for mtry_spec in mtry_specs:
        # Add subtable header
        latex_str += f"\n\\multicolumn{{4}}{{c}}{{{mtry_labels[mtry_spec]}}} \\\\\n\\hline\n"
        
        # Filter data for this mtry specification
        mtry_data = df[df['mtry_spec'] == mtry_spec]
        
        for sample_size in sample_sizes:
            size_data = mtry_data[mtry_data['n_samples'] == sample_size]
            
            # Get MD-RF (sklearn_unlimited) runtime
            md_rf_data = size_data[size_data['method'] == 'sklearn_unlimited']
            md_rf_time = md_rf_data['fit_time_median'].iloc[0] if len(md_rf_data) > 0 else np.nan
            
            # Get UGES (sklearn_optimal) runtime  
            uges_data = size_data[size_data['method'] == 'sklearn_optimal']
            uges_time = uges_data['fit_time_median'].iloc[0] if len(uges_data) > 0 else np.nan
            
            # Get ratio
            ratio_data = size_data[size_data['method'] == 'sklearn_unlimited_vs_optimal_ratio']
            ratio = ratio_data['fit_time_median'].iloc[0] if len(ratio_data) > 0 else np.nan
            
            # Format values
            md_rf_str = f"{md_rf_time:.3f}" if not np.isnan(md_rf_time) else "N/A"
            uges_str = f"{uges_time:.3f}" if not np.isnan(uges_time) else "N/A"
            ratio_str = f"{ratio:.2f}" if not np.isnan(ratio) else "N/A"
            
            # Add row to table
            latex_str += f"{sample_size:,} & {md_rf_str} & {uges_str} & {ratio_str} \\\\\n"
        
        # Add separator between subtables (except for last one)
        if mtry_spec != mtry_specs[-1]:
            latex_str += "\\hline\n"
    
    # Close table
    latex_str += r"""\hline
\end{tabular}
\begin{tablenotes}
    \small
    \item Note: MD-RF refers to sklearn RandomForest with unlimited depth. UGES refers to sklearn RandomForest with UGES-determined optimal depth. Runtime is measured in seconds (median across 10 Monte Carlo iterations). 
    \item The ratio shows how many times longer MD-RF takes compared to UGES.
\end{tablenotes}
\end{table}"""
    
    return latex_str

def create_accuracy_comparison_latex_table():
    """
    Create LaTeX table showing accuracy comparison across mtry specifications.
    """
    # Read the CSV file
    df = pd.read_csv('../../results/sklearn_depth_sample_size_mtry_comparison_p_0.8.csv')
    
    # Define sample sizes in order
    sample_sizes = [2000, 5000, 10000, 20000, 50000]
    
    # Define mtry specifications
    mtry_specs = ['1', 'sqrt', 'd']
    mtry_labels = {
        '1': 'mtry = 1',
        'sqrt': 'mtry = √d', 
        'd': 'mtry = d'
    }
    
    # Initialize LaTeX string
    latex_str = r"""\begin{table}[htbp]
\centering
\caption{Test Accuracy Comparison: sklearn vs UGES across different mtry specifications}
\label{tab:mtry_accuracy_comparison}
\begin{tabular}{c|c|c|c}
\hline
Sample Size & MD-RF & UGES & Difference \\
\hline
"""
    
    # Create each subtable
    for mtry_spec in mtry_specs:
        # Add subtable header
        latex_str += f"\n\\multicolumn{{4}}{{c}}{{{mtry_labels[mtry_spec]}}} \\\\\n\\hline\n"
        
        # Filter data for this mtry specification
        mtry_data = df[df['mtry_spec'] == mtry_spec]
        
        for sample_size in sample_sizes:
            size_data = mtry_data[mtry_data['n_samples'] == sample_size]
            
            # Get MD-RF (sklearn_unlimited) accuracy
            md_rf_data = size_data[size_data['method'] == 'sklearn_unlimited']
            md_rf_acc = md_rf_data['test_acc_median'].iloc[0] if len(md_rf_data) > 0 else np.nan
            
            # Get UGES (sklearn_optimal) accuracy
            uges_data = size_data[size_data['method'] == 'sklearn_optimal']
            uges_acc = uges_data['test_acc_median'].iloc[0] if len(uges_data) > 0 else np.nan
            
            # Calculate difference
            diff = md_rf_acc - uges_acc if (not np.isnan(md_rf_acc) and not np.isnan(uges_acc)) else np.nan
            
            # Format values
            md_rf_str = f"{md_rf_acc:.3f}" if not np.isnan(md_rf_acc) else "N/A"
            uges_str = f"{uges_acc:.3f}" if not np.isnan(uges_acc) else "N/A"
            diff_str = f"{diff:+.3f}" if not np.isnan(diff) else "N/A"
            
            # Add row to table
            latex_str += f"{sample_size:,} & {md_rf_str} & {uges_str} & {diff_str} \\\\\n"
        
        # Add separator between subtables (except for last one)
        if mtry_spec != mtry_specs[-1]:
            latex_str += "\\hline\n"
    
    # Close table
    latex_str += r"""\hline
\end{tabular}
\begin{tablenotes}
    \small
    \item Note: MD-RF refers to sklearn RandomForest with unlimited depth. UGES refers to sklearn RandomForest with UGES-determined optimal depth. Test accuracy is median across 10 Monte Carlo iterations.
    \item Difference = MD-RF accuracy - UGES accuracy. Negative values indicate UGES performs better.
\end{tablenotes}
\end{table}"""
    
    return latex_str

if __name__ == "__main__":
    # Create runtime comparison table
    runtime_table = create_mtry_comparison_latex_table()
    
    # Create accuracy comparison table  
    accuracy_table = create_accuracy_comparison_latex_table()
    
    # Save to file
    with open('../../results/mtry_comparison_runtime_table.tex', 'w') as f:
        f.write(runtime_table)
    
    with open('../../results/mtry_comparison_accuracy_table.tex', 'w') as f:
        f.write(accuracy_table)
    
    print("LaTeX tables created:")
    print("- ../../results/mtry_comparison_runtime_table.tex")
    print("- ../../results/mtry_comparison_accuracy_table.tex")
    
    print("\nRuntime Table Preview:")
    print("="*60)
    print(runtime_table)
    
    print("\n\nAccuracy Table Preview:")
    print("="*60)
    print(accuracy_table)