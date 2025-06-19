import pandas as pd
import numpy as np

def create_latex_tables():
    # Read the CSV files
    dt_df = pd.read_csv('results/dt_simulation.csv')
    rf_df = pd.read_csv('results/rf_simulation_B30.csv')
    
    # Define the datasets to include (in order)
    datasets = [
        "Add. H.I. Jump",
        "Add. Het.", 
        "Add. Jump",
        "Add. Smooth",
        "Circular",
        "Circular Smooth",
        "Rectangular",
        "Sine Cosine"
    ]
    
    # Define column mapping for methods/algorithms
    # DT methods: MD, CCP, ES, TS
    # RF methods: MD_scikit, IGES, UGES
    dt_method_mapping = {
        'MD': 'MD',
        'CCP': 'CCP', 
        'ES': 'ES',
        'TS': 'TS'
    }
    
    rf_method_mapping = {
        'MD_scikit': 'MD_scikit',
        'IGES': 'IGES',
        'UGES': 'UGES'
    }
    
    # Column order as requested
    column_order = ['MD', 'CCP', 'ES', 'TS', 'MD_scikit', 'IGES', 'UGES']
    
    # Create empty dataframes for each table
    test_acc_table = pd.DataFrame(index=datasets, columns=column_order)
    test_mcc_table = pd.DataFrame(index=datasets, columns=column_order)
    depth_nodes_table = pd.DataFrame(index=datasets, columns=column_order)
    
    # Fill DT data
    for dataset in datasets:
        dt_data = dt_df[dt_df['dataset'] == dataset]
        
        for method, col_name in dt_method_mapping.items():
            method_data = dt_data[dt_data['method'] == method]
            if not method_data.empty:
                row = method_data.iloc[0]
                test_acc_table.loc[dataset, col_name] = f"{row['test_acc_median']:.2f}"
                test_mcc_table.loc[dataset, col_name] = f"{row['test_mcc_median']:.2f}"
                depth_nodes_table.loc[dataset, col_name] = f"{row['depth_median']:.1f} ({row['n_leaves_median']:.1f})"
    
    # Fill RF data
    for dataset in datasets:
        rf_data = rf_df[rf_df['dataset'] == dataset]
        
        for method, col_name in rf_method_mapping.items():
            method_data = rf_data[rf_data['method'] == method]
            if not method_data.empty:
                row = method_data.iloc[0]
                test_acc_table.loc[dataset, col_name] = f"{row['test_acc_median']:.2f}"
                test_mcc_table.loc[dataset, col_name] = f"{row['test_mcc_median']:.2f}"
                depth_nodes_table.loc[dataset, col_name] = f"{row['depth_median']:.1f} ({row['n_leaves_median']:.1f})"
    
    # Generate LaTeX tables
    def df_to_latex_table(df, caption, label):
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += f"\\label{{{label}}}\n"
        latex += "\\begin{tabular}{l" + "c" * len(df.columns) + "}\n"
        latex += "\\toprule\n"
        
        # Header
        latex += "Dataset & " + " & ".join(df.columns) + " \\\\\n"
        latex += "\\midrule\n"
        
        # Data rows
        for dataset in df.index:
            row_data = []
            for col in df.columns:
                value = df.loc[dataset, col]
                if pd.isna(value) or value is None:
                    row_data.append("-")
                else:
                    row_data.append(str(value))
            latex += dataset + " & " + " & ".join(row_data) + " \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n\n"
        
        return latex
    
    # Generate all three tables
    print("=== LaTeX Table 1: Test Accuracy (Simulation) ===")
    print(df_to_latex_table(test_acc_table, "Test Accuracy Results - Simulation Study", "tab:test_acc_sim"))
    
    print("=== LaTeX Table 2: Test MCC (Simulation) ===")
    print(df_to_latex_table(test_mcc_table, "Test Matthews Correlation Coefficient Results - Simulation Study", "tab:test_mcc_sim"))
    
    print("=== LaTeX Table 3: Depth with Number of Nodes (Simulation) ===")
    print(df_to_latex_table(depth_nodes_table, "Tree Depth with Number of Leaves (in brackets) - Simulation Study", "tab:depth_nodes_sim"))

if __name__ == "__main__":
    create_latex_tables() 