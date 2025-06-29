import pandas as pd
import numpy as np

def create_runtime_table():
    """Create runtime comparison table for CCP, ES, TS with CCP/ES ratios"""
    
    # Read data
    empirical_df = pd.read_csv('reporting/results/saved/csvs/dt_empirical_study_median.csv')
    simulation_df = pd.read_csv('reporting/results/saved/csvs/dt_simulation.csv')
    
    methods = ['ES', 'TS', 'CCP']
    
    print("=" * 80)
    print("RUNTIME COMPARISON TABLE (10 iterations)")
    print("=" * 80)
    
    # EMPIRICAL DATA
    print("\nEMPIRICAL STUDIES:")
    print("-" * 50)
    
    empirical_results = []
    for dataset in empirical_df['dataset'].unique():
        dataset_data = empirical_df[empirical_df['dataset'] == dataset]
        
        es_time = dataset_data[dataset_data['method'] == 'ES']['fit_time_mean'].iloc[0] * 10
        ts_time = dataset_data[dataset_data['method'] == 'TS']['fit_time_mean'].iloc[0] * 10
        ccp_time = dataset_data[dataset_data['method'] == 'CCP']['fit_time_mean'].iloc[0] * 10
        
        ratio = ccp_time / es_time
        
        empirical_results.append({
            'Dataset': dataset,
            'CCP (s)': f"{ccp_time:.1f}",
            'ES (s)': f"{es_time:.3f}",
            'TS (s)': f"{ts_time:.2f}",
            'CCP/ES Ratio': f"{ratio:.0f}×"
        })
    
    emp_df = pd.DataFrame(empirical_results)
    print(emp_df.to_string(index=False))
    
    # Calculate empirical means
    emp_es_mean = empirical_df[empirical_df['method'] == 'ES']['fit_time_mean'].mean() * 10
    emp_ts_mean = empirical_df[empirical_df['method'] == 'TS']['fit_time_mean'].mean() * 10
    emp_ccp_mean = empirical_df[empirical_df['method'] == 'CCP']['fit_time_mean'].mean() * 10
    emp_ratio_mean = emp_ccp_mean / emp_es_mean
    
    print(f"\nEMPIRICAL MEANS:")
    print(f"ES: {emp_es_mean:.3f}s | TS: {emp_ts_mean:.2f}s | CCP: {emp_ccp_mean:.1f}s | CCP/ES: {emp_ratio_mean:.0f}×")
    
    # SIMULATION DATA
    print("\n\nSIMULATION STUDIES:")
    print("-" * 50)
    
    simulation_results = []
    for dataset in simulation_df['dataset'].unique():
        dataset_data = simulation_df[simulation_df['dataset'] == dataset]
        
        es_time = dataset_data[dataset_data['method'] == 'ES']['fit_time_median'].iloc[0] * 10
        ts_time = dataset_data[dataset_data['method'] == 'TS']['fit_time_median'].iloc[0] * 10
        ccp_time = dataset_data[dataset_data['method'] == 'CCP']['fit_time_median'].iloc[0] * 10
        
        ratio = ccp_time / es_time
        
        simulation_results.append({
            'Dataset/DGP': dataset,
            'CCP (s)': f"{ccp_time:.1f}",
            'ES (s)': f"{es_time:.3f}",
            'TS (s)': f"{ts_time:.2f}",
            'CCP/ES Ratio': f"{ratio:.0f}×"
        })
    
    sim_df = pd.DataFrame(simulation_results)
    print(sim_df.to_string(index=False))
    
    # Calculate simulation means
    sim_es_mean = simulation_df[simulation_df['method'] == 'ES']['fit_time_median'].mean() * 10
    sim_ts_mean = simulation_df[simulation_df['method'] == 'TS']['fit_time_median'].mean() * 10
    sim_ccp_mean = simulation_df[simulation_df['method'] == 'CCP']['fit_time_median'].mean() * 10
    sim_ratio_mean = sim_ccp_mean / sim_es_mean
    
    print(f"\nSIMULATION MEANS:")
    print(f"ES: {sim_es_mean:.3f}s | TS: {sim_ts_mean:.2f}s | CCP: {sim_ccp_mean:.1f}s | CCP/ES: {sim_ratio_mean:.0f}×")
    
    # SUMMARY TABLE
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    
    summary_data = [
        {
            'Study Type': 'Empirical',
            'CCP (s)': f"{emp_ccp_mean:.1f}",
            'ES (s)': f"{emp_es_mean:.3f}",
            'TS (s)': f"{emp_ts_mean:.2f}",
            'CCP/ES Ratio': f"{emp_ratio_mean:.0f}×"
        },
        {
            'Study Type': 'Simulation',
            'CCP (s)': f"{sim_ccp_mean:.1f}",
            'ES (s)': f"{sim_es_mean:.3f}",
            'TS (s)': f"{sim_ts_mean:.2f}",
            'CCP/ES Ratio': f"{sim_ratio_mean:.0f}×"
        }
    ]
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # CREATE LATEX TABLE
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)
    
    latex_content = []
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Runtime Comparison of Decision Tree Methods (10 iterations)}")
    latex_content.append("\\label{tab:runtime_comparison}")
    latex_content.append("\\begin{tabular}{lcccc}")
    latex_content.append("\\hline")
    latex_content.append("Dataset/DGP & CCP (s) & ES (s) & TS (s) & CCP/ES Ratio \\\\")
    latex_content.append("\\hline")
    latex_content.append("\\multicolumn{5}{l}{\\textbf{Empirical Studies}} \\\\")
    
    # Empirical data
    for _, row in emp_df.iterrows():
        dataset = row['Dataset'].replace('_', '\\_').replace('.', '.')  # Escape for LaTeX
        ccp_time = row['CCP (s)']
        es_time = row['ES (s)']
        ts_time = row['TS (s)']
        ratio = row['CCP/ES Ratio']
        latex_content.append(f"{dataset} & {ccp_time} & {es_time} & {ts_time} & {ratio} \\\\")
    
    latex_content.append("\\hline")
    latex_content.append("\\multicolumn{5}{l}{\\textbf{Simulation Studies}} \\\\")
    
    # Simulation data
    for _, row in sim_df.iterrows():
        dataset = row['Dataset/DGP'].replace('_', '\\_').replace('.', '.')  # Escape for LaTeX
        ccp_time = row['CCP (s)']
        es_time = row['ES (s)']
        ts_time = row['TS (s)']
        ratio = row['CCP/ES Ratio']
        latex_content.append(f"{dataset} & {ccp_time} & {es_time} & {ts_time} & {ratio} \\\\")
    
    latex_content.append("\\hline")
    
    # Summary row
    latex_content.append("\\multicolumn{5}{l}{\\textbf{Mean Runtime}} \\\\")
    latex_content.append(f"Empirical & {emp_ccp_mean:.1f} & {emp_es_mean:.3f} & {emp_ts_mean:.2f} & {emp_ratio_mean:.0f}× \\\\")
    latex_content.append(f"Simulation & {sim_ccp_mean:.1f} & {sim_es_mean:.3f} & {sim_ts_mean:.2f} & {sim_ratio_mean:.0f}× \\\\")
    
    latex_content.append("\\hline")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\begin{tablenotes}")
    latex_content.append("\\item Runtime measured in seconds for 10 iterations.")
    latex_content.append("\\item CCP: Cost Complexity Pruning, ES: Early Stopping, TS: Two-Step")
    latex_content.append("\\end{tablenotes}")
    latex_content.append("\\end{table}")
    
    latex_table = "\n".join(latex_content)
    print(latex_table)
    
    # Save results
    emp_df.to_csv('results/runtime_empirical.csv', index=False)
    sim_df.to_csv('results/runtime_simulation.csv', index=False)
    summary_df.to_csv('results/runtime_summary.csv', index=False)
    
    # Save LaTeX table
    with open('results/runtime_table.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"\n\nFiles saved:")
    print(f"- results/runtime_empirical.csv")
    print(f"- results/runtime_simulation.csv") 
    print(f"- results/runtime_summary.csv")
    print(f"- results/runtime_table.tex")

if __name__ == "__main__":
    create_runtime_table() 