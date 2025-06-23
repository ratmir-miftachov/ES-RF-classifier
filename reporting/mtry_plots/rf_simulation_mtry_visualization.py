import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set minimalistic style
plt.rcParams.update({
    'font.size': 10,
    'axes.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

# Read the data
df = pd.read_csv('reporting/results/saved/csvs/rf_simulation_B50.csv')

# Filter for IGES and MD_scikit methods only
methods_of_interest = ['IGES', 'IGES_1', 'IGES_d', 'MD_scikit', 'MD_scikit_1', 'MD_scikit_d']
df_filtered = df[df['method'].isin(methods_of_interest)].copy()

# Extract base method and mtry setting
df_filtered['base_method'] = df_filtered['method'].apply(lambda x: 'ES random forest (IGES)' if 'IGES' in x else 'deep random forest')
df_filtered['mtry_setting'] = df_filtered['method'].apply(
    lambda x: '1' if x.endswith('_1') else 'd' if x.endswith('_d') else '√d'
)

# Dataset order - all 8 DGPs (additive models + low dimensional)
dataset_order = [
    "Add. H.I. Jump", "Add. Het.", "Add. Jump", "Add. Smooth",
    "Circular", "Circular Smooth", "Rectangular", "Sine Cosine"
]

# Create the visualization with all 8 DGPs in 2x4 layout
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Matthews Correlation Coefficient: ES Random Forest (IGES) vs Deep Random Forest', fontsize=14, y=0.98)

# Minimalistic color scheme - back to original colors
uges_color = '#2E86AB'  # Blue for ES methods (keeping original name for consistency)
md_color = '#A23B72'    # Purple for deep

# Flatten axes for easier iteration
axes_flat = axes.flatten()

# Create MCC plots for all 8 datasets
for dataset_idx, dataset in enumerate(dataset_order):
    ax = axes_flat[dataset_idx]
    
    # Filter data for current dataset
    dataset_data = df_filtered[df_filtered['dataset'] == dataset]
    
    if dataset_data.empty:
        ax.set_title(f'{dataset}\n(No data)')
        continue
    
    # Prepare data for plotting
    x_positions = np.arange(3)  # Three mtry settings
    width = 0.3  # Narrower bars for cleaner look
    
    # Get data for each method and mtry setting
    es_data = []
    deep_data = []
    mtry_labels = ['1', '√d', 'd']
    
    for mtry in mtry_labels:
        es_row = dataset_data[(dataset_data['base_method'] == 'ES random forest (IGES)') & 
                             (dataset_data['mtry_setting'] == mtry)]
        deep_row = dataset_data[(dataset_data['base_method'] == 'deep random forest') & 
                               (dataset_data['mtry_setting'] == mtry)]
        
        es_score = es_row['test_mcc_median'].iloc[0] if not es_row.empty else 0
        deep_score = deep_row['test_mcc_median'].iloc[0] if not deep_row.empty else 0
        
        es_data.append(es_score)
        deep_data.append(deep_score)
    
    # Create clean bars without patterns (deep random forest on left, ES random forest on right)
    bars1 = ax.bar(x_positions - width/2, deep_data, width,
                   label='deep random forest', color=md_color, alpha=0.8,
                   edgecolor='white', linewidth=1)
    
    bars2 = ax.bar(x_positions + width/2, es_data, width, 
                   label='ES random forest (IGES)', color=uges_color, alpha=0.8, 
                   edgecolor='white', linewidth=1)
    
    # Customize subplot with minimal styling
    ax.set_title(f'{dataset}', fontsize=11, pad=10)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['1', '√d', 'd'], fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=9)
    
    # Only show horizontal grid lines
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set fixed y-axis limits
    ax.set_ylim(0, 1)
    
    # Add x-axis label to bottom row
    if dataset_idx >= 4:  # Bottom row
        ax.set_xlabel('mtry', fontsize=9)
    
    # Only add y-axis label to leftmost plots
    if dataset_idx % 4 == 0:
        ax.set_ylabel('MCC', fontsize=9)
    else:
        ax.set_ylabel('')

# Create simple legend (matching left-to-right order)
legend_elements = [
    plt.Rectangle((0,0),1,1, facecolor=md_color, alpha=0.8, label='deep random forest'),
    plt.Rectangle((0,0),1,1, facecolor=uges_color, alpha=0.8, label='ES random forest (IGES)')
]

# Note: All 8 subplots are now used (4 additive + 4 low dimensional DGPs)
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
          ncol=2, fontsize=10, frameon=False)

# Adjust layout for minimal spacing
plt.tight_layout()
plt.subplots_adjust(top=0.88, hspace=0.3, wspace=0.2)

# Save the plot
plt.savefig('results/rf_comparison_plot.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

print("Visualization saved as 'results/rf_comparison_plot.png'")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

for dataset in dataset_order:
    dataset_data = df_filtered[df_filtered['dataset'] == dataset]
    if dataset_data.empty:
        continue
        
    print(f"\n{dataset}:")
    print("-" * len(dataset))
    
    for method in ['ES random forest (IGES)', 'deep random forest']:
        method_data = dataset_data[dataset_data['base_method'] == method]
        if method_data.empty:
            continue
            
        print(f"\n{method}:")
        for mtry in ['1', '√d', 'd']:
            mtry_data = method_data[method_data['mtry_setting'] == mtry]
            if not mtry_data.empty:
                mcc = mtry_data['test_mcc_median'].iloc[0]
                acc = mtry_data['test_acc_median'].iloc[0]
                depth = mtry_data['depth_median'].iloc[0]
                leaves = mtry_data['n_leaves_median'].iloc[0]
                time = mtry_data['fit_time_median'].iloc[0]
                print(f"  mtry={mtry}: MCC={mcc:.3f}, Acc={acc:.3f}, Depth={depth:.1f}, Leaves={leaves:.1f}, Time={time:.3f}s") 