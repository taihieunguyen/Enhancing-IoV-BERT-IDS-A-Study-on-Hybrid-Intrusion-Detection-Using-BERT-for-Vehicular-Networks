import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Data from eval_results.txt files
data = {
    'BoT-OiT': {
        'eval_loss': 0.0002460483810864389,
        'eval_accuracy': 0.99996625,
        'eval_precision': 0.9999661712473327,
        'eval_recall': 0.99996625,
        'eval_f1': 0.9999661836554236,
        'eval_runtime': 2526.7612,
        'eval_samples_per_second': 316.611,
        'eval_steps_per_second': 9.894,
        'eval_model_preparation_time': 0.0025
    },
    'Car-Hacking': {
        'eval_loss': 0.4699828624725342,
        'eval_accuracy': 0.8471777168558449,
        'eval_precision': 0.717710083937082,
        'eval_recall': 0.8471777168558449,
        'eval_f1': 0.7770882870530996,
        'eval_runtime': 9891.1924,
        'eval_samples_per_second': 335.035,
        'eval_steps_per_second': 10.47,
        'eval_model_preparation_time': 0.0026
    },
    'CICIDS-2017': {
        'eval_loss': 0.8723333477973938,
        'eval_accuracy': 0.8857018976117748,
        'eval_precision': 0.887488118253634,
        'eval_recall': 0.8857018976117748,
        'eval_f1': 0.8865941082591914,
        'eval_runtime': 1846.8476,
        'eval_samples_per_second': 277.236,
        'eval_steps_per_second': 8.664,
        'eval_model_preparation_time': 0.0028
    },
    'IVN-IDS': {
        'eval_loss': 0.5134580135345459,
        'eval_accuracy': 0.7817261221720594,
        'eval_precision': 0.6110957300861656,
        'eval_recall': 0.7817261221720594,
        'eval_f1': 0.6859592195249329,
        'eval_runtime': 675.0461,
        'eval_samples_per_second': 353.721,
        'eval_steps_per_second': 11.054,
        'eval_model_preparation_time': 0.0012
    }
}

# Create DataFrame from the data
df = pd.DataFrame(data)

# Create comparison table
comparison_table = df.T

# Create a DataFrame to display main indices as percentages
percentage_metrics = ['eval_accuracy', 'eval_precision', 'eval_recall', 'eval_f1']
df_percentage = comparison_table[percentage_metrics] * 100

# Create a DataFrame to display performance indices
performance_metrics = ['eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second']
df_performance = comparison_table[performance_metrics]

# Display full comparison table
print("Full comparison table:")
print(comparison_table)

# Display percentage comparison for evaluation metrics
print("\nEvaluation metrics comparison (%):")
percentage_formatted = df_percentage.astype(str) + '%'
print(percentage_formatted)

# Display loss comparison
print("\nLoss comparison:")
loss_comparison = comparison_table[['eval_loss']]
print(loss_comparison)

# Create a summary table of important metrics
important_metrics = ['eval_loss', 'eval_accuracy', 'eval_precision', 'eval_recall', 'eval_f1']
summary_table = comparison_table[important_metrics]
print("\nSummary table of important metrics:")
print(summary_table)

# Set style for charts
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")  # Use colorblind-friendly palette for better distinction
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 1. Create evaluation metrics chart - IMPROVED VERSION WITH SUBPLOTS
metrics_to_plot = ['eval_accuracy', 'eval_precision', 'eval_recall', 'eval_f1']

# Create a figure with two subplots: one for BoT-OiT and one for the rest
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), height_ratios=[1, 3])

# Plot BoT-OiT separately with a different y-scale in the top subplot
bot_oit_data = comparison_table.loc[['BoT-OiT'], metrics_to_plot]
bot_oit_data.plot(kind='bar', ax=ax1, color=sns.color_palette("colorblind")[0:4])
ax1.set_title('BoT-OiT Evaluation Metrics', fontsize=16)
ax1.set_ylabel('Value', fontsize=14)
ax1.set_ylim(0.9998, 1.0001)  # Narrow y-range for visibility of high values
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.legend(title='Metrics', fontsize=12)

# Add values on top of each bar for BoT-OiT
for container in ax1.containers:
    ax1.bar_label(container, fmt='%.8f', fontsize=9, rotation=45, padding=3)

# Plot other datasets in the bottom subplot with a different y-scale
other_data = comparison_table.drop('BoT-OiT').loc[:, metrics_to_plot]
other_data.plot(kind='bar', ax=ax2, color=sns.color_palette("colorblind")[0:4])
ax2.set_title('Other Datasets Evaluation Metrics', fontsize=16)
ax2.set_ylabel('Value', fontsize=14)
ax2.set_xlabel('Dataset', fontsize=14)
ax2.set_ylim(0.5, 0.95)  # Adjusted y-range for better visibility
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.legend(title='Metrics', fontsize=12)

# Add values on top of each bar for other datasets
for container in ax2.containers:
    ax2.bar_label(container, fmt='%.4f', fontsize=9, rotation=45, padding=3)

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Create comparison chart for eval_loss
fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette("colorblind")
bars = ax.bar(comparison_table.index, comparison_table['eval_loss'], color=colors)
plt.title('Loss Comparison', fontsize=16)
plt.ylabel('eval_loss', fontsize=14)
plt.xlabel('Dataset', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top of each bar - show 8 decimal places for BoT-OiT
for i, bar in enumerate(bars):
    height = bar.get_height()
    if comparison_table.index[i] == 'BoT-OiT':
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.8f}', ha='center', va='bottom', rotation=0, fontsize=10)
    else:
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', rotation=0, fontsize=10)

plt.tight_layout()
plt.savefig('loss_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Create image table WITHOUT rounding for BoT-OiT
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

# Format the data for display - keep original precision for BoT-OiT
table_data = []
for idx, row in summary_table.iterrows():
    if idx == 'BoT-OiT':
        # Keep full precision for BoT-OiT
        table_data.append([f"{val:.8f}" if i == 0 else f"{val:.8f}" for i, val in enumerate(row)])
    else:
        # Round to 4 decimal places for other datasets
        table_data.append([f"{val:.4f}" for val in row])

table = ax.table(cellText=table_data,
                 rowLabels=summary_table.index,
                 colLabels=summary_table.columns,
                 cellLoc='center',
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
plt.title('Summary Table of Important Metrics', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('summary_table.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Create full comparison table visualization
fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('tight')
ax.axis('off')

# Format the data for display - keep original precision for BoT-OiT
all_metrics = ['eval_loss', 'eval_accuracy', 'eval_precision', 'eval_recall', 'eval_f1', 
               'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second', 'eval_model_preparation_time']
full_table = comparison_table[all_metrics]

# Create formatted data for table display
table_data = []
for idx, row in full_table.iterrows():
    formatted_row = []
    for col, val in zip(full_table.columns, row):
        if idx == 'BoT-OiT':
            # High precision for BoT-OiT
            if col == 'eval_loss':
                formatted_row.append(f"{val:.8f}")
            elif col in percentage_metrics:
                formatted_row.append(f"{val:.8f}")
            elif col in ['eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second']:
                formatted_row.append(f"{val:.3f}")
            else:
                formatted_row.append(f"{val:.8f}")
        else:
            # Regular precision for other datasets
            if col == 'eval_loss':
                formatted_row.append(f"{val:.6f}")
            elif col in percentage_metrics:
                formatted_row.append(f"{val:.6f}")
            elif col in ['eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second']:
                formatted_row.append(f"{val:.3f}")
            else:
                formatted_row.append(f"{val:.6f}")
    table_data.append(formatted_row)

# Create the table
table = ax.table(cellText=table_data,
                 rowLabels=full_table.index,
                 colLabels=full_table.columns,
                 cellLoc='center',
                 loc='center')

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

# Add a title
plt.title('Full Comparison Table', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('full_comparison_table.png', dpi=300, bbox_inches='tight')
plt.close()