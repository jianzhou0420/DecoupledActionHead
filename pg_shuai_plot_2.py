import pandas as pd
import matplotlib.pyplot as plt

# Apply a modern, clean style to the plots
plt.style.use('seaborn-v0_8-darkgrid')

# --- Data Definition ---
# Data for Group 1 with the 'tfr' metric
data_group1 = {
    'steps': [25, 50, 100],
    'fid': [29.5561, 26.1112, 24.9252],
    'chr': [12.95, 13.85, 14.55],
    'ncrf': [18.06, 12.51, 10.63],
    'tfr': [31.01, 26.36, 25.19]
}
df1 = pd.DataFrame(data_group1)

# Data for Group 2 with the 'tfr' metric
data_group2 = {
    'steps': [25, 50, 100],
    'fid': [24.1521, 23.1295, 24.3040],
    'chr': [14.48, 15.99, 15.43],
    'ncrf': [9.33, 7.22, 8.94],
    'tfr': [23.82, 23.21, 24.38]
}
df2 = pd.DataFrame(data_group2)

# Define a consistent color and marker scheme
colors = {'fid': '#1f77b4', 'chr': '#ff7f0e', 'ncrf': '#2ca02c', 'tfr': '#9467bd'}
markers = {'fid': 'o', 'chr': 's', 'ncrf': '^', 'tfr': 'd'}

# List of all metrics to plot
metrics_to_plot = ['fid', 'chr', 'ncrf', 'tfr']

# --- Calculate a consistent Y-axis range for both plots ---
# Combine all relevant data points to find the global min and max
all_data_values = pd.concat([
    df1[metrics_to_plot].stack(),
    df2[metrics_to_plot].stack()
])
y_min = all_data_values.min()
y_max = all_data_values.max()

# Add a small buffer to the y-axis limits for better visualization
y_buffer = (y_max - y_min) * 0.1
y_limits = (y_min - y_buffer, y_max + y_buffer)

# --- Plotting the First Group as a standalone figure ---
plt.figure(figsize=(10, 7))

for metric in metrics_to_plot:
    plt.plot(df1['steps'], df1[metric],
             marker=markers[metric],
             label=metric.upper(),
             color=colors[metric])

plt.title('Performance Metrics - Group 1', fontsize=16, fontweight='bold')
plt.xlabel('Steps', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.xticks(df1['steps'])
plt.ylim(y_limits)  # Apply the consistent y-axis limits
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Metrics')
plt.tight_layout()
plt.savefig('performance_group1.png')
plt.show()
plt.close()

# --- Plotting the Second Group as a standalone figure ---
plt.figure(figsize=(10, 7))

for metric in metrics_to_plot:
    plt.plot(df2['steps'], df2[metric],
             marker=markers[metric],
             label=metric.upper(),
             color=colors[metric])

plt.title('Performance Metrics - Group 2', fontsize=16, fontweight='bold')
plt.xlabel('Steps', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.xticks(df2['steps'])
plt.ylim(y_limits)  # Apply the consistent y-axis limits
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Metrics')
plt.tight_layout()
plt.savefig('performance_group2.png')
plt.show()
plt.close()

print("Two separate plots with a consistent y-axis have been saved as 'performance_group1.png' and 'performance_group2.png'")
