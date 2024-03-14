import matplotlib.pyplot as plt

# Data
num_workers = [0, 4, 8, 12, 16]
data_loading_times = [71.173484972, 3.425580946, 2.412396077, 2.260822554, 2.741398371]
total_running_times = [160.378956894, 108.731285594, 109.435557806, 109.559673072, 109.327103911]

# Plotting
plt.figure(figsize=(6, 4))
bar_width = 1.25
plt.bar(num_workers, data_loading_times, bar_width, color='b', alpha=0.7, label='Data-loading Time')
plt.bar([p + bar_width for p in num_workers], total_running_times, bar_width, color='r', alpha=0.7, label='Total Running Time')

# Adding labels and title
plt.xlabel('Number of Workers')
plt.ylabel('Time (seconds)')
plt.title('Performance vs. Number of Workers')
plt.xticks([p + bar_width/2 for p in num_workers], num_workers)
plt.legend()

# Adding annotations
for i, time in enumerate(data_loading_times):
    plt.annotate(f'{time:.2f}', (num_workers[i], time), textcoords="offset points", xytext=(0,2), ha='center', fontsize=8)
for i, time in enumerate(total_running_times):
    plt.annotate(f'{time:.2f}', (num_workers[i] + bar_width, time), textcoords="offset points", xytext=(0,2), ha='center', fontsize=8)

plt.tight_layout()

# Show plot
plt.show()
