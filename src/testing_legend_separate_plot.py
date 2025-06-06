import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Create some data.
x = np.linspace(0, 10, 100)
y = np.sin(x)
y_upper = y + 0.5
y_lower = y - 0.5

# Create the main figure.
fig, ax = plt.subplots()
line, = ax.plot(x, y, label='Sine Wave')
# The fill_between creates a PolyCollection that cannot be reused directly.
fill = ax.fill_between(x, y_lower, y_upper, color='blue', alpha=0.2, label='Confidence Interval')

# Instead of reusing the fill_between artist, we create a custom handle.
custom_fill = mpatches.Patch(color='blue', alpha=0.2, label='Confidence Interval')

# Get the existing handles and labels.
handles, labels = ax.get_legend_handles_labels()

# Replace the fill_between handle with the custom one.
new_handles = []
new_labels = []
for h, lab in zip(handles, labels):
    if lab == 'Confidence Interval':
        new_handles.append(custom_fill)
    else:
        new_handles.append(h)
    new_labels.append(lab)

# Create a separate figure for the legend.
figLegend = plt.figure(figsize=(3, 3))
figLegend.legend(new_handles, new_labels, loc='center', fontsize=12)

# Save or show the figures.
fig.savefig('main_plot.png', bbox_inches='tight')
figLegend.savefig('legend_plot.png', bbox_inches='tight')
plt.show()
