import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# Trimodal-like function on [0,2] (but actually two peaks in this code).
def f(x):
    peak1 = 0.5 * np.exp(-((x - 0.5)**2) / (2 * 0.04))
    peak2 = 0.7 * np.exp(-((x - 1.5)**2) / (2 * 0.04))
    return peak1 + peak2

# Grid for plotting
X_grid = np.linspace(0, 2, 500).reshape(-1, 1)
y_true = f(X_grid)
ymin, ymax = y_true.min() - 0.1, y_true.max() + 3.1

# Initial data
X_train = np.array([[0.5], [1.0]])
y_train = f(X_train)

# GP model
kernel = ConstantKernel(1.0, (0.1, 10.0)) * RBF(length_scale=0.1, length_scale_bounds=(0.01, 1.0))
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=5)
gp.fit(X_train, y_train)

# Step 1: single-sample TS
sample1 = gp.sample_y(X_grid, n_samples=1, random_state=0).ravel()
f_star_1 = sample1.max()  # maximum of the single TS sample
idx1 = np.argmax(sample1)
x1 = X_grid[idx1]
mean1, std1 = gp.predict(X_grid, return_std=True)
y1_fantasy = mean1[idx1]

# Update data
X_train2 = np.vstack([X_train, x1.reshape(1, -1)])
y_train2 = np.append(y_train, y1_fantasy)

# Step 2: new GP, single-sample TS
gp2 = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=5)
gp2.fit(X_train2, y_train2)
sample2 = gp2.sample_y(X_grid, n_samples=1, random_state=0).ravel()
f_star_2 = sample2.max()
idx2 = np.argmax(sample2)
x2 = X_grid[idx2]
mean2, std2 = gp2.predict(X_grid, return_std=True)
y2_fantasy = mean2[idx2]

# Update data
X_train3 = np.vstack([X_train2, x2.reshape(1, -1)])
y_train3 = np.append(y_train2, y2_fantasy)

# Step 3: final GP, single-sample TS
gp3 = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=5)
gp3.fit(X_train3, y_train3)
sample3 = gp3.sample_y(X_grid, n_samples=1, random_state=0).ravel()
f_star_3 = sample3.max()
idx3 = np.argmax(sample3)
x3 = X_grid[idx3]
mean3, std3 = gp3.predict(X_grid, return_std=True)
y3_fantasy = mean3[idx3]

# Plot
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# ---- Plot 1 ----
axs[0].plot(X_grid, y_true, 'r--', label='True function')
axs[0].scatter(X_train, y_train, c='black', zorder=10, label='Initial Observations')
axs[0].plot(X_grid, mean1, 'b-', label='GP Mean')
axs[0].fill_between(X_grid.ravel(), mean1 - 2*std1, mean1 + 2*std1, 
                    color='blue', alpha=0.2, label='Uncertainty Band')
axs[0].plot(X_grid, sample1, 'c-', label='TS Sample')
axs[0].axvline(x1, color='blue', linestyle='--', label='Selected Point')
axs[0].scatter(x1, y1_fantasy, color='blue', s=100, zorder=10)
# Horizontal line for f_star_1
axs[0].axhline(f_star_1, color='gray', linestyle='--', label='TS sample max')

axs[0].set_title("Thompson Sampling: Step 1")
axs[0].set_xlabel("x")
axs[0].set_ylabel("f(x)")
axs[0].set_ylim(ymin, ymax)
axs[0].legend(fontsize=8)

# ---- Plot 2 ----
axs[1].plot(X_grid, y_true, 'r--', label='True function')
axs[1].scatter(X_train2, y_train2, c='black', zorder=10, label='Obs + Fantasy')
axs[1].plot(X_grid, mean2, 'g-', label='Updated GP Mean')
axs[1].fill_between(X_grid.ravel(), mean2 - 2*std2, mean2 + 2*std2, 
                    color='green', alpha=0.2, label='Uncertainty Band')
axs[1].plot(X_grid, sample2, 'm-', label='TS Sample')
axs[1].axvline(x2, color='green', linestyle='--', label='Selected Point')
axs[1].scatter(x2, y2_fantasy, color='green', s=100, zorder=10)
# Horizontal line for f_star_2
axs[1].axhline(f_star_2, color='gray', linestyle='--', label='TS sample max')

axs[1].set_title("Thompson Sampling: Step 2")
axs[1].set_xlabel("x")
axs[1].set_ylim(ymin, ymax)
axs[1].legend(fontsize=8)

# ---- Plot 3 ----
axs[2].plot(X_grid, y_true, 'r--', label='True function')
axs[2].scatter(X_train3, y_train3, c='black', zorder=10, label='Obs + Fantasy')
axs[2].plot(X_grid, mean3, color='magenta', label='Updated GP Mean')
axs[2].fill_between(X_grid.ravel(), mean3 - 2*std3, mean3 + 2*std3, 
                    color='magenta', alpha=0.2, label='Uncertainty Band')
axs[2].plot(X_grid, sample3, 'k-', label='TS Sample')
axs[2].axvline(x3, color='magenta', linestyle='--', label='Selected Point')
axs[2].scatter(x3, y3_fantasy, color='magenta', s=100, zorder=10)
# Horizontal line for f_star_3
axs[2].axhline(f_star_3, color='gray', linestyle='--', label='TS sample max')

axs[2].set_title("Thompson Sampling: Step 3")
axs[2].set_xlabel("x")
axs[2].set_ylim(ymin, ymax)
axs[2].legend(fontsize=8)

plt.tight_layout()
plt.show()

print("Thompson Sampling selected points:")
print("Step 1:", x1.ravel())
print("Step 2:", x2.ravel())
print("Step 3:", x3.ravel())
