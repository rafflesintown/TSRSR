import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

# 0) Setup
# ----------------------------------------------------
# Make a folder for outputs.
outdir = os.path.join("visualization", "ts_no_update")
os.makedirs(outdir, exist_ok=True)

plt.rc('text', usetex=True)
plt.rc('font', family='arial')
plt.rcParams.update({'font.size': 20})

# 1) Define a toy function on [0,2] with two broad peaks.
# ----------------------------------------------------
def f(x):
    peak1 = 0.8 * np.exp(-((x - 0.5)**2)/(2*0.04))
    peak2 = 1.2 * np.exp(-((x - 1.5)**2)/(2*0.04))
    return peak1 + peak2

X_grid = np.linspace(0, 2, 500).reshape(-1, 1)
y_true = f(X_grid)
ymin, ymax = y_true.min() - 0.1, y_true.max() + 1.0

# 2) Initial data & GP
# ----------------------------------------------------
X_train = np.array([[0.5],[1.0]])
y_train = f(X_train)

# Use a fixed kernel so that the GP does not optimize hyperparameters.
kernel = ConstantKernel(1.0) * RBF(length_scale=0.2)
gp_init = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, optimizer=None)
gp_init.fit(X_train, y_train)

# We'll use this same gp_init for both TS draws.

# 3) Step 1: Draw the first TS sample from gp_init
# ----------------------------------------------------
sample1 = gp_init.sample_y(X_grid, n_samples=1, random_state=10).ravel()
idx1 = np.argmax(sample1)
x1 = X_grid[idx1]
# For the final real measurement, we want f(x1).
real_y1 = f(x1)

# For "fantasy" in step 1, we use the GP's mean at x1 (just for the notional "picked" value).
mean_init, std_init = gp_init.predict(X_grid, return_std=True)
y1_fantasy = mean_init[idx1]

# 4) Step 2: Another TS sample from the same gp_init (no update)
# ----------------------------------------------------
sample2 = gp_init.sample_y(X_grid, n_samples=1, random_state=20).ravel()
idx2 = np.argmax(sample2)
x2 = X_grid[idx2]
real_y2 = f(x2)
y2_fantasy = mean_init[idx2]  # again, the same GP's mean

# 5) Final GP: incorporate the two *real* measurements at x1, x2
# ----------------------------------------------------
X_final = np.vstack([X_train, x1, x2])
y_final = np.append(y_train, [real_y1, real_y2])
gp_final = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, optimizer=None)
gp_final.fit(X_final, y_final)
mean_final, std_final = gp_final.predict(X_grid, return_std=True)
sample3 = gp_final.sample_y(X_grid, n_samples=1, random_state=30).ravel()
idx3 = np.argmax(sample3)
x3 = X_grid[idx3]  # Not used further, just for reference if needed.

# 6) Plot Step 1: Two-row figure
#    - top row: GP posterior (blue) & TS sample1
#    - bottom row: just TS sample1 (or any separate view you want)
# ----------------------------------------------------
fig1, (ax1_top, ax1_bot) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)

# Top row: Posterior + first TS
ax1_top.plot(X_grid, y_true, 'r--', label=r'True function')
ax1_top.scatter(X_train, y_train, color='black', zorder=10, label=r'Prior obs')
ax1_top.plot(X_grid, mean_init, 'b-', label=r'$\mu_t(x)$')
ax1_top.fill_between(X_grid.ravel(), mean_init - 1.5*std_init, mean_init + 1.5*std_init,
                     color='blue', alpha=0.2, label=r'$\mu_t(x)\pm 1.5\sigma_t(x)$')
ax1_top.plot(X_grid, sample1, 'c-', label=r'TS Sample')
ax1_top.axvline(x1, color='blue', linestyle='--', label=r'$x_{t,1}$')
ax1_top.scatter(x1, y1_fantasy, color='blue', s=80, zorder=10)
ax1_top.set_title(r'\textbf{Picking $x_{t,1}$}')
ax1_top.set_ylabel(r'$f(x)$')
# ax1_top.legend(fontsize=8)
ax1_top.set_ylim(ymin, ymax+0.3)

# Bottom row: Just the first TS sample (optional view)
ax1_bot.plot(X_grid, sample1, 'c-', label=r'TS Sample $\tilde{f}_{\mathrm{TS},1}$')
ax1_bot.axvline(x1, color='blue', linestyle='--', label=r'$x_{t,1}$')
ax1_bot.set_title(r'\textbf{TS Acquisition}')
ax1_bot.set_xlabel(r'$x$')
ax1_bot.set_ylabel(r'Acq')
ax1_bot.legend(fontsize=8)
# Different y-limits if desired
sample1_min, sample1_max = sample1.min(), sample1.max()
ax1_bot.set_ylim(sample1_min - 0.1, sample1_max + 0.1)

plt.tight_layout()
fig1.savefig(os.path.join(outdir, "TS_no_update_Step1.pdf"))
plt.close(fig1)

# 7) Plot Step 2: Another two-row figure
#    - top row: the *same* initial posterior (blue) & second TS sample2
#    - bottom row: just TS sample2
# ----------------------------------------------------
fig2, (ax2_top, ax2_bot) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)

# Top row: Posterior + second TS (still from gp_init)
ax2_top.plot(X_grid, y_true, 'r--', label=r'True function')
ax2_top.scatter(X_train, y_train, color='black', zorder=10, label=r'Prior obs')
# same mean_init, std_init as step 1
ax2_top.plot(X_grid, mean_init, 'b-', label=r'$\mu_t(x)$')
ax2_top.fill_between(X_grid.ravel(), mean_init - 1.5*std_init, mean_init + 1.5*std_init,
                     color='blue', alpha=0.2, label=r'$\mu_t(x)\pm 1.5\sigma_t(x)$')
ax2_top.plot(X_grid, sample2, 'm-', label=r'TS Sample')
ax2_top.axvline(x2, color='green', linestyle='--', label=r'$x_{t,2}$')
ax2_top.scatter(x2, y2_fantasy, color='magenta', s=80, zorder=10)
ax2_top.set_title(r'\textbf{Picking $x_{t,2}$}')
ax2_top.set_ylabel(r'$f(x)$')
# ax2_top.legend(fontsize=8)
ax2_top.set_ylim(ymin, ymax+0.3)

# Bottom row: Just the second TS sample
ax2_bot.plot(X_grid, sample2, 'm-', label=r'TS Sample $\tilde{f}_{\mathrm{TS},2}$')
ax2_bot.axvline(x2, color='green', linestyle='--', label=r'$x_{t,2}$')
ax2_bot.set_title(r'\textbf{TS Acquisition}')
ax2_bot.set_xlabel(r'$x$')
# ax2_bot.set_ylabel(r'')
ax2_bot.legend(fontsize=8)
sample2_min, sample2_max = sample2.min(), sample2.max()
ax2_bot.set_ylim(sample2_min - 0.1, sample2_max + 0.1)

plt.tight_layout()
fig2.savefig(os.path.join(outdir, "TS_no_update_Step2.pdf"))
plt.close(fig2)

# 8) Final plot: after both real measurements
# ----------------------------------------------------
fig3, ax3 = plt.subplots(1, 1, figsize=(6, 3))
ax3.plot(X_grid, y_true, 'r--', label=r'True function')
ax3.scatter(X_final, y_final, color='black', zorder=10, label=r'Prior obs')
ax3.plot(X_grid, mean_final, color='maroon', label=r'$\mu_{\mathrm{final}}(x)$')
ax3.fill_between(X_grid.ravel(), mean_final - 1.5*std_final, mean_final + 1.5*std_final,
                 color='maroon', alpha=0.2, label=r'$\mu_{t+1}(x)\pm 1.5\sigma_{t+1}(x)$')
# ax3.plot(X_grid, sample3, 'k-', label=r'Final TS Sample')

ax3.scatter(x1, real_y1, color='maroon', s=100, zorder=10)
# ax3.axvline(x2, color='maroon', linestyle='--', label=r'$x_{t,2}$')
ax3.scatter(x2, real_y2, color='maroon', s=100, zorder=10)
ax3.set_title(r'\textbf{GP after round $t$}')
ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$f(x)$')
ax3.set_ylim(ymin, ymax+0.3)
# ax3.legend(fontsize=8)

plt.tight_layout()
fig3.savefig(os.path.join(outdir, "TS_no_update_Final.pdf"))
plt.close(fig3)

# Print the chosen points
print("Two-step Thompson Sampling (no update between steps) selected points:")
print("Step 1 (x1):", x1.ravel())
print("Step 2 (x2):", x2.ravel())
