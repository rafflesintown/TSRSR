import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# Create the output directory if it doesn't exist.
output_dir = os.path.join("visualization", "bucb")
os.makedirs(output_dir, exist_ok=True)

# Use LaTeX rendering in matplotlib for nicer labels.
plt.rc('text', usetex=True)
plt.rc('font', family='arial')
plt.rcParams.update({'font.size': 20})

# ----------------------------
# Define toy function on [0,2]
# ----------------------------
def f(x):
    peak1 = 0.8 * np.exp(-((x - 0.5)**2)/(2*0.04))
    peak2 = 1.2 * np.exp(-((x - 1.5)**2)/(2*0.04))
    return peak1 + peak2 

# ----------------------------
# Create grid and set y-axis limits
# ----------------------------
X_grid = np.linspace(0, 2, 500).reshape(-1, 1)
y_true = f(X_grid)
ymin, ymax = y_true.min() - 0.1, y_true.max() + 1.0

# ----------------------------
# Initial training data.
# ----------------------------
X_train = np.array([[0.5],[1.0]])
y_train = f(X_train)

# ----------------------------
# Define GP model (BUCB) with fixed hyperparameters.
# ----------------------------
kernel = ConstantKernel(1.0, (0.1, 10.0)) * RBF(length_scale=0.2, length_scale_bounds="fixed")
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, optimizer=None, n_restarts_optimizer=5)
gp.fit(X_train, y_train)
gp.fit(X_train, y_train)

# BUCB uses UCB = GP mean + beta * std.
beta = 1.5

# ----------------------------
# Step 1: Compute initial BUCB region and select the first point.
# ----------------------------
mean1, std1 = gp.predict(X_grid, return_std=True)
ucb1 = mean1 + beta * std1
idx1 = np.argmax(ucb1)              # maximize UCB
x1 = X_grid[idx1]
y1_fantasy = mean1[idx1]             # Fantasy observation: use GP mean at x1

# ----------------------------
# Step 2: Update the GP with the first fantasy observation.
# ----------------------------
X_train_2 = np.vstack([X_train, x1.reshape(1, -1)])
y_train_2 = np.append(y_train, y1_fantasy)
gp2 = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=5)
gp2.fit(X_train_2, y_train_2)

mean2, std2 = gp2.predict(X_grid, return_std=True)
ucb2 = mean2 + beta * std2
idx2 = np.argmax(ucb2)
x2 = X_grid[idx2]
y2_fantasy = mean2[idx2]

# ----------------------------
# Final: Update the GP with the second fantasy observation,
# but for the final posterior, we use the real measurements.
# ----------------------------
real_y1 = f(x1)
real_y2 = f(x2)
X_train_final = np.vstack([X_train, x1.reshape(1, -1), x2.reshape(1, -1)])
y_train_final = np.append(y_train, [real_y1, real_y2])
gp_final = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=5)
gp_final.fit(X_train_final, y_train_final)

mean_final, std_final = gp_final.predict(X_grid, return_std=True)
ucb_final = mean_final + beta * std_final  # computed for selection but not used for plotting here

# ----------------------------------
# Also draw TS samples for plotting.
# ----------------------------------
sample1 = gp.sample_y(X_grid, n_samples=1, random_state=0).ravel()
f_star_1 = sample1.max()

sample2 = gp2.sample_y(X_grid, n_samples=1, random_state=0).ravel()
f_star_2 = sample2.max()

sample_final = gp_final.sample_y(X_grid, n_samples=1, random_state=0).ravel()
f_star_final = sample_final.max()

# ----------------------------------
# Figure 1: Save plot for Step 1 (Posterior and Acquisition)
# ----------------------------------
fig1, (ax1_top, ax1_bot) = plt.subplots(2, 1, figsize=(8, 4), sharex=True, sharey=True)

# Top: GP Posterior at Step 1.
ax1_top.plot(X_grid, y_true, r'r--', label=r'\textbf{True function}')
ax1_top.scatter(X_train, y_train, c='black', zorder=10, label=r'\textbf{Prior obs}')
ax1_top.plot(X_grid, mean1, r'b-', label=r'$\mu_t(x)$')
ax1_top.fill_between(X_grid.ravel(), mean1 - beta*std1, mean1 + beta*std1,
                     color='blue', alpha=0.2, label=r'$\mu_t(x)\pm 1.5\sigma_t(x)$')
ax1_top.axvline(x1, color='blue', linestyle='--', label=r'$x_{t,1}$')
ax1_top.scatter(x1, y1_fantasy, color='blue', s=100, zorder=10)
# ax1_top.axhline(f_star_1, color='gray', linestyle='--', label=r'$f^*_{\mathrm{TS}}$')
ax1_top.set_title(r'\textbf{Picking $x_{t,1}$}')
ax1_top.set_ylabel(r'$f(x)$')
plt.subplots_adjust(right=0.8)
ax1_top.legend(fontsize=10, loc = 'upper right',bbox_to_anchor=(1.5, 1.2))

# Bottom: Acquisition (UCB) at Step 1.
ax1_bot.plot(X_grid, ucb1, r'k-', label=r'\textbf{Acquisition: }$\mu_t(x)+1.5\sigma_t(x)$')
ax1_bot.axvline(x1, color='blue', linestyle='--', label=r'$x_{t,1}$')
ax1_bot.set_title(r'\textbf{BUCB Acquisition}')
ax1_bot.set_xlabel(r'$x$')
ax1_bot.set_ylabel(r'Acq')
# ax1_bot.legend(fontsize=8,loc = 'upper right')

for ax in (ax1_top, ax1_bot):
    ax.set_ylim(ymin, ymax)
    
plt.tight_layout()
fig1.savefig(os.path.join(output_dir, "BUCB_Step1.pdf"))
plt.close(fig1)

# ----------------------------------
# Figure 2: Save plot for Step 2 (Posterior and Acquisition)
# ----------------------------------
fig2, (ax2_top, ax2_bot) = plt.subplots(2, 1, figsize=(6, 4), sharex=True, sharey=True)

# Top: GP Posterior at Step 2.
ax2_top.plot(X_grid, y_true, r'r--', label=r'\textbf{True function}')
ax2_top.scatter(X_train_2, y_train_2, c='black', zorder=10, label=r'\textbf{Prior obs}')
# ax2_top.plot(X_grid, mean2, r'b-', label=r'$\mu_t(x)$')
ax2_top.plot(X_grid, mean2, r'b-')
ax2_top.fill_between(X_grid.ravel(), mean2 - beta*std2, mean2 + beta*std2,
                     color='green', alpha=0.2, label=r'$\mu_t(x)\pm 1.5\sigma_t(x\,|\,x_{t,1})$')
ax2_top.axvline(x2, color='green', linestyle='--', label=r'$x_{t,2}$')
ax2_top.scatter(x1, y1_fantasy, color='blue', s=100, zorder=10,label=r'$x_{t,1}$')
ax2_top.scatter(x2, y2_fantasy, color='green', s=100, zorder=10)
# ax2_top.axhline(f_star_2, color='gray', linestyle='--', label=r'$f^*_{\mathrm{TS}}$')
ax2_top.set_title(r'\textbf{Picking $x_{t,2}$}')
ax2_top.set_ylabel(r'$f(x)$')
plt.subplots_adjust(right=0.8)
ax2_top.legend(fontsize=10, loc = 'upper right',bbox_to_anchor=(1.5, 1.2))
# ax2_top.legend(fontsize=8, loc = 'upper left')

# Bottom: Acquisition (UCB) at Step 2.
ax2_bot.plot(X_grid, ucb2, r'k-', label=r'\textbf{Acquisition: }$\mu_t(x)+1.5\sigma(x | x_{t,1})$')
ax2_bot.axvline(x2, color='green', linestyle='--', label=r'$x_{t,2}$')
ax2_bot.set_title(r'\textbf{BUCB Acquisition}')
ax2_bot.set_xlabel(r'$x$')
# ax2_bot.legend(fontsize=8,loc = 'upper right')

for ax in (ax2_top, ax2_bot):
    ax.set_ylim(ymin, ymax)
    
plt.tight_layout()
fig2.savefig(os.path.join(output_dir, "BUCB_Step2.pdf"))
plt.close(fig2)

# ----------------------------------
# Figure 3: Save final GP Posterior (Step 3) using real measurements.
# ----------------------------------
fig3, ax3 = plt.subplots(1, 1, figsize=(6, 3))
ax3.plot(X_grid, y_true, r'r--', label=r'\textbf{True function}')
ax3.scatter(X_train_final, y_train_final, c='black', zorder=10, label=r'\textbf{Prior obs}')
ax3.plot(X_grid, mean_final, color='maroon', label=r'$\mu_{t+1}(x)$')
ax3.fill_between(X_grid.ravel(), mean_final - beta*std_final, mean_final + beta*std_final,
                 color='maroon', alpha=0.2, label=r'$\mu_{t+1}(x)\pm 1.5\sigma_{t+1}(x)$')
# ax3.plot(X_grid, ucb_final, r'k-', label=r'\textbf{Acquisition: }$\mu(x)+1.5\sigma(x)$')
# Mark both newly selected points.
# ax3.axvline(x1, color='maroon', linestyle='--', label=r'$x_{t,1}$')
ax3.scatter(x1, real_y1, color='maroon', s=100, zorder=10, label=r'\textbf{Round $t$ sampled pts}')
# ax3.axvline(x2, color='maroon', linestyle='--', label=r'$x_{t,2}$')
ax3.scatter(x2, real_y2, color='maroon', s=100, zorder=10)
ax3.set_title(r'\textbf{GP after round $t$ ($m=2$)}')
ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$f(x)$')
ax3.set_ylim(ymin, ymax)
# Place the legend below the figure.
# ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=2, fontsize=8)

plt.tight_layout()
fig3.savefig(os.path.join(output_dir, "BUCB_Final.pdf"))
plt.close(fig3)

print(r"BUCB selected points:")
print(r"Step 1: $x_1=$", x1.ravel())
print(r"Step 2: $x_2=$", x2.ravel())
