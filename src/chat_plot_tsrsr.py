import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import os



# Create output directory for TS-RSR figures.
output_dir = os.path.join("visualization", "tsrsr")
os.makedirs(output_dir, exist_ok=True)

# Use LaTeX for nicer labels.
plt.rc('text', usetex=True)
plt.rc('font', family='arial')
plt.rcParams.update({'font.size': 20})



# Define our test function on [0,2] (here with two broad peaks).
def f(x):
    # peak1 = 0.5 * np.exp(-((x - 0.5)**2) / (2 * 0.04))
    # peak2 = 0.7 * np.exp(-((x - 1.5)**2) / (2 * 0.04))
    peak1 = 0.8 * np.exp(-((x - 0.5)**2) / (2 * 0.04))
    peak2 = 1.2 * np.exp(-((x - 1.5)**2) / (2 * 0.04))
    return peak1 + peak2

# Create a grid for plotting.
X_grid = np.linspace(0, 2, 500).reshape(-1, 1)
y_true = f(X_grid)
ymin, ymax = y_true.min() - 0.1, y_true.max() + 1.0

# Define the TS-RSR acquisition function.
# Here, we define it as: a(x) = (f* - mu(x)) / (std(x) + epsilon),
# where f* is computed by drawing 5 samples from the GP posterior and taking their maximum.
def ts_rsr_acquisition(gp_model, X_candidates, n_samples=5, eps=1e-8):
    y_samples = gp_model.sample_y(X_candidates, n_samples=n_samples, random_state=0)
    f_star = y_samples.max()   # Maximum over all samples and candidates
    mu, std = gp_model.predict(X_candidates, return_std=True)
    acq = (f_star - mu) / (std + eps)
    return acq

# ---------------------------
# Initial data and GP setup
# ---------------------------
# Start with two initial points.
X_train = np.array([[0.5], [1.0]])
y_train = f(X_train)

# We'll fix the kernel parameters (for reproducibility).
kernel = ConstantKernel(1.0, (0.1, 10.0)) * RBF(length_scale=0.2, length_scale_bounds="fixed")
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, optimizer=None, n_restarts_optimizer=5)
gp.fit(X_train, y_train)

# We'll use beta only for visualizing the uncertainty bands.
beta = 1.5

# ----------------------------------
# Step 1: TS-RSR selection using the acquisition function.
# ----------------------------------
acq1 = ts_rsr_acquisition(gp, X_grid)
idx1 = np.argmin(acq1)            # Select x1 based on max acq value.
x1 = X_grid[idx1]
mean1, std1 = gp.predict(X_grid, return_std=True)
# For TS display, we also draw one TS sample.
sample1 = gp.sample_y(X_grid, n_samples=1, random_state=10).ravel()
f_star_1 = sample1.max()

# For TS-RSR selection, we use the fantasy value (GP mean) at x1.
y1_fantasy = mean1[idx1]
# For final GP, we will later use the real measurement at x1.
real_y1 = f(x1)

# Update training data for Step 2 (selection still uses fantasy values).
X_train_2 = np.vstack([X_train, x1.reshape(1, -1)])
y_train_2 = np.append(y_train, y1_fantasy)

# ----------------------------------
# Step 2: TS-RSR selection using the acquisition function.
# ----------------------------------
gp2 = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, optimizer=None, n_restarts_optimizer=5)
gp2.fit(X_train_2, y_train_2)
acq2 = ts_rsr_acquisition(gp2, X_grid)
idx2 = np.argmin(acq2)
x2 = X_grid[idx2]
mean2, std2 = gp2.predict(X_grid, return_std=True)
sample2 = gp2.sample_y(X_grid, n_samples=1, random_state=20).ravel()
f_star_2 = sample2.max()
y2_fantasy = mean2[idx2]
real_y2 = f(x2)  # For final GP, use the real measurement.

# Update training data for final GP.
X_train_temp = np.vstack([X_train_2, x2.reshape(1, -1)])
y_train_temp = np.append(y_train_2, y2_fantasy)

# ----------------------------------
# Final: Refit GP using real measurements at x1 and x2.
# ----------------------------------
X_train_final = np.vstack([X_train, x1.reshape(1, -1), x2.reshape(1, -1)])
y_train_final = np.append(y_train, [real_y1, real_y2])
gp_final = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, optimizer=None, n_restarts_optimizer=5)
gp_final.fit(X_train_final, y_train_final)
mean_final, std_final = gp_final.predict(X_grid, return_std=True)
sample3 = gp_final.sample_y(X_grid, n_samples=1, random_state=0).ravel()
f_star_3 = sample3.max()
acq_final = ts_rsr_acquisition(gp_final, X_grid)




# ----------------------------
# Figure 1: Save plot for Step 1 (Posterior and Acquisition)
# ----------------------------
fig1, (ax1_top, ax1_bot) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

# Top: GP Posterior at Step 1.
ax1_top.plot(X_grid, y_true, r'r--', label=r'\textbf{True function}')
ax1_top.scatter(X_train, y_train, c='black', zorder=10, label=r'\textbf{Prior obs}')
ax1_top.plot(X_grid, mean1, r'b-', label=r'$\mu_t(x)$')
ax1_top.fill_between(X_grid.ravel(), mean1 - beta*std1, mean1 + beta*std1,
                     color='blue', alpha=0.2, label=r'$\mu_t(x)\pm 1.5\sigma_t(x)$')
# ax1_top.plot(X_grid, sample1, r'c-', label=r'\textbf{TS Sample $\tilde{f}_{\mathrm{TS},1}$}')
ax1_top.plot(X_grid, sample1, r'c-', label=r'\textbf{TS Sample $\tilde{f}_{t}$}')
ax1_top.plot(X_grid, sample1, r'c-')
# ax1_top.axvline(x1, color='blue', linestyle='--', label=r'$x_{t,1}$')
ax1_top.axvline(x1, color='blue', linestyle='--', label=r'$x_{t}$')
ax1_top.scatter(x1, y1_fantasy, color='blue', s=100, zorder=10)
# ax1_top.axhline(f_star_1, color='gray', linestyle='--', label=r'$\tilde{f}^*_{\mathrm{TS},1}$')
ax1_top.axhline(f_star_1, color='gray', linestyle='--')
ax1_top.set_title(r'\textbf{Picking $x_{t,1}$}')
ax1_top.set_ylabel(r'$f(x)$')
# ax1_top.legend(fontsize=8, loc='upper center')
ax1_top.legend(loc='center right', bbox_to_anchor=(1.6, 0.5), fontsize=10)

# # Assume 'ax' is an existing axis with a legend.
# handles, labels = ax1_top.get_legend_handles_labels()

# # Create a new figure for the legend.
# figLegend = plt.figure(figsize=(3, 3))
# figLegend.legend(handles, labels, loc='center', fontsize=12)
# plt.axis('off')  # Optionally turn off the axis for a clean legend.
# plt.tight_layout()
# figLegend.savefig('legend_ax1_top.pdf')



# Bottom: Acquisition (TS-RSR) at Step 1.
ax1_bot.plot(X_grid, acq1, r'k-', label=r'$Acq(x)=\frac{\tilde{f}^*_{\mathrm{TS},1}-\mu_t(x)}{\sigma_t(x)}$')
ax1_bot.axvline(x1, color='blue', linestyle='--', label=r'$x_{t,1}$')
ax1_bot.set_title(r'\textbf{TS-RSR Acquisition Function}')
ax1_bot.set_xlabel(r'$x$')
ax1_bot.set_ylabel(r'Acq')
plt.tight_layout()
ax1_bot.legend(loc='center right', bbox_to_anchor=(1.7, 0.5), fontsize=10)

# # Assume 'ax' is an existing axis with a legend.
# handles, labels = ax1_bot.get_legend_handles_labels()

# # Create a new figure for the legend.
# figLegend = plt.figure(figsize=(3, 3))
# figLegend.legend(handles, labels, loc='center', fontsize=12)
# plt.axis('off')  # Optionally turn off the axis for a clean legend.
# plt.tight_layout()
# figLegend.savefig('legend_ax1_bot.pdf')


ax1_top.set_ylim(ymin, ymax)
ax1_bot.set_ylim(ymin, ymax+3)

    
fig1.savefig(os.path.join(output_dir, "TSRSR_Step1.pdf"))
plt.close(fig1)

# ----------------------------
# Figure 2: Save plot for Step 2 (Posterior and Acquisition)
# ----------------------------
fig2, (ax2_top, ax2_bot) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)

# Top: GP Posterior at Step 2.
# ax2_top.plot(X_grid, y_true, r'r--', label=r'\textbf{True function}')
# ax2_top.scatter(X_train_2, y_train_2, c='black', zorder=10, label=r'\textbf{Prior obs}')
ax2_top.plot(X_grid, y_true, r'r--')
ax2_top.scatter(X_train_2, y_train_2, c='black', zorder=10)
# ax2_top.plot(X_grid, mean2, r'b-', label=r'$\mu_t(x)$')
ax2_top.plot(X_grid, mean2, r'b-', label=r'$\mu_t(x)$')
ax2_top.fill_between(X_grid.ravel(), mean2 - beta*std2, mean2 + beta*std2,
                     color='green', alpha=0.2, label=r'$\mu_t(x)\pm 1.5\sigma_t(x\,|\,x_{t,1})$')
ax2_top.plot(X_grid, sample2, r'm-', label=r'\textbf{TS Sample $\tilde{f}_{\mathrm{TS},2}$}')
ax2_top.plot(X_grid, sample2, r'm-')
ax2_top.axvline(x2, color='green', linestyle='--', label=r'$x_{t,2}$')
ax2_top.scatter(x1, y1_fantasy, color='blue', s=100, zorder=10, label = r'$x_{t,1}$')
ax2_top.scatter(x2, y2_fantasy, color='green', s=100, zorder=10)
ax2_top.axhline(f_star_2, color='gray', linestyle='--', label=r'$\tilde{f}^*_{\mathrm{TS},2}$')
ax2_top.axhline(f_star_2, color='gray', linestyle='--')
ax2_top.set_title(r'\textbf{Picking $x_{t,2}$')
ax2_top.set_ylabel(r'$f(x)$')
# ax2_top.legend(fontsize=8, loc='upper center')
# ax2_top.legend(loc='center right', bbox_to_anchor=(1.7, 0.5), fontsize=10)

# # Assume 'ax' is an existing axis with a legend.
# handles, labels = ax2_top.get_legend_handles_labels()

# # Create a new figure for the legend.
# figLegend = plt.figure(figsize=(3, 3))
# figLegend.legend(handles, labels, loc='center', fontsize=12)
# plt.axis('off')  # Optionally turn off the axis for a clean legend.
# plt.tight_layout()
# figLegend.savefig('legend_ax2_top.pdf')

# Bottom: Acquisition (TS-RSR) at Step 2.
ax2_bot.plot(X_grid, acq2, r'k-', label=r'$Acq(x)=\frac{\tilde{f}^*_{\mathrm{TS},2}-\mu_t(x)}{\sigma_t(x \mid x_{t,1})}$')
ax2_bot.axvline(x2, color='green', linestyle='--', label=r'$x_{t,2}$')
ax2_bot.set_title(r'\textbf{TS-RSR Acquisition Function}')
ax2_bot.set_xlabel(r'$x$')
ax2_bot.set_ylabel(r'Acq')
    
plt.tight_layout()
# ax2_bot.legend(loc='center right', bbox_to_anchor=(1.7, 0.5), fontsize=10)

ax2_top.set_ylim(ymin, ymax)
ax2_bot.set_ylim(ymin, ymax+3)

# # Assume 'ax' is an existing axis with a legend.
# handles, labels = ax2_bot.get_legend_handles_labels()

# # Create a new figure for the legend.
# figLegend = plt.figure(figsize=(3, 3))
# figLegend.legend(handles, labels, loc='center', fontsize=12)
# plt.axis('off')  # Optionally turn off the axis for a clean legend.
# plt.tight_layout()
# figLegend.savefig('legend_ax2_bot.pdf')

fig2.savefig(os.path.join(output_dir, "TSRSR_Step2.pdf"))
plt.close(fig2)

# ----------------------------
# Figure 3: Save final GP Posterior (Step 3) using real measurements.
# (Final GP is trained using real measurements at x1 and x2.)
# ----------------------------
# Here we want the final posterior plot to have the same size as the top plot in Figure 2.
fig3, ax3 = plt.subplots(1, 1, figsize=(8, 4))
ax3.plot(X_grid, y_true, r'r--', label=r'\textbf{True function}')
ax3.scatter(X_train_final, y_train_final, c='black', zorder=10, label=r'\textbf{Prior obs}')
ax3.plot(X_grid, mean_final, color='maroon', label=r'$\mu_{t+1}(x)$')
ax3.fill_between(X_grid.ravel(), mean_final - beta*std_final, mean_final + beta*std_final,
                 color='maroon', alpha=0.2, label=r'$\mu_{t+1}(x)\pm 1.5\sigma_{t+1}(x)$')
# ax3.plot(X_grid, acq_final, r'k-', label=r'\textbf{Acquisition: }$\mu(x)+1.5\sigma(x)$')
# Mark both selected points.
# ax3.axvline(x1, color='maroon', linestyle='--', label=r'$x_{t,1}$')
# ax3.scatter(x1, real_y1, color='maroon', s=100, zorder=10,label=r'Round $t$ sampled pts')
ax3.scatter(x1, real_y1, color='maroon', s=100, zorder=10,label=r'Round $t$ sampled pt')
# ax3.axvline(x2, color='maroon', linestyle='--', label=r'$x_{t,2}$')
ax3.scatter(x2, real_y2, color='maroon', s=100, zorder=10)
ax3.set_title(r'\textbf{GP after round $t$ ($m=2$)}')
ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$f(x)$')
ax3.set_ylim(ymin, ymax)
# Place the legend below the figure.
plt.subplots_adjust(right=0.7)
ax3.legend(loc='center', bbox_to_anchor=(1.2, 0.5), fontsize=10)

# # Assume 'ax' is an existing axis with a legend.
# handles, labels = ax3.get_legend_handles_labels()

# # Create a new figure for the legend.
# figLegend = plt.figure(figsize=(3, 3))
# figLegend.legend(handles, labels, loc='center', fontsize=12)
# plt.axis('off')  # Optionally turn off the axis for a clean legend.
# plt.tight_layout()
# figLegend.savefig('legend_ax3.pdf')

fig3.savefig(os.path.join(output_dir, "TSRSR_Final.pdf"))
plt.close(fig3)

print(r"TS-RSR selected points (final GP uses real measurements):")
print(r"Step 1: $x_1=$", x1.ravel())
print(r"Step 2: $x_2=$", x2.ravel())




# # ----------------------------------
# # Figure 1: Two-row, two-column plots for Steps 1 & 2.
# # Top row: GP posterior & TS sample.
# # Bottom row: TS-RSR acquisition function.
# # ----------------------------------
# fig1, axs1 = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

# # Step 1 Posterior (Top Left)
# axs1[0, 0].plot(X_grid, y_true, 'r--', label='True function')
# axs1[0, 0].scatter(X_train, y_train, c='black', zorder=10, label='Initial Observations')
# axs1[0, 0].plot(X_grid, mean1, 'b-', label='GP Mean')
# axs1[0, 0].fill_between(X_grid.ravel(), mean1 - beta*std1, mean1 + beta*std1,
#                          color='blue', alpha=0.2, label='Uncertainty Band')
# axs1[0, 0].plot(X_grid, sample1, 'c-', label='TS Sample')
# axs1[0, 0].axvline(x1, color='blue', linestyle='--', label='Selected Point')
# axs1[0, 0].scatter(x1, y1_fantasy, color='blue', s=100, zorder=10)
# axs1[0, 0].axhline(f_star_1, color='gray', linestyle='--', label='TS sample max')
# axs1[0, 0].set_title("Step 1: Posterior")
# axs1[0, 0].set_ylabel("f(x)")

# # Step 1 Acquisition (Bottom Left)
# axs1[1, 0].plot(X_grid, acq1, 'k-', label='TS-RSR Acq')
# axs1[1, 0].axvline(x1, color='blue', linestyle='--', label='Selected Point')
# axs1[1, 0].set_title("Step 1: Acquisition")
# axs1[1, 0].set_xlabel("x")
# axs1[1, 0].set_ylabel("Acq")
# axs1[1, 0].legend(fontsize=8)

# # Step 2 Posterior (Top Right)
# axs1[0, 1].plot(X_grid, y_true, 'r--', label='True function')
# axs1[0, 1].scatter(X_train_2, y_train_2, c='black', zorder=10, label='Obs + Fantasy')
# axs1[0, 1].plot(X_grid, mean2, 'g-', label='Updated GP Mean')
# axs1[0, 1].fill_between(X_grid.ravel(), mean2 - beta*std2, mean2 + beta*std2,
#                          color='green', alpha=0.2, label='Uncertainty Band')
# axs1[0, 1].plot(X_grid, sample2, 'm-', label='TS Sample')
# axs1[0, 1].axvline(x2, color='green', linestyle='--', label='Selected Point')
# axs1[0, 1].scatter(x2, y2_fantasy, color='green', s=100, zorder=10)
# axs1[0, 1].axhline(f_star_2, color='gray', linestyle='--', label='TS sample max')
# axs1[0, 1].set_title("Step 2: Posterior")
# # Step 2 Acquisition (Bottom Right)
# axs1[1, 1].plot(X_grid, acq2, 'k-', label='TS-RSR Acq')
# axs1[1, 1].axvline(x2, color='green', linestyle='--', label='Selected Point')
# axs1[1, 1].set_title("Step 2: Acquisition")
# axs1[1, 1].set_xlabel("x")
# axs1[1, 1].legend(fontsize=8)

# for ax in axs1.flat:
#     ax.set_ylim(ymin, ymax)

# plt.tight_layout()
# plt.show()

# # ----------------------------------
# # Figure 2: Final GP Posterior (Step 3)
# # (Final GP is trained using real measurements at x1 and x2.)
# # ----------------------------------
# fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))

# ax2.plot(X_grid, y_true, 'r--', label='True function')
# ax2.scatter(X_train_final, y_train_final, c='black', zorder=10, label='Obs + Real Measurements')
# ax2.plot(X_grid, mean_final, color='magenta', label='Final GP Mean')
# ax2.fill_between(X_grid.ravel(), mean_final - beta*std_final, mean_final + beta*std_final,
#                  color='magenta', alpha=0.2, label='Uncertainty Band')
# ax2.plot(X_grid, sample3, 'k-', label='TS Sample')
# ax2.axvline(x2, color='magenta', linestyle='--', label='Last Selected Point')
# ax2.scatter(x2, real_y2, color='magenta', s=100, zorder=10)
# ax2.axhline(f_star_3, color='gray', linestyle='--', label='TS sample max')
# ax2.set_title("Final Posterior (after 2 measurements)")
# ax2.set_xlabel("x")
# ax2.set_ylabel("f(x)")
# ax2.set_ylim(ymin, ymax)
# ax2.legend(fontsize=8)

# plt.tight_layout()
# plt.show()

# print("TS-RSR selected points (final GP uses real measurements):")
# print("Step 1:", x1.ravel())
# print("Step 2:", x2.ravel())
