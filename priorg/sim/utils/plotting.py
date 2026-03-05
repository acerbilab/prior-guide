import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import multivariate_normal, norm


def plot_prior_visualization(
    all_samples,
    all_samples_with_prior,
    all_unobserved,
    mean_prior,
    Sigma_prior,
    num_samples,
    true_sampled_data=None,
    i=0,
    prefix="",
    title="Predictive visualization",
):
    """
    Visualizes simulated data and parameter samples for Simformer and PriorGuide,
    along with the bivariate Gaussian prior and its marginal densities.

    Adds dotted lines at +/- 3 * sigma for each marginal distribution on the
    parameter space subplot.

    Parameters:
      all_samples: np.array
         Array of Simformer samples (parameters and data).
      all_samples_with_prior: np.array
         Array of PriorGuide samples (parameters and data).
      all_unobserved: np.array
         Array containing the true (unobserved) parameter and data.
      mean_prior: np.array
         Mean of the prior (at least 2D; first two entries correspond to parameters).
      Sigma_prior: np.array
         Covariance matrix of the prior (at least 2x2).
      num_samples: int
         Number of sample trajectories to plot.
      i: int, optional
         Index for selecting the sample (default is 0).
    """
    # Create a figure with 2 rows and 2 columns of subplots.
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axs.flatten()

    # --- Left subplot: Simformer Data ---
    for j in range(num_samples):
        if j == 0:
            ax1.plot(
                all_samples[i, 2:, j],
                alpha=0.3,
                color="orange",
                label="Simformer data sample",
            )
        else:
            ax1.plot(all_samples[i, 2:, j], alpha=0.3, color="orange")
        if true_sampled_data is not None:
            if j == 0:
                ax1.plot(
                    true_sampled_data[j],
                    color="grey",
                    label="~ P(x|theta)",
                    alpha=0.3,
                    linestyle="--",
                )
            else:
                ax1.plot(true_sampled_data[j], color="grey", alpha=0.3, linestyle="--")
    ax1.plot(all_unobserved[i, 2:], color="black", label="True data")
    ax1.set_title("Simformer")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_ylim(-4, 15)
    ax1.legend()

    # --- Right subplot: PriorGuide Data ---
    for j in range(num_samples):
        if j == 0:
            ax2.plot(
                all_samples_with_prior[i, 2:, j],
                alpha=0.3,
                color="green",
                label="PriorGuide data sample",
            )
        else:
            ax2.plot(all_samples_with_prior[i, 2:, j], alpha=0.3, color="green")

        if true_sampled_data is not None:
            if j == 0:
                ax2.plot(
                    true_sampled_data[j],
                    color="grey",
                    label="~ P(x|theta)",
                    alpha=0.3,
                    linestyle="--",
                )
            else:
                ax2.plot(true_sampled_data[j], color="grey", alpha=0.3, linestyle="--")

    ax2.plot(all_unobserved[i, 2:], color="black", label="True data")
    ax2.set_title("PriorGuide")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_ylim(-4, 15)
    ax2.legend()

    # --- Third subplot: Simformer Parameter Samples ---
    ax3.scatter(
        all_samples[i, 0, :],
        all_samples[i, 1, :],
        label="Simformer param samples",
        alpha=0.3,
        color="orange",
    )
    ax3.scatter(
        all_unobserved[i, 0],
        all_unobserved[i, 1],
        color="black",
        label="True parameter",
        marker="x",
    )
    ax3.set_xlim(0, 2)
    ax3.set_ylim(-2, 2)
    ax3.set_xlabel("$\\theta_1$")
    ax3.set_ylabel("$\\theta_2$")
    ax3.legend()

    # --- Fourth subplot: PriorGuide Parameter Samples & Prior Visualization ---
    ax4.scatter(
        all_samples_with_prior[i, 0, :],
        all_samples_with_prior[i, 1, :],
        label="PriorGuide param samples",
        alpha=0.3,
        color="green",
    )
    ax4.scatter(
        all_unobserved[i, 0],
        all_unobserved[i, 1],
        color="black",
        label="True parameter",
        marker="x",
    )
    ax4.set_xlim(0, 2)
    ax4.set_ylim(-2, 2)
    ax4.set_xlabel("$\\theta_1$")
    ax4.set_ylabel("$\\theta_2$")

    # --- Bivariate Prior Contour ---
    # Create grid to evaluate the bivariate Gaussian PDF.
    x = np.linspace(0, 2, 500)
    y = np.linspace(-2, 2, 500)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # Evaluate the bivariate Gaussian PDF (using the first two dimensions).
    rv = multivariate_normal(mean_prior[:2], Sigma_prior[:2, :2])
    Z = rv.pdf(pos)

    # Plot the contour with a solid line.
    cs = ax4.contour(X, Y, Z, colors="blue", alpha=0.4)
    # Create a proxy artist so that the contour appears in the legend.
    contour_proxy = Line2D(
        [0], [0], color="blue", linestyle="solid", label="PriorGuide prior"
    )

    # --- Marginal Densities scaled to 10% of the frame ---
    # Get current frame dimensions from ax4.
    x_limits = ax4.get_xlim()  # (0, 2)
    y_limits = ax4.get_ylim()  # (-2, 2)
    frame_width = x_limits[1] - x_limits[0]  # 2
    frame_height = y_limits[1] - y_limits[0]  # 4

    # Marginal for theta1: plot along the bottom.
    dim1rv = norm(loc=mean_prior[0], scale=np.sqrt(Sigma_prior[0, 0]))
    Z1 = dim1rv.pdf(x)
    # Scale so that the maximum is 10% of the frame height (0.4 here).
    Z1_scaled = (Z1 / Z1.max()) * (0.1 * frame_height)
    # Shift so that the baseline is at the bottom (y_limits[0]).
    Z1_shifted = y_limits[0] + Z1_scaled
    (line_theta1,) = ax4.plot(
        x, Z1_shifted, color="red", label="Prior on $\\theta_1$", alpha=0.7
    )

    # Marginal for theta2: plot along the left side.
    dim2rv = norm(loc=mean_prior[1], scale=np.sqrt(Sigma_prior[1, 1]))
    Z2 = dim2rv.pdf(y)
    # Scale so that the maximum is 10% of the frame width (0.2 here).
    Z2_scaled = (Z2 / Z2.max()) * (0.1 * frame_width)
    # Shift so that the baseline is at the left (x_limits[0]).
    X2_shifted = x_limits[0] + Z2_scaled
    (line_theta2,) = ax4.plot(
        X2_shifted, y, color="purple", label="Prior on $\\theta_2$", alpha=0.7
    )

    # --- 3σ Dotted Lines in the Marginals ---
    # For theta1 (vertical lines at mean +/- 3 std)
    std_theta1 = np.sqrt(Sigma_prior[0, 0])
    ax4.axvline(
        mean_prior[0] - 2 * std_theta1,
        color="red",
        linestyle=":",
        alpha=0.8,
        label=r"$\pm 2\sigma\ (\theta_1)$",
    )
    ax4.axvline(mean_prior[0] + 2 * std_theta1, color="red", linestyle=":", alpha=0.8)

    # For theta2 (horizontal lines at mean +/- 3 std)
    std_theta2 = np.sqrt(Sigma_prior[1, 1])
    ax4.axhline(
        mean_prior[1] - 2 * std_theta2,
        color="purple",
        linestyle=":",
        alpha=0.8,
        label=r"$\pm 2\sigma\ (\theta_2)$",
    )
    ax4.axhline(
        mean_prior[1] + 2 * std_theta2, color="purple", linestyle=":", alpha=0.8
    )

    # --- Combine Legend Handles ---
    handles, labels = ax4.get_legend_handles_labels()
    # Add contour proxy
    handles.append(contour_proxy)
    labels.append("PriorGuide prior")
    ax4.legend(handles, labels, loc="upper right")

    # Overall title and layout adjustment.
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{prefix}_viz_oup_{num_samples}_prior_{i}.png")


def plot_prior_visualization_4param(
    all_samples,
    all_samples_with_prior,
    all_unobserved,
    theta_prior_mean,
    theta_prior_cov,
    num_samples,
    true_sampled_data=None,
    i=0,
):
    """
    Visualizes simulated data and parameter samples for Simformer and PriorGuide,
    along with the bivariate Gaussian prior and its marginal densities.

    Adds dotted lines at +/- 3 * sigma for each marginal distribution on the
    parameter space subplot.

    Parameters:
      all_samples: np.array
         Array of Simformer samples (parameters and data).
      all_samples_with_prior: np.array
         Array of PriorGuide samples (parameters and data).
      all_unobserved: np.array
         Array containing the true (unobserved) parameter and data.
      mean_prior: np.array
         Mean of the prior (at least 2D; first two entries correspond to parameters).
      Sigma_prior: np.array
         Covariance matrix of the prior (at least 2x2).
      num_samples: int
         Number of sample trajectories to plot.
      i: int, optional
         Index for selecting the sample (default is 0).
    """
    # Create a figure with 4 rows and 2 columns of subplots.
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axs.flatten()

    # --- Left subplot: Simformer Data ---
    for j in range(num_samples):
        if j == 0:
            ax1.plot(
                all_samples[i, 4:, j],
                alpha=0.3,
                color="orange",
                label="Simformer data sample",
            )
        else:
            ax1.plot(all_samples[i, 4:, j], alpha=0.3, color="orange")
        if true_sampled_data is not None:
            if j == 0:
                ax1.plot(
                    true_sampled_data[j],
                    color="grey",
                    label="~ P(x|theta)",
                    alpha=0.3,
                    linestyle="--",
                )
            else:
                ax1.plot(true_sampled_data[j], color="grey", alpha=0.3, linestyle="--")
    ax1.plot(all_unobserved[i, 4:], color="black", label="True data")
    ax1.set_title("Simformer")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_ylim(0, 1)
    ax1.legend()

    # --- Right subplot: PriorGuide Data ---
    for j in range(num_samples):
        if j == 0:
            ax2.plot(
                all_samples_with_prior[i, 4:, j],
                alpha=0.3,
                color="green",
                label="PriorGuide data sample",
            )
        else:
            ax2.plot(all_samples_with_prior[i, 4:, j], alpha=0.3, color="green")

        if true_sampled_data is not None:
            if j == 0:
                ax2.plot(
                    true_sampled_data[j],
                    color="grey",
                    label="~ P(x|theta)",
                    alpha=0.3,
                    linestyle="--",
                )
            else:
                ax2.plot(true_sampled_data[j], color="grey", alpha=0.3, linestyle="--")

    ax2.plot(all_unobserved[i, 4:], color="black", label="True data")
    ax2.set_title("PriorGuide")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_ylim(0, 1)
    ax2.legend()

    # Overall title and layout adjustment.
    fig.suptitle("Predictive prior visualization", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"viz_turin_{num_samples}_prior_{i}.png")