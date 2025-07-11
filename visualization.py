import numpy as np
import matplotlib.pyplot as plt

def plot_ewma(ewma_stat):
    """
    plots the ewma monitoring chart for the process.
    shows the ewma statistic for each frame, with a 3-sigma upper control limit.

    parameters:
        ewma_stat (np.ndarray): 1d array of ewma statistics
    """
    frames = np.arange(len(ewma_stat))
    plt.figure(figsize=(10, 4))
    plt.plot(frames, ewma_stat, label="EWMA Monitoring Stat", linewidth=2)
    plt.axhline(np.mean(ewma_stat) + 3*np.std(ewma_stat), color='r', linestyle='--', label="UCL (3σ)")
    plt.title("EWMA Control Chart")
    plt.xlabel("Frame")
    plt.ylabel("Monitoring Stat")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_anomaly_magnitude(anomalies):
    """
    plots the anomaly magnitude for each feature group over time.
    uses the L2 norm of anomaly scores for centerline, side, and top geometry groups.

    parameters:
        anomalies (np.ndarray): 2d array (frames x features) of anomaly scores
    """
    frames = np.arange(len(anomalies))
    plt.figure(figsize=(12, 6))
    plt.plot(frames, np.linalg.norm(anomalies[:, 0:4], axis=1), label="Centerline (cx, cy)", linewidth=2)
    plt.plot(frames, np.linalg.norm(anomalies[:, 4:8], axis=1), label="Side Geometry", linewidth=2)
    plt.plot(frames, np.linalg.norm(anomalies[:, 8:12], axis=1), label="Top Geometry", linewidth=2)
    plt.title("Anomaly Magnitude by Feature Category")
    plt.xlabel("Frame")
    plt.ylabel("L2 Norm of Anomaly")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_actual_vs_reconstructed(features, reconstructed, idx, title, ylabel, labels):
    """
    plots actual vs reconstructed values for two selected features (e.g., side and top for angle, symmetry, or width).

    parameters:
        features (np.ndarray): original feature matrix
        reconstructed (np.ndarray): reconstructed feature matrix
        idx (list): list of two indices to plot
        title (str): plot title
        ylabel (str): y-axis label
        labels (list): list of four labels [actual1, recon1, actual2, recon2]
    """
    frames = np.arange(len(features))
    plt.figure(figsize=(12, 5))
    lines = []
    # only plots if not all NaN or constant
    for i, label in zip(idx, labels):
        if not np.all(np.isnan(features[:, i])) and not np.all(features[:, i] == features[:, i][0]):
            l, = plt.plot(frames, features[:, i], label=label, alpha=0.7)
            lines.append(l)
        if not np.all(np.isnan(reconstructed[:, i])) and not np.all(reconstructed[:, i] == reconstructed[:, i][0]):
            l, = plt.plot(frames, reconstructed[:, i], label="Reconstructed " + label, linestyle='--')
            lines.append(l)
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def plot_side_top_difference(features, idx_side, idx_top, ylabel, title):
    """
    plots the difference between a side-view feature and its corresponding top-view feature for each frame.
    this helps visualize deviations from axisymmetry: values near zero indicate axisymmetry, while larger values indicate asymmetry.

    parameters:
        features (np.ndarray): feature matrix, shape (frames, num_features)
        idx_side (int): column index for the side-view feature (e.g., 4 for angle)
        idx_top (int): column index for the top-view feature (e.g., 8 for angle)
        ylabel (str): label for the y-axis (e.g., "Angle Difference (deg)")
        title (str): title for the plot
    """
    diff = features[:, idx_side] - features[:, idx_top]
    frames = np.arange(len(diff))
    plt.figure(figsize=(10, 4))
    plt.plot(frames, diff, label=f"{ylabel} difference (side - top)")
    plt.axhline(0, color='k', linestyle='--')
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Difference")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_side_vs_top_scatter(features, idx_side, idx_top, xlabel, ylabel, title):
    """
    creates a scatter plot comparing a side-view feature to its corresponding top-view feature across all frames.
    points along the diagonal indicate axisymmetry; deviations from the diagonal indicate asymmetry.

    parameters:
        features (np.ndarray): feature matrix, shape (frames, num_features)
        idx_side (int): column index for the side-view feature (e.g., 4 for angle)
        idx_top (int): column index for the top-view feature (e.g., 8 for angle)
        xlabel (str): label for the x-axis (side-view feature)
        ylabel (str): label for the y-axis (top-view feature)
        title (str): title for the plot
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(features[:, idx_side], features[:, idx_top], alpha=0.5)
    plt.plot([features[:, idx_side].min(), features[:, idx_side].max()],
             [features[:, idx_side].min(), features[:, idx_side].max()],
             'r--', label="Perfect symmetry")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
