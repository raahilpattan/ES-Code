"""
runs the full electrospinning video monitoring pipeline. this script extracts features from a video, cleans 
the data, performs CP decomposition, computes reconstruction errors, detects anomalies, applies EWMA monitoring, 
saves all results, and visualizes the monitoring and reconstruction outputs.

requires: numpy, opencv, tensorly, scikit-learn, matplotlib
"""

import numpy as np
from feature_extraction import process_video
from tensor_utils import cp_decomposition, reconstruct_cp, compute_reconstruction_error
from monitoring import lasso_bic_anomaly, ewma_monitoring
from visualization import (
    plot_ewma,
    plot_anomaly_magnitude,
    plot_actual_vs_reconstructed,
    plot_side_top_difference, 
    plot_side_vs_top_scatter
)

# step 1: load and process video
# extracts features from both camera views of the input video
video_path = "C:/Users/raahi/Downloads/ES Videos/no_anomaly_12kV_1.0mLhr_15cm_10_v1.mov" # insert video path
features = process_video(video_path)

#error handling
if features.size == 0 or len(features.shape) != 2:
    raise ValueError(f"video feature extraction failed: {video_path} (is the path and format correct?)")


# step 1.5: clean data
# removes rows with NaNs and replaces any remaining NaNs or infs with zero
features = features[~np.isnan(features).any(axis=1)]  # Drop rows with NaNs
features = np.nan_to_num(features)  # Replace NaNs or inf with 0
np.save("features_tensor.npy", features)

# step 2: CP Decomposition
# decomposes the feature matrix into low-rank factors and reconstructs the matrix
cp_factors = cp_decomposition(features, rank=3)  # returns [A, B]
A, B = cp_factors  # unpack factor matrices
reconstructed = reconstruct_cp((A, B))  # manual reconstruction

# step 3: compute the errors and detect anomalies
# calculates the per-frame reconstruction error, detects anomalies, and computes EWMA statistics
errors = compute_reconstruction_error(features, reconstructed)
anomalies = lasso_bic_anomaly(features - reconstructed)
ewma_stat = ewma_monitoring(errors)

# saves all main arrays for later analysis or visualization
np.save("cp_reconstruction.npy", reconstructed)
np.save("anomalies.npy", anomalies)
np.save("ewma_stat.npy", ewma_stat)


# EWMA monitoring chart
plot_ewma(ewma_stat)

# anomaly magnitude by feature group
plot_anomaly_magnitude(anomalies)

# actual vs reconstructed: jet angle (side + top)
plot_actual_vs_reconstructed(
    features, reconstructed, [4, 8],
    "Jet Angle: Actual vs Reconstructed", "Angle (degrees)",
    ["Actual Side Angle", "Reconstructed Side Angle", "Actual Top Angle", "Reconstructed Top Angle"]
)

# actual vs reconstructed: symmetry (side + top)
plot_actual_vs_reconstructed(
    features, reconstructed, [6, 10],
    "Jet Symmetry: Actual vs Reconstructed", "Symmetry Score",
    ["Actual Side Symmetry", "Reconstructed Side Symmetry", "Actual Top Symmetry", "Reconstructed Top Symmetry"]
)

# actual vs reconstructed: cone width (side + top)
plot_actual_vs_reconstructed(
    features, reconstructed, [5, 9],
    "Cone Width: Actual vs Reconstructed", "Width (pixels)",
    ["Actual Side Width", "Reconstructed Side Width", "Actual Top Width", "Reconstructed Top Width"]
)


# axisymmetry: difference plots (side - top)
plot_side_top_difference(
    features, 4, 8, "Angle Difference (deg)",
    "Side vs Top Jet Angle Difference"
)
plot_side_top_difference(
    features, 5, 9, "Width Difference (pixels)",
    "Side vs Top Cone Width Difference"
)
plot_side_top_difference(
    features, 6, 10, "Symmetry Difference",
    "Side vs Top Symmetry Difference"
)

# axisymmetry: scatter plots (side vs top)
plot_side_vs_top_scatter(
    features, 4, 8, "Side Angle (deg)", "Top Angle (deg)",
    "Side vs Top Jet Angle"
)
plot_side_vs_top_scatter(
    features, 5, 9, "Side Width (pixels)", "Top Width (pixels)",
    "Side vs Top Cone Width"
)
plot_side_vs_top_scatter(
    features, 6, 10, "Side Symmetry", "Top Symmetry",
    "Side vs Top Symmetry"
)


print("Monitoring complete. EWMA statistics and anomalies saved.")

# -------ignore------
#print("side actual width (first 10):", features[:10, 5])
#print("side reconstructed width (first 10):", reconstructed[:10, 5])
#print("top actual width (first 10):", features[:10, 9])
#print("top reconstructed width (first 10):", reconstructed[:10, 9])

