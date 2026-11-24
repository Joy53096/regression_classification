from pathlib import Path
import numpy as np
import pandas as pd

# Separate generators use this seed for the split and the centre ordering.
SEED = 42
M_VALUES = [10, 20, 40, 80]
WIDTH_VALUES = [0.5, 1, 2, 4, 6]
# These 20 configurations vary the number of basis functions and their spread.
# Width is measured in the standardised input space: larger values spread further.


# 1. Split by input group, keeping all observations of the same input together.
def split_groups(X):
    # Group on all eight inputs, regardless of the measured strength. Otherwise,
    # the same input could appear in training and make test performance optimistic.
    _, group_id, group_size = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    if len(group_size) < 3:
        raise ValueError("At least three distinct input groups are required.")
    rng = np.random.default_rng(SEED)
    # np.unique sorts the groups; shuffle them before choosing the split boundaries.
    group_order = rng.permutation(len(group_size))
    row_count = np.cumsum(group_size[group_order])
    # Choose whole-group boundaries near 70% and 85% of the rows. Group sizes
    # differ, so the actual proportions can deviate from 70/15/15.
    # The candidate boundaries leave at least one group in each subset.
    split_points = np.arange(1, len(group_order) - 1)
    difference = np.abs(row_count[split_points - 1] - 0.70 * len(X))
    train_end = split_points[np.argmin(difference)]
    split_points = np.arange(train_end + 1, len(group_order))
    difference = np.abs(row_count[split_points - 1] - 0.85 * len(X))
    val_end = split_points[np.argmin(difference)]
    # Return row indices so that X and y are sliced in exactly the same way.
    train_idx = np.flatnonzero(np.isin(group_id, group_order[:train_end]))
    val_idx = np.flatnonzero(np.isin(group_id, group_order[train_end:val_end]))
    test_idx = np.flatnonzero(np.isin(group_id, group_order[val_end:]))
    return train_idx, val_idx, test_idx


# 2. Least squares is used for both the raw inputs and the Gaussian features.
def add_bias(X):
    # The leading column of ones lets w[0] act as the intercept.
    return np.column_stack((np.ones(len(X)), X))


def fit_linear(X, y):
    X_bias = add_bias(X)
    # Minimise ||X_bias @ w - y||^2 without forming an inverse of X_bias.T @ X_bias.
    w, _, rank, singular_values = np.linalg.lstsq(X_bias, y, rcond=None)
    # Rank and condition number help identify dependent or nearly dependent columns.
    # They are diagnostics; validation RMSE still decides the Gaussian configuration.
    if singular_values[-1] > 0:
        cond = singular_values[0] / singular_values[-1]
    else:
        cond = np.inf
    if not np.isfinite(w).all():
        raise ValueError("Least squares produced non-finite weights.")
    return w, rank, cond


def predict(X, w):
    return add_bias(X) @ w


# 3. Map each input to its Gaussian response at M training centres.
def gaussian_features(X, centres, width):
    # X: (N, d), centres: (M, d), output: (N, M). Here d = 8.
    # Each column measures closeness to one centre; width controls how fast it falls.
    diff = X[:, None, :] - centres[None, :, :]  # Pairwise differences: (N, M, d).
    squared_distance = np.sum(diff ** 2, axis=2)
    return np.exp(-squared_distance / (2 * width ** 2))


# 4. RMSE and MAE are in MPa; MSE is in MPa^2 and R2 has no units.
def calc_metrics(y_true, y_pred):
    error = y_true - y_pred
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(error))
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    if sst > 0:
        r2 = 1 - np.sum(error ** 2) / sst
    else:
        r2 = np.nan  # R2 is undefined when all target values are identical.
    return np.array([mse, rmse, mae, r2])


# 5. Pandas only loads the CSV file; all later calculations use NumPy.
# Resolve from the script location so the working directory does not affect loading.
data_path = Path(__file__).resolve().parent / "data" / "concrete_strength.csv"
df = pd.read_csv(data_path, float_precision="round_trip")
columns = [str(name).strip() for name in df.columns]
data = df.to_numpy(dtype=float)
target_col = columns.index("Concrete compressive strength(MPa, megapascals)")
X = np.delete(data, target_col, axis=1)
y = data[:, target_col]
if not np.isfinite(data).all():
    raise ValueError("Data contains missing or infinite values.")
print("Data / X / y shapes:", data.shape, X.shape, y.shape)
# Count full-row duplicates without removing them. They still contribute to the
# fitted loss and evaluation metrics; grouping only prevents overlap across subsets.
print("Extra full-row duplicates:", len(data) - len(np.unique(data, axis=0)))

# Both models use the same fixed split, so their results are directly comparable.
train_idx, val_idx, test_idx = split_groups(X)
X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
print("Train / validation / test sizes:", len(train_idx), len(val_idx), len(test_idx))

# Estimate scaling from training inputs only; held-out data must not affect it.
# This also keeps Gaussian distances from being dominated by large-unit features.
# Keep y in MPa so the errors remain easy to interpret.
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0, ddof=0)
scale = np.where(train_std == 0, 1.0, train_std)  # Avoid division by zero for constant columns.
X_train = (X_train - train_mean) / scale
X_val = (X_val - train_mean) / scale
X_test = (X_test - train_mean) / scale

# 6. Linear baseline: eight standardised inputs plus an intercept.
w_linear, linear_rank, linear_cond = fit_linear(X_train, y_train)
# Print the training result as well, so every Linear RMSE in the regression comparison can be checked.
linear_train_scores = calc_metrics(y_train, predict(X_train, w_linear))
linear_val_scores = calc_metrics(y_val, predict(X_val, w_linear))
print(f"Linear training RMSE: {linear_train_scores[1]:.6f} MPa")
print(f"\nLinear validation RMSE: {linear_val_scores[1]:.6f} MPa")
print(f"Linear design rank: {linear_rank}/{X_train.shape[1] + 1}; condition(D): {linear_cond:.3e}")

# 7. Choose distinct training inputs as centres, with one fixed random ordering.
# Taking longer prefixes gives nested centre sets as M increases. For a fixed M,
# reuse the centres across widths so only the width changes in that comparison.
_, unique_idx = np.unique(X_train, axis=0, return_index=True)
rng = np.random.default_rng(SEED)
centre_order = rng.permutation(unique_idx)
if len(centre_order) < max(M_VALUES):
    raise ValueError("Not enough distinct training inputs for this grid.")
best_rmse = np.inf
best_model = None
grid_results = []
print("\n   M   Width   Train RMSE     Val RMSE    Rank    Condition(D)")
for m in M_VALUES:
    centres = X_train[centre_order[:m]]
    for width in WIDTH_VALUES:
        phi_train = gaussian_features(X_train, centres, width)
        phi_val = gaussian_features(X_val, centres, width)
        w, rank, cond = fit_linear(phi_train, y_train)  # Fit new weights for each feature map.
        train_rmse = calc_metrics(y_train, predict(phi_train, w))[1]
        val_rmse = calc_metrics(y_val, predict(phi_val, w))[1]
        grid_results.append((m, width, train_rmse, val_rmse, rank, cond))
        print(f"{m:4d} {width:7g} {train_rmse:12.6f} {val_rmse:12.6f} {rank:3d}/{m + 1:<3d} {cond:14.3e}")
        # Training RMSE describes fit; validation RMSE selects the configuration.
        # Keep test data out of this decision. Exact ties retain the first entry.
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_model = (m, width, centres.copy(), w.copy())
if best_model is None:
    raise ValueError("No Gaussian configuration produced a finite validation RMSE.")

# Show the strongest validation candidates; stable sorting preserves the grid order in ties.
ranking = sorted(grid_results, key=lambda row: row[3])
print("\nTop 5 configurations by validation RMSE (MPa):")
print("Rank    M   Width   Train RMSE     Val RMSE")
for position, result in enumerate(ranking[:5], start=1):
    m, width, train_rmse, val_rmse = result[:4]
    print(f"{position:4d} {m:4d} {width:7g} {train_rmse:12.6f} {val_rmse:12.6f}")

# 8. Evaluate after selection, using the weights already fitted on training data.
# There is no train-plus-validation refit here, and test scores do not tune the models.
best_m, best_width, best_centres, w_gaussian = best_model
# Rebuild the selected features: phi_train left by the loop belongs to its last entry.
best_phi_train = gaussian_features(X_train, best_centres, best_width)
# Print the selected model directly, so its the regression comparison training RMSE is easy to find.
gaussian_train_rmse = calc_metrics(y_train, predict(best_phi_train, w_gaussian))[1]
print(f"Gaussian training RMSE: {gaussian_train_rmse:.6f} MPa")
if not np.isfinite(best_phi_train).all() or np.any((best_phi_train < 0) | (best_phi_train > 1)):
    raise ValueError("Gaussian features must be finite and within [0, 1].")
print(f"\nSelected Gaussian: M={best_m}, width={best_width:g}; validation RMSE={best_rmse:.6f} MPa")
print(f"Selected training feature range: [{best_phi_train.min():.6g}, {best_phi_train.max():.6g}]")
linear_pred = predict(X_test, w_linear)
phi_test = gaussian_features(X_test, best_centres, best_width)
gaussian_pred = predict(phi_test, w_gaussian)
linear_scores = calc_metrics(y_test, linear_pred)
gaussian_scores = calc_metrics(y_test, gaussian_pred)
print(f"\n{'Test model':12s} {'MSE (MPa^2)':>14s} {'RMSE (MPa)':>14s} {'MAE (MPa)':>14s} {'R2':>14s}")
for name, scores in [("Linear", linear_scores), ("Gaussian", gaussian_scores)]:
    mse, rmse, mae, r2 = scores
    print(f"{name:12s} {mse:14.6f} {rmse:14.6f} {mae:14.6f} {r2:14.6f}")
if linear_scores[1] > 0:
    improvement = 100 * (1 - gaussian_scores[1] / linear_scores[1])
else:
    improvement = np.nan
print(f"Gaussian test RMSE reduction relative to Linear: {improvement:.2f}%")

# Save the same test predictions used above, rather than recreating them for Figure 4.
# Row indices refer to the original data rows, starting at zero (not Excel row numbers).
output_dir = Path(__file__).resolve().parent / "outputs" / "regression"
output_dir.mkdir(parents=True, exist_ok=True)
prediction_path = output_dir / "test_predictions.csv"
np.savetxt(
    prediction_path,
    np.column_stack((test_idx, y_test, linear_pred, gaussian_pred)),
    delimiter=",",
    header="source_row_index,actual_strength_MPa,linear_prediction_MPa,gaussian_prediction_MPa",
    comments="",
    fmt=["%d", "%.17g", "%.17g", "%.17g"],
)
print(f"Test predictions saved to: {prediction_path}")

# Figure 2 data: keep all candidate scores at full precision for plot.py.
grid_path = output_dir / "gaussian_grid.csv"
np.savetxt(
    grid_path, np.asarray(grid_results), delimiter=",", comments="",
    header="n_centres,width,train_RMSE,validation_RMSE,rank,design_condition",
    fmt=["%d", "%.17g", "%.17g", "%.17g", "%d", "%.17g"],
)
print(f"Gaussian grid saved to: {grid_path}")
