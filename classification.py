from pathlib import Path
import csv
import numpy as np

# Keep the split reproducible and fix these settings before comparing test results.
SEED = 42
LEARNING_RATE = 0.1
ITER_VALUES = [1000, 2000, 5000]
PROB_THRESHOLD = 0.5


def split_data(y):
    # Split within each class to keep class proportions close to those in the dataset.
    # Rounding fixes the total sizes; any remaining rows go into the test subset.
    rng = np.random.default_rng(SEED)
    n_train = round(0.70 * len(y))
    n_val = round(0.15 * len(y))
    class_counts = np.bincount(y, minlength=2)

    # Round the class 0 share; class 1 gets the remaining places.
    train_0 = round(n_train * class_counts[0] / len(y))
    val_0 = round(n_val * class_counts[0] / len(y))
    train_counts = [train_0, n_train - train_0]
    val_counts = [val_0, n_val - val_0]
    train_parts, val_parts, test_parts = [], [], []

    for label in [0, 1]:
        idx = rng.permutation(np.flatnonzero(y == label))
        train_end = train_counts[label]
        val_end = train_end + val_counts[label]
        train_parts.append(idx[:train_end])
        val_parts.append(idx[train_end:val_end])
        test_parts.append(idx[val_end:])

    # Shuffle again after combining classes so each subset is not ordered by label.
    train_idx = rng.permutation(np.concatenate(train_parts))
    val_idx = rng.permutation(np.concatenate(val_parts))
    test_idx = rng.permutation(np.concatenate(test_parts))
    # Every row index must occur exactly once across the three subsets.
    all_idx = np.concatenate((train_idx, val_idx, test_idx))
    if not np.array_equal(np.sort(all_idx), np.arange(len(y))):
        raise ValueError("The data split has missing or repeated row indices.")
    return train_idx, val_idx, test_idx


def prepare_data(data_path):
    data = np.loadtxt(data_path, delimiter=",")
    if data.ndim != 2 or data.shape[1] != 5:
        raise ValueError("Expected four features and one class label.")
    print("=== DATA INSPECTION ===")
    print("Raw data shape:", data.shape)
    print("Missing values:", np.isnan(data).sum())
    print("Infinite values:", np.isinf(data).sum())
    if not np.isfinite(data).all():
        raise ValueError("Missing or infinite values need to be handled first.")
    if not np.isin(data[:, 4], [0, 1]).all():
        raise ValueError("Class labels must be 0 or 1.")
    print("Raw class counts [0, 1]:", np.bincount(data[:, 4].astype(int)))
    print("Feature order: variance, skewness, curtosis, entropy")
    print("Feature minimum:", data[:, :4].min(axis=0))
    print("Feature maximum:", data[:, :4].max(axis=0))

    # Remove identical feature-label rows before splitting so they cannot appear
    # on both sides of an evaluation. Keep the first occurrence in the original order.
    _, first_idx = np.unique(data, axis=0, return_index=True)
    print("Full-row duplicates removed:", len(data) - len(first_idx))
    data = data[np.sort(first_idx)]
    X = data[:, :4]
    y = data[:, 4].astype(int)
    print("Clean X / y shapes:", X.shape, y.shape)
    print("Clean class counts [0, 1]:", np.bincount(y))
    # Equal inputs with different labels survive full-row deduplication. Stop for
    # review rather than putting these inputs into different subsets by chance.
    if len(np.unique(X, axis=0)) != len(X):
        raise ValueError("Repeated inputs remain; review them before splitting.")

    train_idx, val_idx, test_idx = split_data(y)
    for name, idx in [("Train", train_idx), ("Validation", val_idx), ("Test", test_idx)]:
        print(f"{name}: {len(idx)} rows ({len(idx) / len(y):.2%}), classes={np.bincount(y[idx], minlength=2)}")
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    # Learn the mean and scale from training inputs only to avoid data leakage.
    # Reuse them for validation and test data; class labels do not need scaling.
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0, ddof=0)
    scale = np.where(train_std == 0, 1.0, train_std)  # Constant columns must not cause division by zero.
    X_train = (X_train - train_mean) / scale
    X_val = (X_val - train_mean) / scale
    X_test = (X_test - train_mean) / scale
    print("Scaled training means:", X_train.mean(axis=0))
    print("Scaled training standard deviations:", X_train.std(axis=0))
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_fisher(X, y):
    X0, X1 = X[y == 0], X[y == 1]
    m0, m1 = X0.mean(axis=0), X1.mean(axis=0)
    diff0, diff1 = X0 - m0, X1 - m1
    # Within-class scatter measures spread around each class's own mean.
    sw = diff0.T @ diff0 + diff1.T @ diff1
    if np.linalg.matrix_rank(sw) < X.shape[1]:
        raise ValueError("Fisher within-class scatter matrix is singular.")
    # Weight the mean difference by inverse scatter: separation is more useful
    # along directions where the classes have less internal variation.
    w = np.linalg.inv(sw) @ (m1 - m0)
    # Use the midpoint of the projected means as a fixed decision rule.
    # This threshold is fitted from training data, not tuned on validation or test.
    threshold = (m0 @ w + m1 @ w) / 2
    return w, threshold


def predict_fisher(X, w, threshold):
    # The direction points towards class 1, so higher projections predict class 1.
    return (X @ w >= threshold).astype(int)


def sigmoid(z):
    # Clipping avoids overflow when evaluating the exponential.
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def cross_entropy(y, p):
    # log(0) would make BCE infinite. Clip only for loss evaluation; the gradient
    # uses sigmoid probabilities directly, without this extra probability clipping.
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_logistic(X, y, learning_rate=LEARNING_RATE, n_iter=1000):
    # Start every iteration budget from the same weights and intercept.
    # The first loss is the untrained model (p = 0.5); later entries follow updates.
    w = np.zeros(X.shape[1])
    b = 0.0
    loss_history = [cross_entropy(y, sigmoid(X @ w + b))]
    for step in range(n_iter):
        p = sigmoid(X @ w + b)
        # For mean BCE, the derivative with respect to each logit is (p - y) / N.
        # Use every training row in each update: this is batch gradient descent.
        error = p - y
        grad_w = X.T @ error / len(y)
        grad_b = error.mean()
        # The intercept has its own gradient because it is shared by all samples.
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b
        loss_history.append(cross_entropy(y, sigmoid(X @ w + b)))
    loss_history = np.array(loss_history)
    # This checks for an overall loss decrease; it does not prove convergence.
    if not np.isfinite(loss_history).all() or loss_history[-1] >= loss_history[0]:
        raise ValueError("Logistic loss is non-finite or has not decreased.")
    return w, b, loss_history


def calc_metrics(y_true, y_pred):
    # Class 1 is positive. Rows = actual class, columns = predicted class.
    # TN/TP are correct negatives/positives; FP is a false alarm, FN a missed positive.
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    accuracy = (tn + tp) / len(y_true)
    # Use 0.0 when a metric has a zero denominator, keeping the output finite.
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return np.array([accuracy, precision, recall, f1]), np.array([[tn, fp], [fn, tp]])


def compare_models(name, X, y, w_fisher, fisher_threshold, w_logistic, b_logistic):
    fisher_pred = predict_fisher(X, w_fisher, fisher_threshold)
    logistic_prob = sigmoid(X @ w_logistic + b_logistic)
    # Apply the same preset probability threshold to every subset.
    logistic_pred = (logistic_prob >= PROB_THRESHOLD).astype(int)
    print(f"\n=== {name.upper()} RESULTS ===")
    print(f"{'Model':12s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    matrices = []
    for model, prediction in [("Fisher", fisher_pred), ("Logistic", logistic_pred)]:
        scores, matrix = calc_metrics(y, prediction)
        accuracy, precision, recall, f1 = scores
        print(f"{model:12s} {accuracy:10.6f} {precision:10.6f} {recall:10.6f} {f1:10.6f}")
        matrices.append((model, matrix))
    for model, matrix in matrices:
        print(f"{model} confusion matrix [[TN, FP], [FN, TP]]:")
        print(matrix)

    return matrices


def main():
    # Resolve from this file so running from another directory still finds the data.
    data_path = Path(__file__).resolve().parent / "data" / "data_banknote_authentication.txt"
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(data_path)
    w_fisher, fisher_threshold = fit_fisher(X_train, y_train)

    # Compare three fixed iteration budgets. Validation BCE measures probability
    # quality without the information lost by thresholding to class labels.
    # It selects the budget; test accuracy plays no part in that choice.
    best_val_loss = np.inf
    best_model = None
    iteration_results = []  # Keep the endpoints for Figure 3 from the actual training runs.
    print("\n=== LOGISTIC ITERATION CHECK ===")
    print(f"{'Iterations':>12s} {'Train BCE':>12s} {'Validation BCE':>16s}")
    for n_iter in ITER_VALUES:
        w, b, loss_history = fit_logistic(X_train, y_train, n_iter=n_iter)
        val_loss = cross_entropy(y_val, sigmoid(X_val @ w + b))
        iteration_results.append((n_iter, loss_history[-1], val_loss))
        print(f"{n_iter:12d} {loss_history[-1]:12.6f} {val_loss:16.6f}")
        # Save the trained candidate itself; exact ties keep the earlier budget.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = (w.copy(), b, loss_history, n_iter)
    if best_model is None:
        raise ValueError("No finite validation loss was obtained.")
    w_logistic, b_logistic, losses, best_iter = best_model

    print("\n=== TRAINING ===")
    print("Fisher weights:", w_fisher)
    print("Fisher midpoint threshold:", fisher_threshold)
    print(f"Logistic: learning rate={LEARNING_RATE}, selected iterations={best_iter}, threshold={PROB_THRESHOLD}")
    print(f"Selected validation BCE: {best_val_loss:.6f}")
    print(f"Logistic training loss: {losses[0]:.6f} -> {losses[-1]:.6f}")
    checkpoints = [0, best_iter // 10, best_iter // 2, best_iter]
    print("Training loss at iterations", checkpoints, ":", losses[checkpoints])
    print("Logistic weights:", w_logistic)
    print("Logistic intercept:", b_logistic)
    compare_models("Training", X_train, y_train, w_fisher, fisher_threshold, w_logistic, b_logistic)
    compare_models("Validation", X_val, y_val, w_fisher, fisher_threshold, w_logistic, b_logistic)
    # Evaluate the selected training-fitted models with both decision thresholds fixed.
    # Test scores describe this split; do not use them to revise the settings.
    test_matrices = compare_models("Test", X_test, y_test, w_fisher, fisher_threshold, w_logistic, b_logistic)
    print(f"\nOne test sample corresponds to {100 / len(y_test):.3f} percentage points of accuracy.")
    print("Results describe this fixed split and do not establish a universal model ranking.")


    # Figure 3 data: save the actual loss endpoints, without display rounding.
    output_dir = Path(__file__).resolve().parent / "outputs" / "classification"
    output_dir.mkdir(parents=True, exist_ok=True)
    budget_path = output_dir / "logistic_iterations.csv"
    np.savetxt(
        budget_path, np.asarray(iteration_results), delimiter=",", comments="",
        header="iterations,training_BCE,validation_BCE", fmt=["%d", "%.17g", "%.17g"],
    )
    # Figure 5 data: export the same matrices printed in the final test evaluation.
    confusion_path = output_dir / "test_confusion_matrices.csv"
    with confusion_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "TN", "FP", "FN", "TP"])
        for name, matrix in test_matrices:
            writer.writerow([name, *matrix.ravel()])
    print(f"Logistic iteration results saved to: {budget_path}")
    print(f"Test confusion matrices saved to: {confusion_path}")


if __name__ == "__main__":
    main()
