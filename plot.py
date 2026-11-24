# Draw the numerical figures used in the report after running both experiments.
# Run: python plot.py
from pathlib import Path
import csv
import math
import numpy as np
import matplotlib

# Write files without opening interactive windows.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


# Match the fonts, colours and line styles of Figures 2-5 in the report.
NAVY = '#2F5D8A'
BLUE = '#5B8DB8'
TEAL = '#2E8B8B'
GRAY = '#6E7781'
GRID = '#D9DEE5'
RED = '#C65D4B'
INK = '#253747'

plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 10.5,
    'axes.titlesize': 13, 'axes.titleweight': 'normal',
    'axes.labelsize': 11, 'axes.labelcolor': INK,
    'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
    'axes.edgecolor': GRAY, 'axes.linewidth': .7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'grid.color': GRID, 'grid.linewidth': .55, 'grid.alpha': .55,
    'legend.frameon': False, 'legend.fontsize': 10,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
})


def save_report_figure(fig, output_dir, name):
    # PNG is suitable for Word; PDF keeps text and lines sharp when scaled.
    for extension in ["png", "pdf"]:
        path = output_dir / f"{name}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.08, facecolor="white")
        print(f"Report figure saved to: {path}")
    plt.close(fig)


def readable_text(cmap, normalized_value):
    r, g, b, _ = cmap(normalized_value)
    return 'white' if .2126*r + .7152*g + .0722*b < .53 else INK


# Figure 2: validation RMSE for all 20 Gaussian configurations.
def plot_gaussian_heatmap(grid_results, selected_m, selected_width, output_dir):
    # grid_results contains this run's training/validation scores, before rounding.
    ms = sorted({int(row[0]) for row in grid_results})
    widths = sorted({row[1] for row in grid_results})
    values = [[next(row[3] for row in grid_results if row[0] == m and row[1] == width)
               for width in widths] for m in ms]
    i, j = ms.index(selected_m), widths.index(selected_width)
    low, high = min(map(min, values)), max(map(max, values))
    cmap = LinearSegmentedColormap.from_list('low_rmse_dark_blue', [NAVY, '#7FA6C8', '#F0F4F8'])
    fig, ax = plt.subplots(figsize=(6.7, 3.55), layout='constrained')
    image = ax.imshow(values, cmap=cmap, vmin=low, vmax=high, aspect='auto')
    for row in range(len(ms)):
        for column in range(len(widths)):
            value = values[row][column]
            color = readable_text(cmap, (value-low)/(high-low))
            if (row, column) == (i, j):
                ax.text(column, row, f'{value:.3f}', ha='center', va='center',
                        color='white', fontsize=11, fontweight='bold')
            else:
                ax.text(column, row, f'{value:.3f}', ha='center', va='center',
                        color=color, fontsize=10.5)
    # Slightly inset the outline so the lower edge remains visible at the plot boundary.
    ax.add_patch(Rectangle((j-.47, i-.47), .94, .94, fill=False, edgecolor=RED, linewidth=2.8))
    ax.set(xticks=range(len(widths)), xticklabels=[f'{w:g}' for w in widths],
           yticks=range(len(ms)), yticklabels=ms,
           xlabel='Width', ylabel='Number of centres (M)', title='Gaussian Validation RMSE')
    ax.set_xticks([k-.5 for k in range(len(widths)+1)], minor=True)
    ax.set_yticks([k-.5 for k in range(len(ms)+1)], minor=True)
    ax.grid(which='minor', color='white', linewidth=1)
    ax.tick_params(which='both', length=0, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=.045, pad=.035)
    colorbar.set_label('RMSE (MPa)', fontsize=10.5)
    colorbar.outline.set_linewidth(.5)
    save_report_figure(fig, output_dir, 'figure2_gaussian_validation_rmse_heatmap')


# Figure 3: training and validation BCE at the three iteration budgets.
def plot_logistic_budgets(iteration_results, output_dir):
    # Match the six-decimal endpoints printed in the experiment table and original figure.
    # This rounding is only for display; model selection uses the unrounded validation BCE.
    rows = [(n, round(float(train), 6), round(float(val), 6))
            for n, train, val in iteration_results]
    iterations = [r[0] for r in rows]
    training = [r[1] for r in rows]
    validation = [r[2] for r in rows]
    selected = min(range(len(rows)), key=lambda i: validation[i])
    fig, ax = plt.subplots(figsize=(6.7, 3.55), layout='constrained')
    ax.plot(iterations, training, 's--', color='#99ACBF', lw=1.1, ms=5, label='Training BCE')
    ax.plot(iterations, validation, 'o-', color=TEAL, lw=1.3, ms=5.5, label='Validation BCE')
    xy = iterations[selected], validation[selected]
    ax.scatter(*xy, s=90, facecolors='white', edgecolors=RED, linewidths=2, zorder=4)
    ax.scatter(*xy, s=17, color=TEAL, zorder=5)
    ax.annotate(f'Selected: {iterations[selected]:,}', xy=xy, xytext=(-10,18),
                textcoords='offset points', color=RED, fontsize=10,
                ha='right', va='bottom')
    ax.set(xlabel='Iterations', ylabel='Binary cross-entropy (BCE)',
           title='Logistic Regression: BCE at Different Iteration Budgets',
           xticks=iterations, xlim=(650, 5450), ylim=(0, .085))
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    ax.legend(loc='upper right')
    save_report_figure(fig, output_dir, 'figure3_logistic_bce_iterations')


# Figure 4: actual versus predicted strength for the 155 test records.
def plot_regression_predictions(prediction_file, output_dir):
    # Figure 4 reads the CSV just written by this run; no stored model is loaded.
    rows = np.genfromtxt(prediction_file, delimiter=",", names=True)
    actual = rows["actual_strength_MPa"].tolist()
    predictions = [rows[name].tolist() for name in
                   ["linear_prediction_MPa", "gaussian_prediction_MPa"]]
    scores = [math.sqrt(sum((a-p)**2 for a, p in zip(actual, values))/len(actual))
              for values in predictions]
    lower = min(0, math.floor(min(min(col) for col in [actual,*predictions])/10)*10)
    upper = math.ceil(max(max(col) for col in [actual,*predictions])/10)*10
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.8), layout='constrained')
    fig.suptitle('Actual vs Predicted Concrete Strength', fontsize=14)
    for ax, name, values, rmse, color in zip(axes, ['Linear Regression','Gaussian Basis Regression'],
                                            predictions, scores, [BLUE,BLUE]):
        ax.plot([lower,upper], [lower,upper], '--', color=GRAY, lw=1.1, zorder=1)
        ax.scatter(actual, values, s=14, alpha=.70, color=color, edgecolors='none', zorder=2)
        ax.set(xlim=(lower,upper), ylim=(lower,upper), xlabel='Actual strength (MPa)',
               ylabel='Predicted strength (MPa)', title=name)
        ax.set_aspect('equal', adjustable='box')
        ax.text(.04, .96, f'RMSE = {rmse:.3f} MPa', transform=ax.transAxes,
                ha='left', va='top', fontsize=10)
        ax.grid()
        ax.set_axisbelow(True)
    save_report_figure(fig, output_dir, 'figure4_actual_vs_predicted')


# Figure 5: final test confusion matrices for the two classifiers.
def plot_test_confusion(test_matrices, output_dir):
    # Reuse the test matrices already printed by compare_models, in the same model order.
    # Rows are true labels and columns are predicted labels; Class 1 is positive.
    matrices = [matrix.tolist() for name, matrix in test_matrices]
    cmap = LinearSegmentedColormap.from_list('confusion_teal', ['#F1F7F7', '#92C1BF', '#276F72'])
    maximum = math.ceil(max(max(map(max, matrix)) for matrix in matrices) / 10) * 10
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.6), layout='constrained')
    fig.suptitle('Test-set Confusion Matrices', fontsize=14)
    for ax, title, matrix in zip(axes, ['Fisher', 'Logistic Regression'], matrices):
        image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=maximum)
        for i in range(2):
            for j in range(2):
                is_fp = (i,j) == (0,1)
                ax.text(j, i, str(matrix[i][j]), ha='center', va='center',
                        color=RED if is_fp else readable_text(cmap, matrix[i][j]/maximum),
                        fontsize=18 if is_fp else 16,
                        fontweight='bold' if is_fp else 'normal')
        ax.add_patch(Rectangle((.53,-.47), .94, .94, fill=False,
                               edgecolor=RED, linewidth=1.5))
        ax.text(1, .31, 'False positives', ha='center', va='center',
                color=RED, fontsize=9)
        ax.set(xticks=[0,1], yticks=[0,1], xticklabels=['Class 0','Class 1'],
               yticklabels=['Class 0','Class 1'], xlabel='Predicted label',
               ylabel='True label', title=title)
        ax.tick_params(length=0, pad=5)
        ax.set_xticks([-.5,.5,1.5], minor=True)
        ax.set_yticks([-.5,.5,1.5], minor=True)
        ax.grid(which='minor', color='white', linewidth=1.3)
        ax.tick_params(which='minor', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=axes, fraction=.038, pad=.025)
    colorbar.set_label('Count', fontsize=10.5)
    colorbar.outline.set_linewidth(.5)
    save_report_figure(fig, output_dir, 'figure5_confusion_matrix_comparison')


def main():
    root = Path(__file__).resolve().parent
    regression_dir = root / "outputs" / "regression"
    classification_dir = root / "outputs" / "classification"
    grid_path = regression_dir / "gaussian_grid.csv"
    prediction_path = regression_dir / "test_predictions.csv"
    budget_path = classification_dir / "logistic_iterations.csv"
    confusion_path = classification_dir / "test_confusion_matrices.csv"
    for path in [grid_path, prediction_path, budget_path, confusion_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path.name}. Run both experiment scripts first.")

    # Read recorded experiment results only; plotting never fits or changes a model.
    grid = np.loadtxt(grid_path, delimiter=",", skiprows=1)
    selected = grid[np.argmin(grid[:, 3])]
    budgets = np.loadtxt(budget_path, delimiter=",", skiprows=1)
    iteration_results = [(int(row[0]), row[1], row[2]) for row in budgets]
    with confusion_path.open(encoding="utf-8", newline="") as stream:
        test_matrices = [
            (row["model"], np.array([[int(row["TN"]), int(row["FP"])],
                                      [int(row["FN"]), int(row["TP"])]]))
            for row in csv.DictReader(stream)
        ]
    output_dir = root / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_gaussian_heatmap(grid.tolist(), int(selected[0]), selected[1], output_dir)
    plot_logistic_budgets(iteration_results, output_dir)
    plot_regression_predictions(prediction_path, output_dir)
    plot_test_confusion(test_matrices, output_dir)


if __name__ == "__main__":
    main()
