import os
import sys
import torch
import sympy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Configuration
sys.setrecursionlimit(10000)
sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.linewidth": 1.0,
    "figure.dpi": 600
})

DIR_CHK = r"./checkpoints"
DIR_FIGS = r"./figures"
os.makedirs(DIR_FIGS, exist_ok=True)

dataset_file = os.path.join(DIR_CHK, "dataset_tensor_mc_physics.pt")
equation_file = os.path.join(DIR_CHK, "boundary_equations.txt")

target_classes = ["Generator Trip", "Line Trip", "Load Trip"]
phases = ["W1", "W2", "W3", "W4", "W5"]

# Dataset Loading
print("Loading test dataset...")
dataset_mc = torch.load(dataset_file, weights_only=False)
X_test_full = dataset_mc['test_input'].numpy()
Y_test_full = torch.argmax(dataset_mc['test_label'], dim=1).numpy()

# Exclude Normal class
mask_faults = Y_test_full != 0
X_test = X_test_full[mask_faults]
y_true = Y_test_full[mask_faults] - 1

dim_total = X_test.shape[1]
N = dim_total // 15
print(f" -> Fault samples: {X_test.shape[0]} | Nodes: {N}")

# Symbolic Parsing
print("\nLoading and patching symbolic equations")
all_equations = []

with open(equation_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    
for i, line in enumerate(lines):
    line = line.strip()
    if line.startswith("[") and line.endswith("]"):
        all_equations.append(lines[i+1].strip())

# Skip Normal equation to evaluate only faults
if len(all_equations) >= 4:
    equations_str = all_equations[1:4]
elif len(all_equations) == 3:
    equations_str = all_equations
else:
    print(f"\n[CRITICAL ERROR] Found {len(all_equations)} equations. Expected 4.")
    sys.exit()

symbolic_vars = []
for phase in phases:
    for prefix in ["λV", "λf_gen", "λf_load"]:
        for i in range(N):
            symbolic_vars.append(sympy.Symbol(f"{prefix}_{i}_{phase}"))

evaluable_functions = []
for eq_str in equations_str:
    expr = sympy.sympify(eq_str)
    expr = expr.replace(sympy.exp, sympy.Abs)
    func = sympy.lambdify(symbolic_vars, expr, modules=["numpy"])
    evaluable_functions.append(func)

# Mass Evaluation
print(f"\nEvaluating {X_test.shape[0]} fault samples...")
raw_scores = []
overflow_count = 0

for i in range(X_test.shape[0]):
    args = X_test[i].tolist()
    scores = []
    for func in evaluable_functions:
        try:
            val = float(func(*args))
            if np.isnan(val) or np.isinf(val):
                val = 0.0
                overflow_count += 1
            scores.append(val)
        except Exception:
            scores.append(0.0)
            overflow_count += 1
            
    raw_scores.append(scores)

raw_scores = np.array(raw_scores)

# Z-Score normalization per equation
mean_scores = np.mean(raw_scores, axis=0)
std_scores = np.std(raw_scores, axis=0) + 1e-8
normalized_scores = (raw_scores - mean_scores) / std_scores

y_pred = np.argmax(raw_scores, axis=1)

if overflow_count > 0:
    print(f"[Warning] Fixed {overflow_count} numerical errors automatically.")

# Performance Metrics Calculation
print("\nPERFORMANCE METRICs")
acc = accuracy_score(y_true, y_pred)
print(f"Global Accuracy: {acc:.4%} \n")
print(classification_report(y_true, y_pred, target_names=target_classes))

# Confusion Matrix Plotting
print("\nConfusion Matrix")
fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", ax=ax,
            xticklabels=target_classes, yticklabels=target_classes,
            linewidths=1, linecolor='black')

ax.set_xlabel('Symbolic Prediction (Equations)', fontweight='bold')
ax.set_ylabel('True Class (Physical)', fontweight='bold')
ax.set_title(f'Symbolic KAN Confusion Matrix (Faults Only)\nGlobal Accuracy: {acc:.2%}', fontweight='bold')

for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1.0)

plt.tight_layout()
fig_path = os.path.join(DIR_FIGS, "Fig9_Faults_Confusion_Matrix.pdf")
fig.savefig(fig_path, bbox_inches='tight')
print(f" -> [Success] Confusion matrix exported to: {fig_path}")