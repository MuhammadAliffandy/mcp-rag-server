
import pandas as pd
import numpy as np
import os
import shutil
from PineBioML.report.utils import pca_plot, pls_plot, umap_plot, corr_heatmap_plot, confusion_matrix_plot, roc_plot
from PineBioML.visualization.style import ChartStyler
import matplotlib.pyplot as plt

# Setup output directory
OUTPUT_DIR = "./tests/output/images/style_verification/"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# Generate Dummy Data
np.random.seed(42)
n_samples = 50
n_features = 10
X = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f"feat_{i}" for i in range(n_features)])
y_binary = pd.Series(np.random.choice([0, 1], n_samples), name="Target_Binary")
y_multiclass = pd.Series(np.random.choice(["A", "B", "C"], n_samples), name="Target_Multi")

# Create Publication Style
pub_style = ChartStyler.create_publication_style(theme='white', font_family='Arial', font_size=12, dpi=100) # Lower DPI for speed
print("Style Config:", pub_style)

# 1. PCA Plot
print("Testing PCA Plot...")
pca = pca_plot(prefix="Test_PCA", save_path=OUTPUT_DIR, save_fig=True, show_fig=False, styling=pub_style)
pca.make_figure(X, y_binary)

# 2. PLS-DA Plot
print("Testing PLS-DA Plot...")
pls = pls_plot(is_classification=True, prefix="Test_PLS", save_path=OUTPUT_DIR, save_fig=True, show_fig=False, styling=pub_style)
pls.make_figure(X, y_binary)

# 3. UMAP Plot
print("Testing UMAP Plot...")
try:
    umap = umap_plot(prefix="Test_UMAP", save_path=OUTPUT_DIR, save_fig=True, show_fig=False, styling=pub_style)
    umap.make_figure(X, y_binary)
except ImportError:
    print("Skipping UMAP (umap-learn not installed)")
except Exception as e:
    print(f"UMAP Failed: {e}")

# 4. Correlation Heatmap
print("Testing Correlation Heatmap...")
corr = corr_heatmap_plot(prefix="Test_Corr", save_path=OUTPUT_DIR, save_fig=True, show_fig=False, styling=pub_style)
corr.make_figure(X)

# 5. Confusion Matrix
print("Testing Confusion Matrix...")
# Mock predictions
y_true = y_binary
y_pred = np.random.choice([0, 1], n_samples)
cm = confusion_matrix_plot(prefix="Test_CM", save_path=OUTPUT_DIR, save_fig=True, show_fig=False, styling=pub_style)
cm.make_figure(y_true, pd.Series(y_pred))

# 6. ROC Plot
print("Testing ROC Plot...")
# Mock probabilities
y_prob = pd.DataFrame({0: np.random.rand(n_samples), 1: np.random.rand(n_samples)})
roc = roc_plot(pos_label=1, prefix="Test_ROC", save_path=OUTPUT_DIR, save_fig=True, show_fig=False, styling=pub_style)
roc.make_figure(y_true, y_prob)

print(f"Verification complete. Check {OUTPUT_DIR} for generated images.")
