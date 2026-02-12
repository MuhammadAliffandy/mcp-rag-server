
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
import os
from PineBioML.visualization.style import ChartStyler

class ShapExplainer:
    """
    Wrapper for SHAP (SHapley Additive exPlanations) analysis.
    Supports summary plots, dependence plots, and force plots.
    """
    
    def __init__(self, model, X_train, model_type="tree"):
        """
        Args:
            model: Trained model (sklearn, xgboost, lightgbm, etc.)
            X_train: Background data for Explainer (pandas DataFrame)
            model_type: "tree" (RandomForest, XGBoost) or "linear" or "kernel" (generic)
        """
        self.model = model
        self.X_train = X_train
        self.model_type = model_type
        
        # Initialize Explainer
        if model_type == "tree":
            self.explainer = shap.TreeExplainer(model)
        elif model_type == "linear":
            self.explainer = shap.LinearExplainer(model, X_train)
        else:
            self.explainer = shap.KernelExplainer(model.predict, X_train)
            
    def calculate_shap_values(self, X):
        """Calculate SHAP values for given data."""
        return self.explainer.shap_values(X)

    def _get_values_matrix(self, X):
        """Helper to get the SHAP values matrix for the positive class (if binary) or regression."""
        vals = self.calculate_shap_values(X)
        if isinstance(vals, list):
            # Binary/Multiclass: default to class 1 (positive) for binary, or just first class if not
            # Ideally we'd support specifying class, but for now defaults to 1 if available
            if len(vals) > 1:
                return vals[1]
            return vals[0]
        
        # Handle 3D array from newer SHAP or specific models (samples, features, classes)
        if hasattr(vals, 'ndim') and vals.ndim == 3:
             # Default to class 1 if available, else class 0
             if vals.shape[2] > 1:
                 return vals[:, :, 1]
             return vals[:, :, 0]
             
        return vals

    def get_feature_importance(self, X):
        """Calculate global feature importance (mean absolute SHAP value)."""
        shap_matrix = self._get_values_matrix(X)
        # Ensure it's numpy array
        if not isinstance(shap_matrix, np.ndarray):
            # Try to convert if it's not (e.g. Explanation object?)
            try:
                shap_matrix = shap_matrix.values
            except:
                pass
        return np.abs(shap_matrix).mean(0)

    def summary_plot(self, X, plot_type="dot", styling=None, save_path=None):
        """
        Generate SHAP summary plot.
        
        Args:
            X: Data to explain
            plot_type: "dot" (beeswarm) or "bar"
            styling: specific styling options
        """
        shap_values = self._get_values_matrix(X)
            
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, plot_type=plot_type, show=False)
        
        if styling:
            styler = ChartStyler(styling)
            styler.apply(plt.gcf(), plt.gca())
            
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

    def dependence_plot(self, feature, X, interaction_index="auto", styling=None, save_path=None):
        """
        Generate SHAP dependence plot for a feature.
        """
        shap_values = self._get_values_matrix(X)
        
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feature, shap_values, X, interaction_index=interaction_index, show=False)
        
        if styling:
            styler = ChartStyler(styling)
            styler.apply(plt.gcf(), plt.gca())
            
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
