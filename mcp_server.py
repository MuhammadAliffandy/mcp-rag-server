import os
import json
import io
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from mcp.server.fastmcp import FastMCP
from rag_processor import DocumentProcessor
from rag_engine import RAGEngine
from PineBioML.report.utils import pca_plot, corr_heatmap_plot
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Helper for debug logging (avoids stdout which breaks MCP)
def debug_log(msg):
    with open("server_debug.log", "a") as f:
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        f.write(f"[{timestamp}] {msg}\n")

# Initialize FastMCP server

# Initialize FastMCP server
mcp = FastMCP("Medical-PineBioML-Server")

# Initialize RAG Engine
rag_engine = RAGEngine()

# State directory for persistence
STATE_DIR = ".mcp_state"
TABULAR_DATA_PATH = os.path.join(STATE_DIR, "current_data.json")
INTERNAL_KNOWLEDGE_PATH = "internal_docs"
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(INTERNAL_KNOWLEDGE_PATH, exist_ok=True)

# Auto-ingest internal knowledge on startup
def auto_ingest_internal():
    if os.path.exists(INTERNAL_KNOWLEDGE_PATH):
        print(f"🚀 Auto-ingesting internal knowledge from {INTERNAL_KNOWLEDGE_PATH}...")
        docs = DocumentProcessor.load_directory(INTERNAL_KNOWLEDGE_PATH, doc_type="internal_record")
        if docs:
            rag_engine.ingest_documents(docs)
            print(f"✅ Ingested {len(docs)} internal documents.")

auto_ingest_internal()

def aggressive_clean(c):
    orig = c
    # 1. Remove common technical noise prefixes
    prefixes = [
        'data image.', 'sp mayo.', 'sp_mayo.', 'metadata.', 
        'patient.', 'clinical.', 'sum_pmayo_', 'sp_mayo_', 
        'metadata_', 'patient_', 'clinical_'
    ]
    for p in prefixes:
        if c.lower().startswith(p):
            c = c[len(p):]
    
    # 2. If it's still messy with dots (like a.b.c), take the parts that aren't noise
    if '.' in c:
        parts = [p for p in c.split('.') if p.lower() not in ['data', 'image', 'metadata', 'sp', 'mayo']]
        if parts:
            c = " ".join(parts)
    
    # 3. Clean underscores, strip, and title case
    cleaned = c.replace('_', ' ').replace('-', ' ').strip().title()
    
    # 4. Final filter: if result is just 'Image' or 'Data', but the original has more, something is wrong
    if cleaned.lower() in ['image', 'data', 'metadata'] and len(orig) > len(cleaned):
         # Try to find any word in orig that isn't noise
         words = re.findall(r'[a-zA-Z]+', orig)
         good_words = [w for w in words if w.lower() not in ['data', 'image', 'metadata', 'sp', 'mayo', 'sum', 'pmayo']]
         if good_words:
             return " ".join(good_words).title()

    return cleaned if cleaned else orig

@mcp.tool()
def ingest_medical_files(directory_path: str, doc_type: str = "internal_patient") -> str:
    """
    Ingests medical files (PDF, Word, Excel, Images) from a directory into the RAG system.
    
    Args:
        directory_path: Path to the directory containing files.
        doc_type: 'internal_patient' for records or 'external_guideline' for medical guides.
    """
    if not os.path.exists(directory_path):
        return f"Error: Directory {directory_path} not found."
    
    all_docs = DocumentProcessor.load_directory(directory_path, doc_type=doc_type)
    
    # Store the first detected tabular data for plotting tools (Persistence to Disk)
    for doc in all_docs:
        if "df_json" in doc.metadata:
            with open(TABULAR_DATA_PATH, "w") as f:
                f.write(doc.metadata["df_json"])
            break
            
    rag_engine.ingest_documents(all_docs)
    return f"Successfully ingested {len(all_docs)} segments into {doc_type}."

@mcp.tool()
def get_data_context() -> str:
    """
    Returns a summary of the currently ingested tabular data (columns, sample count, etc.).
    Use this to understand what can be plotted.
    """
    if not os.path.exists(TABULAR_DATA_PATH):
        return "No tabular data ingested yet. Please upload an Excel/CSV file."
    
    try:
        with open(TABULAR_DATA_PATH, "r") as f:
            df_json = f.read()
        df = pd.read_json(io.StringIO(df_json))
        
        info = {
            "Total Samples": len(df),
            "Columns": list(df.columns),
            "Numeric Columns": list(df.select_dtypes(include=['number']).columns),
            "Categorical Columns": list(df.select_dtypes(exclude=['number']).columns),
            "Missing Values": df.isnull().sum().to_dict()
        }
        return f"### Ingested Data Summary\n{json.dumps(info, indent=2)}"
    except Exception as e:
        return f"Error reading data: {e}"

@mcp.tool()
def smart_intent_dispatch(question: str, patient_id_filter: str = None) -> str:
    """
    Decides the user's intent (plot, clean, train, etc.) and returns a JSON list of tasks.
    """
    schema = ""
    if os.path.exists(TABULAR_DATA_PATH):
        try:
            with open(TABULAR_DATA_PATH, "r") as f:
                df_temp = pd.read_json(io.StringIO(f.read()))
                schema = ", ".join([aggressive_clean(c) for c in df_temp.columns])
        except:
            pass

    answer, tool, tasks = rag_engine.smart_query(question, patient_id_filter, schema_context=schema)
    return json.dumps({
        "answer": answer,
        "tool": tool,
        "tasks": tasks
    })

@mcp.tool()
def query_medical_rag(question: str, patient_id_filter: str = None) -> str:
    """
    Queries the medical RAG system for patient info or medical guidelines.
    """
    answer, sources = rag_engine.query(question, patient_id_filter=patient_id_filter)
    source_info = "\n\nSources explored:\n"
    for s_set in set([os.path.basename(doc.metadata.get('source', 'unknown')) for doc in sources]):
        source_info += f"- {s_set}\n"
    return f"{answer}{source_info}"

@mcp.tool()
def generate_medical_plot(plot_type: str, patient_ids: Optional[str] = None, target_column: Optional[str] = None) -> str:
    """
    Generates a PineBioML statistical plot (PCA, Heatmap, UMAP, Volcano).
    
    Args:
        plot_type: Type of plot ('pca', 'heatmap', 'umap', 'volcano', 'distribution').
        patient_ids: Optional filter (e.g., '1-5').
        target_column: For 'volcano' or colored plots, specify the categorical/label column (e.g., 'Diagnosis').
    """
    if not os.path.exists(TABULAR_DATA_PATH):
        return "Error: No tabular data ingested. Please upload Excel/CSV first."
    
    try:
        with open(TABULAR_DATA_PATH, "r") as f:
            df_json = f.read()
        df = pd.read_json(io.StringIO(df_json))
    except Exception as e:
        debug_log(f"Error loading data: {e}")
        return f"Error loading data: {e}"
    
    debug_log(f"Data loaded. Shape: {df.shape}")
    
    # Filter by IDs
    if patient_ids:
        try:
            id_col = [col for col in df.columns if 'id' in col.lower() or 'patient' in col.lower()]
            if id_col:
                ids = [i.strip() for i in patient_ids.replace('-', ',').split(',')]
                df = df[df[id_col[0]].astype(str).str.contains('|'.join(ids), na=False)]
        except Exception as e:
            return f"Filter error: {e}"

    # Map original names to cleaned names
    original_cols = df.columns.tolist()
    df.columns = [aggressive_clean(c) for c in original_cols]
    
    numeric_df_raw = df.select_dtypes(include=['number']).dropna(axis=1, how='all').dropna()
    
    # AMBIGUITY FILTER: Strict Data Sanitization
    # We explicitly BAN these terms because they are technical artifacts, metadata, or ambiguous indices.
    # The user request is: "Visualisasi akan dibaca oleh manusia" -> Clean Medical Features ONLY.
    # REVISED: 'sum pmayo' is likely a valid score, just poorly named. We unban it and will RENAME it later.
    STRICT_EXCLUDE = ['id', 'date', 'time', 'image', 'path', 'file', 'index', 'unnamed']
    
    # Apply Strict Filter
    numeric_df = numeric_df_raw[[c for c in numeric_df_raw.columns if not any(k in c.lower() for k in STRICT_EXCLUDE)]]
    
    debug_log(f"Numeric DF Shape after Ambiguity Filter: {numeric_df.shape}")

    # AUTO-SWITCH (Single Feature Case):
    if numeric_df.shape[1] == 1:
        single_col = numeric_df.columns[0]
        debug_log(f"Auto-switching to Distribution plot for single feature: {single_col}")
        plot_type = 'distribution'
        target_column = single_col 
    
    # ERROR HANDLING (Insufficient Data):
    elif numeric_df.empty or numeric_df.shape[1] < 2:
        # Generate a helpful error message explaining the Ambiguity Filter's action
        removed_cols = [c for c in numeric_df_raw.columns if c not in numeric_df.columns]
        return (f"Error: Insufficient VALID Medical Data.\n\n"
                f"I applied the **Ambiguity Filter** to ensure human-readable results.\n"
                f"- **Found Raw Columns**: {list(numeric_df_raw.columns)}\n"
                f"- **Removed Ambiguous/Metadata**: {removed_cols}\n"
                f"- **Remaining Valid Features**: {list(numeric_df.columns)}\n\n"
                f"We need at least 2 valid medical features for a heatmap. Please check your data source.")


    import time
    timestamp = int(time.time())
    plot_path = f"plots/{plot_type}_{timestamp}.png"
    os.makedirs("plots", exist_ok=True)
    
    plt.close('all')
    interpretation = ""
    
    def format_for_display(text: str) -> str:
        if not text: return "General Analysis"
        
        # SMART ALIASING: Rename technical columns to human-readable medical labels
        aliases = {
            "sum pmayo": "P-Mayo Total Score",
            "pmayo": "P-Mayo Score",
            "avg_intensity": "Average Intensity",
            "diagnosis_code": "Diagnosis Category"
        }
        
        lower_text = text.lower()
        # Direct match check
        if lower_text in aliases:
            return aliases[lower_text]
        # Partial match check (slower but covers variations)
        for key, val in aliases.items():
            if key in lower_text:
                return val
                
        return text.replace('_', ' ').replace('.', ' ').title()

    debug_log(f"Plotting {plot_type} for target: {target_column}")
    display_target = format_for_display(target_column) # Safe formatted string for titles
    
    try:
        # 1. Target Column discovery
        target_series = None
        if target_column:
            # Try exact match on cleaned name
            if target_column in df.columns:
                target_series = df[target_column]
            else:
                # Try case-insensitive fuzzy match
                matches = [c for c in df.columns if target_column.lower() in c.lower()]
                if matches:
                    target_column = matches[0]
                    target_series = df[target_column]
                else:
                    # Search original names just in case
                    orig_matches = [c for c in original_cols if target_column.lower() in c.lower()]
                    if orig_matches:
                        target_column = aggressive_clean(orig_matches[0])
                        target_series = df[target_column]

        if target_series is None:
            debug_log("Target series not found yet. Trying categorical fallback.")
            # Fallback: categorical search
            potential = [c for c in df.columns if df[c].nunique() <= 5 and not pd.api.types.is_numeric_dtype(df[c])]
            if potential:
                target_column = potential[0]
                target_series = df[target_column]
            else:
                debug_log("Attributes: No categorical found. Creating General Sample.")
                target_column = "General Sample"
                df["General Sample"] = "Sample" # Create the column so it exists!
                target_series = df["General Sample"]
        
        debug_log(f"Final Target Column: {target_column}")
        display_target = format_for_display(target_column) # Update display label with final resolved target

        import seaborn as sns
        # 2. Plot Generation
        if plot_type.lower() == 'pca':
            from PineBioML.report.utils import pca_plot
            if numeric_df.shape[1] < 2 or numeric_df.shape[0] < 2:
                return "Error: PCA requires at least 2 markers and 2 samples."
            
            pp = pca_plot(n_pc=2, show_fig=False, save_fig=False)
            pp.draw(numeric_df, y=target_series)
            plt.title(f"PCA: Biological Grouping by {target_column}\n(N={len(df)} samples, {len(numeric_df.columns)} medical markers)")
            plt.xlabel("Principal Component 1 (PC1)")
            plt.ylabel("Principal Component 2 (PC2)")
            
            # Interpretation
            from sklearn.decomposition import PCA as SkPCA
            from sklearn.preprocessing import StandardScaler
            scaled = StandardScaler().fit_transform(numeric_df)
            sk_pca = SkPCA(n_components=2).fit(scaled)
            top_markers = pd.Series(abs(sk_pca.components_[0]), index=numeric_df.columns).sort_values(ascending=False).head(3).index.tolist()
            interpretation = f"### PCA (Principal Component Analysis)\n- **Variance**: PC1 ({sk_pca.explained_variance_ratio_[0]:.1%}), PC2 ({sk_pca.explained_variance_ratio_[1]:.1%})\n- **Top Contributors**: {', '.join(top_markers)}\n\n*This plot shows how patients cluster based on their overall medical profile.*"
            plt.gcf().savefig(plot_path, bbox_inches='tight')

        elif plot_type.lower() == 'umap':
            from PineBioML.report.utils import umap_plot
            plt.figure(figsize=(10, 6))
            up = umap_plot(show_fig=False, save_fig=False)
            up.draw(numeric_df, y=target_series)
            plt.title(f"UMAP: Non-linear Clustering ({target_column or 'All Data'})\n(Complex patterns in high-dimensional data)")
            plt.xlabel("UMAP Dimension 1")
            plt.ylabel("UMAP Dimension 2")
            interpretation = "### UMAP Visualization\nDisplays non-linear clusters. Points closer together have more similar biological profiles."
            plt.savefig(plot_path, bbox_inches='tight')

        elif plot_type.lower() == 'volcano':
            if target_series is None or target_series.dropna().nunique() != 2:
                binary = [c for c in df.columns if df[c].dropna().nunique() == 2]
                if binary:
                    target_column = binary[0]
                    target_series = df[target_column]
                else:
                    return "Error: Volcano plot needs exactly 2 groups (e.g. Case vs Control)."

            v_numeric = numeric_df.loc[:, numeric_df.std() > 0]
            if v_numeric.empty:
                return "Error: Not enough variation in markers to perform statistical test."

            target_series_cleaned = target_series.dropna()
            counts = target_series_cleaned.value_counts()
            if counts.min() < 2:
                return f"Error: Both groups in '{target_column}' must have at least 2 samples."

            from PineBioML.selection.Volcano import Volcano_selection
            valid_idx = target_series_cleaned.index
            y_v = target_series_cleaned
            x_v = v_numeric.reindex(index=valid_idx).dropna()
            y_v = y_v.loc[x_v.index]
            
            groups = y_v.unique()
            k_val = int(max(1, min(10, x_v.shape[1])))
            
            v_sel = Volcano_selection(k=k_val, target_label=groups[0])
            scores = v_sel.Scoring(x_v, y_v)
            v_sel.scores = scores
            v_sel.selected_score = v_sel.Select(scores)
            
            plt.figure(figsize=(10, 8))
            v_sel.plotting(show=False, saving=False)
            plt.title(f"Volcano Plot: Differential Expression ({groups[0]} vs {groups[1]})\n(N={len(y_v)} total samples)")
            plt.xlabel("Log2 Fold Change (Difference)")
            plt.ylabel("-Log10 P-value (Significance)")
            plt.savefig(plot_path, bbox_inches='tight')
            interpretation = f"### Volcano Plot: {target_column}\nComparing group **{groups[0]}** vs **{groups[1]}**.\nMarkers in the upper corners are biologically significant."

        elif plot_type.lower() == 'plsda':
            if target_series is None:
                return "Error: PLS-DA requires a target_column."
            from PineBioML.report.utils import pls_plot
            is_clf = not pd.api.types.is_numeric_dtype(target_series)
            pp = pls_plot(is_classification=is_clf, show_fig=False, save_fig=False)
            pp.draw(numeric_df, target_series)
            plt.title(f"PLS-DA: Supervised Separation by {target_column}")
            plt.gcf().savefig(plot_path, bbox_inches='tight')
            interpretation = f"### PLS-DA\nFocuses on the markers that best separate the labels in '{target_column}'."

        elif plot_type.lower() == 'distribution':
            if not target_column or target_column not in df.columns:
                return "Error: Distribution plot requires a valid target_column (e.g., 'Age')."
            
            plt.figure(figsize=(10, 6))
            is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
            
            if is_numeric:
                sns.histplot(df[target_column].dropna(), kde=True, color='teal')
                plt.title(f"Distribution Analysis: {display_target}")
                plt.xlabel(target_column)
                plt.ylabel("Frequency / Count")
                interpretation = f"### Distribution Analysis: {target_column}\nThis histogram shows the spread of **{target_column}**. The curve (KDE) represents the estimated density."
            else:
                counts = df[target_column].value_counts()
                sns.barplot(x=counts.index, y=counts.values, palette='viridis')
                plt.title(f"Breakdown of {target_column}\n(Frequency count of categorical groups)")
                plt.xlabel(target_column)
                plt.ylabel("Total Count")
                
                # Statistical Summary
                stats = "\n".join([f"- **{k}**: {v}" for k, v in counts.head(5).items()])
                interpretation = f"### Stats: {target_column}\n{stats}"
            
            plt.savefig(plot_path, bbox_inches='tight')

        else: # Heatmap
            # Direct Seaborn for better annotations
            corr = numeric_df.corr()
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", center=0)
            plt.title(f"Correlation Map: Relationship between Markers\nFocus: {display_target}")
            plt.savefig(plot_path, bbox_inches='tight')
            
            # Find unique strongest correlations (Upper Triangle only to avoid A-B / B-A duplicates)
            import numpy as np
            # Create mask for upper triangle (excluding diagonal k=1)
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            
            # Apply mask to keep only upper triangle values
            corr_upper = corr.where(mask)
            
            # Stack and sort by absolute correlation
            s = corr_upper.unstack().dropna()
            
            # Sort by strength (absolute value) but keep sign
            # We want to see the strongest relationships (positive or negative)
            so = s.iloc[s.abs().argsort()[::-1]]
            
            # User request: "Can I see ALL data?"
            # Strategy: If it's a reasonable size (<= 100 pairs), show EVERYTHING.
            # If it's huge, show Top 50 to prevent LLM overload.
            total_pairs = len(so)
            limit = 100 if total_pairs <= 100 else 50
            
            top_corr = so.head(limit)
            
            stats = "\n".join([f"- **{i[1]} vs {i[0]}**: {v:.2f}" for i, v in top_corr.items()])
            interpretation = f"### Correlation Analysis (Unique Pairs)\n{stats}"
            
            if total_pairs > limit:
                interpretation += f"\n\n*(Displaying top {limit} of {total_pairs} pairs to avoid text overflow)*"
            else:
                interpretation += f"\n\n*(Showing ALL {total_pairs} unique correlations)*"

        plt.close('all')
        return f"{plot_path}|||{interpretation}"

        plt.close('all')
        return f"{plot_path}|||{interpretation}"
    except Exception as e:
        plt.close('all')
        debug_log(f"Plotting CRASH: {e}")
        import traceback
        debug_log(traceback.format_exc())
        return f"Plotting failed: {str(e)}"

@mcp.tool()
def clean_medical_data(impute_strategy: str = "median", outlier_method: str = "isolation_forest") -> str:
    """
    Cleans data: Imputes missing values and removes outliers (PineBioML).
    """
    if not os.path.exists(TABULAR_DATA_PATH): return "Error: No data."
    try:
        with open(TABULAR_DATA_PATH, "r") as f:
            df = pd.read_json(io.StringIO(f.read()))
        numeric_cols = df.select_dtypes(include=['number']).columns
        from PineBioML.preprocessing.impute import simple_imputer, knn_imputer
        imp = knn_imputer() if impute_strategy == "knn" else simple_imputer(strategy=impute_strategy)
        cleaned_numeric = imp.fit_transform(df[numeric_cols])
        if outlier_method == "isolation_forest":
            from PineBioML.preprocessing.outlier import IsolationForest
            iso = IsolationForest()
            cleaned_numeric = iso.fit_transform(cleaned_numeric)
        df_cleaned = df.loc[cleaned_numeric.index].copy()
        df_cleaned[numeric_cols] = cleaned_numeric
        with open(TABULAR_DATA_PATH, "w") as f:
            f.write(df_cleaned.to_json())
        return f"Cleaned successfully. Samples remaining: {len(df_cleaned)}."
    except Exception as e:
        return f"Cleaning failed: {e}"

@mcp.tool()
def predict_patient_outcome(input_data_json: str) -> str:
    """
    Predicts outcome (Class/Reg) for a new patient using the trained model.
    """
    model_path = os.path.join(STATE_DIR, "best_model.joblib")
    if not os.path.exists(model_path): return "Error: No model trained."
    try:
        import joblib
        tuner = joblib.load(model_path)
        input_data = json.loads(input_data_json)
        df_new = pd.DataFrame([input_data]).reindex(columns=tuner.x.columns, fill_value=0)
        prob = tuner.predict_proba(df_new) if not tuner.is_regression() else None
        pred = tuner.best_model.predict(df_new)
        res = f"### Prediction Result\n- **Outcome**: {pred[0]}"
        if prob is not None: res += f"\n- **Confidence**: {prob.max(axis=1).values[0]:.2%}"
        return res
    except Exception as e:
        return f"Prediction failed: {e}"

@mcp.tool()
def train_medical_model(target_column: Optional[str] = None, model_type: str = "random_forest") -> str:
    """
    Automatically detects task (Classification or Regression) and trains a PineBioML pipeline.
    """
    if not os.path.exists(TABULAR_DATA_PATH):
        return "Error: No data ingested."
    
    try:
        with open(TABULAR_DATA_PATH, "r") as f:
            df = pd.read_json(io.StringIO(f.read()))
            
        if not target_column or target_column not in df.columns:
            # Fallback to Diagnosis or any categorical-looking column
            categorical = [c for c in df.columns if df[c].nunique() <= 5 and not pd.api.types.is_numeric_dtype(df[c])]
            if categorical:
                target_column = categorical[0]
            else:
                return f"Error: target_column '{target_column}' not found and no categorical fallback available."
            
        from PineBioML.model.utils import Pine
        y = df[target_column].dropna()
        x = df.loc[y.index].select_dtypes(include=['number']).drop(columns=[target_column], errors='ignore').dropna()
        y = y.loc[x.index]

        # Auto Detect Task
        is_regression = pd.api.types.is_numeric_dtype(y) and y.nunique() > 10
        
        if is_regression:
            from PineBioML.model.supervised.Regression import RandomForest_tuner as RFR, XGBoost_tuner as XGBR, SVM_tuner as SVR
            if model_type == "xgboost": tuner = XGBR(n_try=10)
            elif model_type == "svm": tuner = SVR(n_try=10)
            else: tuner = RFR(n_try=10)
        else:
            from PineBioML.model.supervised.Classification import RandomForest_tuner, XGBoost_tuner, SVM_tuner
            if model_type == "xgboost": tuner = XGBoost_tuner(n_try=10)
            elif model_type == "svm": tuner = SVM_tuner(n_try=10)
            else: tuner = RandomForest_tuner(n_try=10)
            
        experiment = [('tune', {'best': tuner})]
        pine = Pine(experiment=experiment, cv_result=True)
        results = pine.do_experiment(x, y)
        
        best_model_path = os.path.join(STATE_DIR, "best_model.joblib")
        import joblib
        joblib.dump(tuner, best_model_path)
        
        task_name = "Regression" if is_regression else "Classification"
        metrics = results.iloc[0].to_dict()
        metric_str = "\n".join([f"- **{k}**: {v:.4f}" if isinstance(v, float) else f"- **{k}**: {v}" for k, v in metrics.items() if "train_" in k or "cv_" in k])
        
        return f"### {task_name} Training Results ({model_type})\n{metric_str}\n\nModel saved successfully."
    except Exception as e:
        return f"Training failed: {e}"

@mcp.tool()
def discover_markers(target_column: Optional[str] = None) -> str:
    """
    Uses Ensemble Selection (Lasso, SVM, RF) to identify the most robust biological markers.
    """
    if not os.path.exists(TABULAR_DATA_PATH):
        return "Error: No data ingested."
    try:
        with open(TABULAR_DATA_PATH, "r") as f:
            df = pd.read_json(io.StringIO(f.read()))
        
        if not target_column or target_column not in df.columns:
            categorical = [c for c in df.columns if df[c].nunique() <= 5 and not pd.api.types.is_numeric_dtype(df[c])]
            target_column = categorical[0] if categorical else df.columns[-1]
        
        y = df[target_column].dropna()
        x = df.loc[y.index].select_dtypes(include=['number']).drop(columns=[target_column], errors='ignore').dropna()
        y = y.loc[x.index]

        from PineBioML.selection.regression import ensemble_selector
        selector = ensemble_selector(k=min(10, x.shape[1]))
        selector.fit(x, y)
        top = selector.selected_score.sort_values(ascending=False).head(10)
        
        res = f"### Ensemble Marker Discovery for {target_column}\n"
        res += "*(Combining insights from Lasso, SVM, and Random Forest for maximum reliability)*\n\n"
        for marker, score in top.items():
            res += f"- **{marker}**: {score:.4f} (Robustness Score)\n"
        return res
    except Exception as e:
        return f"Discovery failed: {e}"

@mcp.tool()
def synthesize_medical_results(question: str, results: str) -> str:
    """
    Synthesizes multiple tool outputs into a final clinical summary.
    """
    return rag_engine.synthesize_results(question, results)

@mcp.tool()
def generate_medical_report(target_column: str = None) -> str:
    """
    Generates a comprehensive Bio-ML report with statistical summaries and key plots.
    """
    if not os.path.exists(TABULAR_DATA_PATH): return "Error: No data."
    try:
        with open(TABULAR_DATA_PATH, "r") as f:
            df = pd.read_json(io.StringIO(f.read()))
        
        # Summary for data overall
        numeric_df = df.select_dtypes(include=['number']).dropna()
        from PineBioML.report.utils import data_overview
        plot_path = "plots/comprehensive_report.png"
        data_overview(numeric_df, save_path="plots/", prefix="report", show_fig=False, save_fig=True)
        
        res = "### Comprehensive Medical Data Report\n"
        res += f"- **Samples Analysed**: {len(df)}\n"
        res += f"- **Numeric Markers**: {len(numeric_df.columns)}\n"
        res += "\n#### Statistical Overview Generated.\n*(Includes PCA, PLS-DA, UMAP, and Correlation Heatmap)*"
        return f"plots/report data_overview.png|||{res}"
    except Exception as e:
        return f"Report generation failed: {e}"

@mcp.tool()
def reset_medical_database() -> str:
    """
    Clears all ingested data (RAG database and tabular state) for a fresh start.
    """
    import shutil
    try:
        # Clear RAG persistence
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        
        # Clear Tabular persistence
        if os.path.exists(STATE_DIR):
            shutil.rmtree(STATE_DIR)
            os.makedirs(STATE_DIR)
            
        # Clear temp uploads
        if os.path.exists("temp_uploads"):
            shutil.rmtree("temp_uploads")
            
        # Clear Plots
        if os.path.exists("plots"):
            shutil.rmtree("plots")
            os.makedirs("plots")
            
        return "Database and cache cleared successfully. You can now ingest new files."
    except Exception as e:
        return f"Error during reset: {e}"

if __name__ == "__main__":
    mcp.run()
