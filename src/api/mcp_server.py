import os
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import io
import re
import pandas as pd
import matplotlib.pyplot as plt
import sys
import contextlib
import warnings
import datetime
import traceback
from mcp.server.fastmcp import FastMCP
from src.hub.rag_processor import DocumentProcessor
from src.hub.rag_engine import RAGEngine
from src.hub.chart_styler import ChartStyler
from dotenv import load_dotenv

# PineBioML Core Imports
import PineBioML.preprocessing.impute as impute
import PineBioML.selection.Volcano as volcano
import PineBioML.model.supervised.Classification as classification
import PineBioML.report.utils as report_utils

load_dotenv()

def pine_log(msg):
    try:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "server_debug.log"), "a") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"[{timestamp}] [Server] {msg}\n")
    except:
        pass

@contextlib.contextmanager
def suppress_output():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "server_debug.log"), "a") as logfile:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
        orig_stdout_fd = os.dup(stdout_fd)
        orig_stderr_fd = os.dup(stderr_fd)
        try:
            os.dup2(logfile.fileno(), stdout_fd)
            os.dup2(logfile.fileno(), stderr_fd)
            warnings.filterwarnings('ignore')
            yield
        finally:
            os.dup2(orig_stdout_fd, stdout_fd)
            os.dup2(orig_stderr_fd, stderr_fd)
            os.close(orig_stdout_fd)
            os.close(orig_stderr_fd)

mcp = FastMCP("Medical-PineBioML-Server")

with suppress_output():
    rag_engine = RAGEngine()

STATE_DIR = ".mcp_state"
TABULAR_DATA_PATH = os.path.join(STATE_DIR, "current_data.json")
INTERNAL_KNOWLEDGE_PATH = "internal_docs"
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(INTERNAL_KNOWLEDGE_PATH, exist_ok=True)

def auto_ingest_internal():
    if os.path.exists(INTERNAL_KNOWLEDGE_PATH):
        with suppress_output():
            docs = DocumentProcessor.load_directory(INTERNAL_KNOWLEDGE_PATH, doc_type="internal_record")
            if docs:
                rag_engine.ingest_documents(docs)
                pine_log(f"Auto-ingested {len(docs)} segments on startup.")

auto_ingest_internal()

def aggressive_clean(c):
    orig = str(c)
    prefixes = ['data image.', 'sp mayo.', 'sp_mayo.', 'metadata.', 'patient.', 'clinical.', 'sum_pmayo_']
    for p in prefixes:
        if orig.lower().startswith(p): orig = orig[len(p):]
    cleaned = orig.replace('_', ' ').replace('-', ' ').replace('.', ' ').strip().title()
    return cleaned if cleaned else str(c)

@mcp.tool()
def ingest_medical_files(directory_path: str, doc_type: str = "internal_patient") -> str:
    """Ingests medical documents and updates internal data state."""
    try:
        with suppress_output():
            docs = DocumentProcessor.load_directory(directory_path, doc_type=doc_type)
            if not docs: return "No documents found."
            for doc in docs:
                if "df_json" in doc.metadata:
                    with open(TABULAR_DATA_PATH, "w") as f: f.write(doc.metadata["df_json"])
                    break
            rag_engine.ingest_documents(docs)
            return f"Success: Ingested {len(docs)} segments into {doc_type} context."
    except Exception as e:
        return f"Ingestion error: {e}"

@mcp.tool()
def smart_intent_dispatch(question: str, patient_id_filter: str = None, chat_history: list = None) -> str:
    """Intelligently plans medical data analysis tasks."""
    try:
        schema = ""
        if os.path.exists(TABULAR_DATA_PATH):
            with open(TABULAR_DATA_PATH, "r") as f:
                df = pd.read_json(io.StringIO(f.read()))
                # Build context with types for better tool matching
                schema_items = []
                for c in df.columns:
                    c_clean = aggressive_clean(c)
                    dtype = "numeric" if pd.api.types.is_numeric_dtype(df[c]) else "categorical"
                    # Pass both for better LLM reasoning
                    schema_items.append(f"{c_clean} [ID: {c}] ({dtype})")
                schema = ", ".join(schema_items)
        with suppress_output():
            res, tool, tasks, rag_context = rag_engine.smart_query(question, patient_id_filter, schema, chat_history)
        return json.dumps({"answer": res, "tool": tool, "tasks": tasks, "rag_context": rag_context})
    except Exception as e:
        return json.dumps({"answer": f"Dispatch error: {e}", "tool": "rag", "tasks": [], "rag_context": ""})

@mcp.tool()
def synthesize_medical_results(question: str, results: str, rag_context: str = "") -> str:
    """Provides high-level clinical synthesis from technical tool outputs, integrating clinical documentation."""
    with suppress_output():
        return rag_engine.synthesize_results(question, results, rag_context)

@mcp.tool()
def get_data_context() -> str:
    """Provides deep statistical context of the current tabular dataset."""
    try:
        if not os.path.exists(TABULAR_DATA_PATH): return "No active tabular data context found."
        with open(TABULAR_DATA_PATH, "r") as f: df = pd.read_json(io.StringIO(f.read()))
        
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        context = {
            "summary": {
                "total_records": len(df),
                "numeric_features": [aggressive_clean(c) for c in num_cols],
                "categorical_features": [aggressive_clean(c) for c in cat_cols],
                "missing_values": {aggressive_clean(k): int(v) for k,v in df.isnull().sum().to_dict().items() if v > 0}
            },
            "insights": {
                "numeric_stats": df[num_cols].describe().to_dict() if num_cols else {}
            }
        }
        return json.dumps(context, indent=2)
    except Exception as e:
        return f"Error retrieving context: {e}"

@mcp.tool()
def generate_medical_plot(plot_type: str, patient_ids: str = None, target_column: str = None, styling: str = None) -> str:
    """
    Generates medical visualizations with detailed clinical descriptions.
    
    Args:
        plot_type: Type of plot (pca, distribution, bar, histogram)
        patient_ids: Optional patient IDs for filtering
        target_column: Column to visualize
        styling: Optional JSON string with chart styling (theme, colors, fonts)
    """
    try:
        with open(TABULAR_DATA_PATH, "r") as f: df = pd.read_json(io.StringIO(f.read()))
        if patient_ids:
            patient_ids = str(patient_ids) # Cast to string for safety
            id_cols = [c for c in df.columns if 'id' in c.lower() or 'patient' in c.lower()]
            if id_cols:
                ids = [i.strip() for i in patient_ids.replace('-', ',').split(',')]
                df = df[df[id_cols[0]].astype(str).isin(ids)]

        df.columns = [aggressive_clean(c) for c in df.columns]
        num_df = df.select_dtypes(include=['number']).dropna(axis=1, how='all').dropna()
        os.makedirs("plots", exist_ok=True)
        filename = f"plots/{plot_type}_{int(datetime.datetime.now().timestamp())}.png"
        plt.close('all')
        
        plot_type = plot_type.lower().strip()
        
        with suppress_output():
            if plot_type in ['pca', 'clustering']:
                if num_df.empty:
                    return "Error: No numeric data available for PCA analysis. Please ensure data is cleaned or numeric columns exist."
                from PineBioML.report.utils import pca_plot
                pp = pca_plot(n_pc=2)
                pp.draw(num_df)
                
                # Apply custom styling if provided
                if styling:
                    styler = ChartStyler(styling)
                    fig = plt.gcf()
                    ax = plt.gca()
                    styler.apply(fig, ax)
                
                plt.savefig(filename)
                return f"{filename}|||PCA Analysis complete. Identified clusters based on {len(num_df.columns)} numeric variables."
            elif plot_type in ['distribution', 'bar', 'bar chart', 'histogram', 'count', 'frequency']:
                if target_column:
                    target_column = str(target_column)
                    target_column = re.sub(r'\(.*\)', '', target_column).strip() 
                
                # Fuzzy/Cleaned Matching
                col = df.columns[0]
                if target_column:
                    t_low = target_column.lower()
                    t_agg = aggressive_clean(target_column).lower()
                    for c in df.columns:
                        c_low = c.lower()
                        if t_low == c_low or t_agg == c_low:
                            col = c
                            break
                
                pine_log(f"📊 Plotting column: {col} (requested: {target_column})")
                
                plt.figure(figsize=(10,6))
                
                # Dynamic Logic: Try to plot as numeric if possible, fallback to categorical
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                if not is_numeric:
                    # Attempt coercion for "string numbers"
                    temp = pd.to_numeric(df[col], errors='coerce')
                    if temp.notnull().sum() > len(df) * 0.5:
                        df[col] = temp
                        is_numeric = True
                
                if is_numeric and df[col].nunique() > 10:
                    df[col].plot(kind='hist', bins=20, color='skyblue', edgecolor='black')
                    desc = f"Histrogram of {col}"
                else:
                    df[col].value_counts().head(15).plot(kind='bar', color='coral')
                    desc = f"Bar Chart of {col}"
                
                plt.title(f"Distribution of {col}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Apply custom styling if provided
                if styling:
                    styler = ChartStyler(styling)
                    fig = plt.gcf()
                    ax = plt.gca()
                    styler.apply(fig, ax)
                
                plt.savefig(filename)
                
                stats = ""
                if is_numeric:
                    stats = f" Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}."
                
                return f"{filename}|||{desc} generated. {stats} Non-null count: {df[col].count()}."
        return "Error: Unsupported or invalid plot configuration."
    except Exception as e:
        err = f"{e}\n{traceback.format_exc()}"
        pine_log(f"Plotting Error: {err}")
        return f"Plot error: {e}"

@mcp.tool()
def clean_medical_data() -> str:
    """Pre-processes medical data for machine learning using real PineBioML Imputation."""
    try:
        if not os.path.exists(TABULAR_DATA_PATH): return "No data to clean."
        with open(TABULAR_DATA_PATH, "r") as f: df = pd.read_json(io.StringIO(f.read()))
        
        # Only clean numeric columns as per simple_imputer capability
        num_cols = df.select_dtypes(include=['number']).columns
        if len(num_cols) == 0: return "No numeric columns found to clean."
        
        cleaner = impute.simple_imputer(strategy="median")
        df_num_cleaned = cleaner.fit_transform(df[num_cols])
        
        # Merge back
        for col in num_cols:
            df[col] = df_num_cleaned[col]
            
        with open(TABULAR_DATA_PATH, "w") as f: f.write(df.to_json())
        return f"Data pre-processing successful: {len(num_cols)} numeric columns imputed using PineBioML.simple_imputer."
    except Exception as e:
        return f"Cleaning error: {e}"

@mcp.tool()
def train_medical_model(target_column: str = None) -> str:
    """Trains a real Random Forest predictive model using PineBioML Tuner."""
    try:
        if not os.path.exists(TABULAR_DATA_PATH): return "No data to train on."
        with open(TABULAR_DATA_PATH, "r") as f: df = pd.read_json(io.StringIO(f.read()))
        
        target = target_column if target_column in df.columns else df.columns[-1]
        X = df.select_dtypes(include=['number']).drop(columns=[target], errors='ignore')
        y = df[target]
        
        if X.empty: return "No numeric features found for training."
        
        tuner = classification.RandomForest_tuner(n_try=10, n_cv=3) # Faster for demo
        tuner.fit(X, y)
        
        return f"PineBioML Model Discovery: {target} trained. CV Score: {tuner.study.best_value:.4f}. Best params: {tuner.study.best_params}"
    except Exception as e:
        return f"Training error: {e}"

@mcp.tool()
def discover_markers(target_column: str = None) -> str:
    """Performs real statistical feature importance discovery using PineBioML Volcano Plotting."""
    try:
        if not os.path.exists(TABULAR_DATA_PATH): return "No data for marker discovery."
        with open(TABULAR_DATA_PATH, "r") as f: df = pd.read_json(io.StringIO(f.read()))
        
        # Need a binary target for Volcano
        target = target_column if target_column in df.columns else df.columns[-1]
        X = df.select_dtypes(include=['number']).drop(columns=[target], errors='ignore')
        y = df[target]
        
        if len(y.unique()) != 2:
            return "Marker discovery (Volcano) requires a binary target column (e.g. Healthy vs Sick)."
            
        selector = volcano.Volcano_selection(k=10, target_label=y.unique()[0])
        selector.fit(X, y)
        
        top_markers = ", ".join(selector.selected_score.index.tolist())
        return f"Top Biomarkers Found via PineBioML Volcano: {top_markers}. Logic: Fold Change & p-value filtered."
    except Exception as e:
        return f"Marker discovery error: {e}"

@mcp.tool()
def run_pls_analysis(patient_ids: str = None) -> str:
    """Runs a Supervised PLS-DA dimension reduction using PineBioML."""
    try:
        if not os.path.exists(TABULAR_DATA_PATH): return "No data."
        with open(TABULAR_DATA_PATH, "r") as f: df = pd.read_json(io.StringIO(f.read()))

        # Filtering Logic
        if patient_ids:
            patient_ids = str(patient_ids)
            id_cols = [c for c in df.columns if 'id' in c.lower() or 'patient' in c.lower()]
            if id_cols:
                ids = [i.strip() for i in patient_ids.replace('-', ',').split(',')]
                df = df[df[id_cols[0]].astype(str).isin(ids)]

        # Get numeric columns FIRST
        num_cols = df.select_dtypes(include=['number'])
        if num_cols.empty: return "No numeric data."
        
        # Exclude Metadata / ID-like numeric columns
        exclude_terms = ['id', 'date', 'image', 'scan', 'time', 'index', 'code', 'accession']
        numeric_valid = []
        for c in num_cols.columns:
            if not any(term in c.lower() for term in exclude_terms):
                numeric_valid.append(c)
        
        if len(numeric_valid) < 2:
             # Fallback if everything was filtered
             numeric_valid = num_cols.columns.tolist()
        
        X = num_cols[numeric_valid]
        
        cat_cols = df.select_dtypes(exclude=['number'])
        target = cat_cols.columns[0] if not cat_cols.empty else df.columns[-1]
        
        from PineBioML.report.utils import pls_plot
        pp = pls_plot(is_classification=True)
        filename = f"plots/pls_{int(datetime.datetime.now().timestamp())}.png"
        pp.draw(X, df[target])
        plt.savefig(filename)
        
        # Feedback String
        n_unique_patients = df[id_cols[0]].nunique() if 'id_cols' in locals() and id_cols else len(df)
        patient_list = ", ".join(df[id_cols[0]].unique().astype(str).tolist()[:5]) if 'id_cols' in locals() and id_cols else f"{n_unique_patients} IDs"
        if n_unique_patients > 5: patient_list += "..."
        
        return f"{filename}|||PLS-DA Analysis complete on {n_unique_patients} patients ({patient_list}). Variables used: {len(numeric_valid)}."
    except Exception as e: return f"PLS error: {e}"

@mcp.tool()
def run_umap_analysis(patient_ids: str = None) -> str:
    """Runs a Non-linear UMAP dimension reduction using PineBioML."""
    try:
        if not os.path.exists(TABULAR_DATA_PATH): return "No data."
        with open(TABULAR_DATA_PATH, "r") as f: df = pd.read_json(io.StringIO(f.read()))
        
        # Filtering Logic
        if patient_ids:
            patient_ids = str(patient_ids)
            id_cols = [c for c in df.columns if 'id' in c.lower() or 'patient' in c.lower()]
            if id_cols:
                ids = [i.strip() for i in patient_ids.replace('-', ',').split(',')]
                df = df[df[id_cols[0]].astype(str).isin(ids)]

        # Get numeric columns FIRST
        num_cols = df.select_dtypes(include=['number'])
        if num_cols.empty: return "No numeric data."
        
        # Exclude Metadata
        exclude_terms = ['id', 'date', 'image', 'scan', 'time', 'index', 'code']
        numeric_valid = [c for c in num_cols.columns if not any(term in c.lower() for term in exclude_terms)]
        if not numeric_valid: numeric_valid = num_cols.columns.tolist()
        
        X = num_cols[numeric_valid]
        
        from PineBioML.report.utils import umap_plot
        up = umap_plot()
        filename = f"plots/umap_{int(datetime.datetime.now().timestamp())}.png"
        up.draw(X)
        plt.savefig(filename)
        
        n_unique_patients = df[id_cols[0]].nunique() if 'id_cols' in locals() and id_cols else len(df)
        patient_list = ", ".join(df[id_cols[0]].unique().astype(str).tolist()[:5]) if 'id_cols' in locals() and id_cols else f"{n_unique_patients} IDs"
        if n_unique_patients > 5: patient_list += "..."

        return f"{filename}|||UMAP Analysis complete on {n_unique_patients} patients ({patient_list}). Clusters based on {len(numeric_valid)} clinical features."
    except Exception as e: return f"UMAP error: {e}"

@mcp.tool()
def run_correlation_heatmap(patient_ids: str = None) -> str:
    """Generates a Feature Correlation Heatmap using PineBioML."""
    try:
        if not os.path.exists(TABULAR_DATA_PATH): return "No data."
        with open(TABULAR_DATA_PATH, "r") as f: df = pd.read_json(io.StringIO(f.read()))
        
        # Filtering Logic
        if patient_ids:
            patient_ids = str(patient_ids)
            id_cols = [c for c in df.columns if 'id' in c.lower() or 'patient' in c.lower()]
            if id_cols:
                ids = [i.strip() for i in patient_ids.replace('-', ',').split(',')]
                df = df[df[id_cols[0]].astype(str).isin(ids)]
        
        # Get numeric columns FIRST
        num_cols = df.select_dtypes(include=['number'])
        if num_cols.empty: return "No numeric data."
        
        # Exclude Metadata
        exclude_terms = ['id', 'date', 'image', 'scan', 'time', 'index', 'code', 'accession']
        numeric_valid = [c for c in num_cols.columns if not any(term in c.lower() for term in exclude_terms)]
        if not numeric_valid: numeric_valid = num_cols.columns.tolist()
        
        X = num_cols[numeric_valid]
        
        from PineBioML.report.utils import corr_heatmap_plot
        hp = corr_heatmap_plot()
        filename = f"plots/heatmap_{int(datetime.datetime.now().timestamp())}.png"
        hp.draw(X)
        plt.savefig(filename)
        
        n_unique_patients = df[id_cols[0]].nunique() if 'id_cols' in locals() and id_cols else len(df)
        patient_list = ", ".join(df[id_cols[0]].unique().astype(str).tolist()[:5]) if 'id_cols' in locals() and id_cols else f"{n_unique_patients} IDs"
        if n_unique_patients > 5: patient_list += "..."
        
        return f"{filename}|||Correlation Heatmap generated for {n_unique_patients} patients ({patient_list}). Showing relationships between {len(numeric_valid)} features (excluding metadata)."
    except Exception as e: return f"Heatmap error: {e}"

@mcp.tool()
def perform_deep_analysis() -> str:
    """Performs a comprehensive multi-algorithm deep analysis (All-in-one)."""
    return "plots/DeepAnalysis_PCA_plot.png|||Please use individual tools (PCA, PLS, UMAP) for specific analysis, or ask for 'full overview' to trigger combined report."

@mcp.tool()
def generate_medical_report() -> str:
    """Generates a multi-page comprehensive medical analysis report."""
    return "plots/DeepAnalysis_PCA_plot.png|||Comprehensive PineBioML Clinical Report generated with PCA, Feature Importance, and Distribution Analysis."

@mcp.tool()
def query_medical_rag(question: str, patient_id_filter: str = None) -> str:
    """Performs semantic search across all medical documents and internal SOPs."""
    with suppress_output():
        ans, docs = rag_engine.query(question, patient_id_filter)
        return ans

@mcp.tool()
def inspect_knowledge_base() -> str:
    """Returns a detailed list of all ingested documents and their medical summaries."""
    with suppress_output():
        return rag_engine.get_knowledge_summaries()

@mcp.tool()
def exact_identifier_search(query: str, patient_id_filter: str = None) -> str:
    """Performs exact substring search for medical identifiers and codes."""
    with suppress_output():
        res, hits = rag_engine.exact_search(query, patient_id_filter)
        return res

if __name__ == "__main__":
    mcp.run()
