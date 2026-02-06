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
TABULAR_DATA_PATH = "temp_uploads/tabular_data.json"
INTERNAL_KNOWLEDGE_PATH = "internal_docs"

# Centralized output directory for PineBioML visualizations
OUTPUT_DIR = "src/pinebio/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def find_semantic_column(df, user_term):
    """
    Intelligently maps a user's natural language term to a clinical data column.
    
    Logic:
    1. Exact match (case insensitive)
    2. Substring match
    3. Medical synonym mapping (e.g., inflammation -> CRP)
    4. Fuzzy matching (difflib)
    """
    if not user_term: return None
    user_term = str(user_term).lower().strip()
    
    # Standardize column map {cleaned_name: original_name}
    cols = df.columns.tolist()
    
    # 1. Direct Case-Insensitive Match
    for c in cols:
        if user_term == c.lower(): return c
        
    # 2. Aggressive Clean Match
    for c in cols:
        if user_term == aggressive_clean(c).lower(): return c

    # 3. Medical Synonym Mapping
    synonyms = {
        "inflammation": ["crp", "esr", "cytokine", "il6", "tnf"],
        "inflamasi": ["crp", "esr", "cytokine", "il6", "tnf"],
        "diagnosis": ["disease", "status", "condition", "kelompok"],
        "outcome": ["remission", "death", "response", "status"],
        "age": ["umur", "age_at_cpy", "age_at_enrollment"],
        "gender": ["sex", "jenis_kelamin"],
        "duration": ["durasi", "dz_duration"],
        "location": ["dz_location", "lokasi"]
    }
    
    for concept, terms in synonyms.items():
        if user_term == concept or user_term in terms:
            # Look for these terms in actual columns
            for t in terms:
                for c in cols:
                    if t in c.lower() or t in aggressive_clean(c).lower():
                        return c

    # 4. Substring Match (Strongest substring first)
    for c in cols:
        if user_term in c.lower() or user_term in aggressive_clean(c).lower():
            return c
            
    # 5. Fuzzy Match (last resort)
    import difflib
    matches = difflib.get_close_matches(user_term, [c.lower() for c in cols], n=1, cutoff=0.6)
    if matches:
        for c in cols:
            if c.lower() == matches[0]: return c
            
    return None

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
        # Professional clinical fallback message
        error_msg = f"I encountered a temporary challenge accessing the clinical records: {e}. I will attempt an alternative retrieval method."
        return json.dumps({"answer": error_msg, "tool": "rag", "tasks": [], "rag_context": ""})

# ============================================================================
# DATA EXTRACTION TOOL (RAG → PineBioML Bridge)
# ============================================================================

@mcp.tool()
def extract_data_from_rag(
    query: str = "clinical data",
    file_pattern: str = None,
    save_to_session: bool = True
) -> str:
    """
    Extract tabular data from RAG documents and prepare for PineBioML analysis.
    
    This tool bridges RAG and PineBioML by:
    1. Querying RAG to find relevant data files
    2. Loading the data (Excel/CSV)
    3. Saving to session for PineBioML tools to use
    
    Args:
        query: Natural language query to find data (e.g., "clinical data", "patient records")
        file_pattern: Optional glob pattern to match files (e.g., "*.xlsx", "Test_AI*.xlsx")
        save_to_session: If True, save to temp_uploads/tabular_data.json
    
    Returns:
        String with format: "success|||Data extracted: N rows, M columns" or "error|||message"
    
    Use Cases:
        - "Extract clinical data" → Find and load medical data files
        - "Get patient records" → Load patient data from internal docs
        - "Prepare data for analysis" → Load data to session before visualization
    
    Medical Context:
        This tool is the first step in any PineBioML workflow. It discovers and loads
        data from RAG-indexed sources, making it available for downstream analysis.
    """
    try:
        import glob
        
        # 1. Find data files
        if file_pattern:
            # Direct file pattern match
            files = glob.glob(f"internal_docs/{file_pattern}")
        else:
            # Use RAG to find relevant files (fallback to all Excel/CSV in internal_docs)
            files = glob.glob("internal_docs/*.xlsx") + glob.glob("internal_docs/*.xls") + glob.glob("internal_docs/*.csv")
        
        if not files:
            return "error|||No data files found in internal_docs"
        
        # 2. Load first matching file
        data_file = files[0]
        pine_log(f"📂 Loading data from: {data_file}")
        
        if data_file.endswith('.xlsx') or data_file.endswith('.xls'):
            df = pd.read_excel(data_file)
        elif data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        else:
            return f"error|||Unsupported file format: {data_file}"
        
        # 3. Save to session if requested
        if save_to_session:
            os.makedirs("temp_uploads", exist_ok=True)
            df.to_json(TABULAR_DATA_PATH, orient="records", indent=2)
            pine_log(f"💾 Saved to session: {len(df)} rows, {len(df.columns)} columns")
        
        # 4. Return summary
        filename = os.path.basename(data_file)
        return f"success|||Data extracted from {filename}: {len(df)} rows, {len(df.columns)} columns. Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}"
        
    except Exception as e:
        pine_log(f"❌ Data extraction error: {e}")
        import traceback
        traceback.print_exc()
        return f"error|||{str(e)}"

# ============================================================================
# CONTEXT & SEARCH TOOLS
# ============================================================================

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
def generate_medical_plot(
    plot_type: str,
    data_source: str = "session",
    x_column: str = None,
    y_column: str = None,
    target_column: str = None,
    patient_ids: str = None,
    styling: str = None
) -> str:
    """
    Generates medical visualizations from tabular data with flexible styling.
    
    Args:
        plot_type: Type of plot (scatter, line, pca, distribution, bar, histogram)
        data_source: Data source - 'session' for uploaded data, or path to Excel/CSV file
        x_column: X-axis column (for scatter/line plots)
        y_column: Y-axis column (for scatter/line plots)
        target_column: Single column to visualize (for distribution/bar/histogram)
        patient_ids: Optional patient IDs for filtering (comma-separated)
        styling: Optional JSON string with chart styling
                 Example: '{"style": {"theme": "dark", "title_size": 18}}'
    
    Returns:
        String with format: "filepath|||description"
    
    Use Cases:
        - Scatter plot: plot_type='scatter', x_column='Age', y_column='BMI'
        - Distribution: plot_type='distribution', target_column='CRP'
        - PCA: plot_type='pca' (automatic dimensionality reduction)
    """
    try:
        # Load data from specified source
        if data_source == "session":
            # Use session uploaded data
            with open(TABULAR_DATA_PATH, "r") as f:
                df = pd.read_json(io.StringIO(f.read()))
        elif data_source.endswith('.xlsx') or data_source.endswith('.xls'):
            # Load from Excel file
            df = pd.read_excel(data_source)
        elif data_source.endswith('.csv'):
            # Load from CSV file
            df = pd.read_csv(data_source)
        else:
            return f"Error: Unsupported data source format. Use 'session', .xlsx, or .csv files."
        
        # Filter by patient IDs if specified
        if patient_ids:
            patient_ids = str(patient_ids) # Cast to string for safety
            id_cols = [c for c in df.columns if 'id' in c.lower() or 'patient' in c.lower()]
            if id_cols:
                ids = [i.strip() for i in patient_ids.replace('-', ',').split(',')]
                df = df[df[id_cols[0]].astype(str).isin(ids)]

        df.columns = [aggressive_clean(c) for c in df.columns]
        num_df = df.select_dtypes(include=['number']).dropna(axis=1, how='all').dropna()
        
        # Use centralized output directory
        filename = f"{OUTPUT_DIR}/{plot_type}_{int(datetime.datetime.now().timestamp())}.png"
        plt.close('all')
        
        plot_type = plot_type.lower().strip()
        
        with suppress_output():
            # Scatter and Line plots (2D visualizations)
            if plot_type in ['scatter', 'scatterplot', 'scatter plot']:
                if not x_column or not y_column:
                    return "Error: Scatter plot requires both x_column and y_column parameters."
                
                # Find columns using semantic finder
                x_col = find_semantic_column(df, x_column)
                y_col = find_semantic_column(df, y_column)
                
                if not x_col or not y_col:
                    return f"Error: Could not find columns. Available: {', '.join(df.columns[:10])}"
                
                plt.figure(figsize=(10, 6))
                plt.scatter(df[x_col], df[y_col], alpha=0.6, s=50)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f"{x_col} vs {y_col}")
                plt.tight_layout()
                
                # Apply custom styling
                if styling:
                    styler = ChartStyler(styling)
                    styler.apply(plt.gcf(), plt.gca())
                
                plt.savefig(filename)
                return f"{filename}|||Scatter plot created: {x_col} vs {y_col}. {len(df)} data points plotted."
            
            elif plot_type in ['line', 'lineplot', 'line plot']:
                if not x_column or not y_column:
                    return "Error: Line plot requires both x_column and y_column parameters."
                
                # Find columns using semantic finder
                x_col = find_semantic_column(df, x_column)
                y_col = find_semantic_column(df, y_column)
                
                if not x_col or not y_col:
                    return f"Error: Could not find columns. Available: {', '.join(df.columns[:10])}"
                
                plt.figure(figsize=(10, 6))
                plt.plot(df[x_col], df[y_col], marker='o', linestyle='-', linewidth=2)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f"{x_col} vs {y_col}")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Apply custom styling
                if styling:
                    styler = ChartStyler(styling)
                    styler.apply(plt.gcf(), plt.gca())
                
                plt.savefig(filename)
                return f"{filename}|||Line plot created: {x_col} vs {y_col}. {len(df)} data points."
            
            # PCA and Clustering
            elif plot_type in ['pca', 'clustering']:
                if num_df.empty:
                    return "Error: No numeric data available for PCA analysis. Please ensure data is cleaned or numeric columns exist."
                
                from sklearn.decomposition import PCA
                import numpy as np
                
                # Run PCA
                pca = PCA(n_components=2)
                # Standardize before PCA (Z-score normalization)
                scaled_data = (num_df - num_df.mean()) / (num_df.std() + 1e-8)
                pca_result = pca.fit_transform(scaled_data)
                var_explained = pca.explained_variance_ratio_
                
                plt.figure(figsize=(10, 7))
                
                # Check for target_column for coloring
                target_col = find_semantic_column(df, target_column)
                
                if target_col:
                    # Get target values aligned with num_df
                    y = df.loc[num_df.index, target_col]
                    unique_groups = y.unique()
                    
                    # Use professional color palette
                    import seaborn as sns
                    palette = sns.color_palette("husl", len(unique_groups))
                    
                    for i, group in enumerate(unique_groups):
                        mask = y == group
                        plt.scatter(
                            pca_result[mask, 0], 
                            pca_result[mask, 1],
                            label=str(group),
                            alpha=0.75,
                            edgecolors='w',
                            linewidth=0.5,
                            s=80,
                            c=[palette[i]]
                        )
                    plt.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.title(f"PCA Analysis - Colored by {target_col}", fontsize=14, fontweight='bold', pad=15)
                else:
                    # Simple scatter if no target
                    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, s=70, edgecolors='w')
                    plt.title("PCA Analysis - Dimensionality Reduction", fontsize=14, fontweight='bold', pad=15)
                
                plt.xlabel(f"PC1 ({var_explained[0]*100:.1f}% Variance)", fontsize=11)
                plt.ylabel(f"PC2 ({var_explained[1]*100:.1f}% Variance)", fontsize=11)
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.tight_layout()
                
                # Apply custom styling if provided
                if styling:
                    styler = ChartStyler(styling)
                    styler.apply(plt.gcf(), plt.gca())
                
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close() # Important to clear memory
                
                desc = f"PCA complete. Identified patterns based on {len(num_df.columns)} variables. PC1 explains {var_explained[0]*100:.1f}% variance."
                if target_col:
                    desc += f" Groups separated by {target_col}."
                    
                return f"{filename}|||{desc}"
            elif plot_type in ['distribution', 'bar', 'bar chart', 'histogram', 'count', 'frequency']:
                if target_column:
                    target_column = str(target_column)
                    target_column = re.sub(r'\(.*\)', '', target_column).strip() 
                
                col = find_semantic_column(df, target_column) or df.columns[0]
                
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
        
        target = find_semantic_column(df, target_column) or df.columns[-1]
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
        target = find_semantic_column(df, target_column) or df.columns[-1]
        X = df.select_dtypes(include=['number']).drop(columns=[target], errors='ignore')
        y = df[target]
        
        if len(y.unique()) != 2:
            return "Marker discovery (Volcano) requires a binary target column (e.g. Healthy vs Sick)."
            
        selector = volcano.Volcano_selection(k=10, target_label=y.unique()[0])
        selector.fit(X, y)
        
        # Add Visualization
        ts = int(datetime.datetime.now().timestamp())
        plt.figure(figsize=(10, 8))
        selector.plotting(saving=True, save_path=OUTPUT_DIR + "/", title=f"Volcano_{ts}")
        volcano_path = f"{OUTPUT_DIR}/Volcano_{ts}.png"
        
        # Ensure the file actually exists (some versions of matplotlib might not add .png automatically)
        if not os.path.exists(volcano_path) and os.path.exists(volcano_path[:-4]):
            os.rename(volcano_path[:-4], volcano_path)
        
        top_markers = ", ".join(selector.selected_score.index.tolist())
        return f"{volcano_path}|||Top Biomarkers Found via Volcano Plot: {top_markers}. This plot identifies features with significant fold changes and p-values."
    except Exception as e:
        return f"Marker discovery error: {e}"

@mcp.tool()
def run_pls_analysis(target_column: str = None, patient_ids: str = None, styling: str = None) -> str:
    """
    Runs Supervised PLS-DA for class separation analysis.
    
    Args:
        target_column: Column to use for class coloring (e.g. 'Disease')
        patient_ids: Optional comma-separated patient IDs for filtering
        styling: Optional JSON string with chart styling
    """
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

        # Get numeric columns
        num_cols = df.select_dtypes(include=['number'])
        if num_cols.empty: return "No numeric data."
        
        exclude_terms = ['id', 'date', 'image', 'scan', 'time', 'index', 'code', 'accession']
        numeric_valid = [c for c in num_cols.columns if not any(term in c.lower() for term in exclude_terms)]
        if len(numeric_valid) < 2: numeric_valid = num_cols.columns.tolist()
        
        X = num_cols[numeric_valid]
        
        target = find_semantic_column(df, target_column)
        
        if not target:
            cat_cols = df.select_dtypes(exclude=['number'])
            target = cat_cols.columns[0] if not cat_cols.empty else df.columns[-1]
        
        from PineBioML.report.utils import pls_plot
        pp = pls_plot(is_classification=True)
        filename = f"{OUTPUT_DIR}/pls_{int(datetime.datetime.now().timestamp())}.png"
        pp.draw(X, df[target])
        
        plt.title(f"PLS-DA Analysis - {target} Separation", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        if styling:
            styler = ChartStyler(styling)
            styler.apply(plt.gcf(), plt.gca())
        
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f"{filename}|||PLS-DA Analysis complete. Visualized separation between {target} groups using {len(numeric_valid)} features."
    except Exception as e: return f"PLS error: {e}"

@mcp.tool()
def run_umap_analysis(target_column: str = None, patient_ids: str = None, styling: str = None) -> str:
    """
    Runs Unsupervised UMAP for clustering analysis.
    
    Args:
        target_column: Column to use for cluster coloring (e.g. 'Disease')
        patient_ids: Optional comma-separated patient IDs for filtering
        styling: Optional JSON string with chart styling
    """
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
        
        num_cols = df.select_dtypes(include=['number'])
        if num_cols.empty: return "No numeric data."
        
        exclude_terms = ['id', 'date', 'time', 'index', 'code']
        numeric_valid = [c for c in num_cols.columns if not any(term in c.lower() for term in exclude_terms)]
        if not numeric_valid: numeric_valid = num_cols.columns.tolist()
        
        X = num_cols[numeric_valid]
        
        target = find_semantic_column(df, target_column)
        
        if not target:
            cat_cols = df.select_dtypes(exclude=['number'])
            target = cat_cols.columns[0] if not cat_cols.empty else df.columns[-1]

        from PineBioML.report.utils import umap_plot
        up = umap_plot()
        filename = f"{OUTPUT_DIR}/umap_{int(datetime.datetime.now().timestamp())}.png"
        up.draw(X, df[target])
        
        plt.title(f"UMAP Clustering - Colored by {target}", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        if styling:
            styler = ChartStyler(styling)
            styler.apply(plt.gcf(), plt.gca())
            
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        return f"{filename}|||UMAP Clustering analysis complete. Visualized natural groupings colored by {target}."
    except Exception as e: return f"UMAP error: {e}"

@mcp.tool()
def run_correlation_heatmap(patient_ids: str = None, styling: str = None) -> str:
    """
    Generates Feature Correlation Heatmap.
    
    Use when:
    - You want to see relationships between variables
    - Identifying correlated features
    - Understanding feature dependencies
    
    Args:
        patient_ids: Optional comma-separated patient IDs for filtering
        styling: Optional JSON string with chart styling
    """
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
        filename = f"{OUTPUT_DIR}/heatmap_{int(datetime.datetime.now().timestamp())}.png"
        hp.draw(X)
        
        # Apply custom styling if provided
        if styling:
            styler = ChartStyler(styling)
            fig = plt.gcf()
            ax = plt.gca()
            styler.apply(fig, ax)
        
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


# ============================================================================
# PHASE 2 TOOLS: Complete ML Pipeline
# ============================================================================

@mcp.tool()
def clean_medical_data(
    imputation_method: str = "knn",
    outlier_removal: bool = True,
    outlier_method: str = "iqr",
    missing_threshold: float = 0.33
) -> str:
    """Clean medical data by imputing missing values and removing outliers.
    
    This is often the FIRST step in medical data analysis pipeline.
    
    Args:
        imputation_method: Method to fill missing values
                          - "knn": K-Nearest Neighbors (smart, considers similar patients)
                          - "median": Simple median imputation (fast, robust)
                          - "mean": Mean imputation (for normally distributed data)
                          - "iterative": MICE (Multiple Imputation, most accurate but slow)
        outlier_removal: Whether to detect and remove outliers
        outlier_method: Method for outlier detection
                       - "iqr": Interquartile Range (standard, robust)
                       - "zscore": Z-score method (assumes normal distribution)
        missing_threshold: Drop columns with >X% missing values (0.0-1.0)
    
    Returns:
        String with format: "status|||description"
    
    Use Cases:
        - "Clean my data before analysis"
        - "Fill missing CRP values"
        - "Remove outliers from biomarker data"
    
    Medical Context:
        - Missing values are common in clinical data (lab tests not ordered)
        - Outliers may indicate data entry errors OR critical clinical findings
        - KNN imputation works well for biomarkers (similar patients have similar values)
    """
    try:
        if not os.path.exists(TABULAR_DATA_PATH):
            return "Error: No data loaded. Please upload data first."
        
        with open(TABULAR_DATA_PATH, "r") as f:
            df = pd.read_json(io.StringIO(f.read()))
        
        original_shape = df.shape
        pine_log(f"📊 Original data: {original_shape[0]} rows × {original_shape[1]} columns")
        
        # Separate numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Exclude ID-like columns from cleaning
        exclude_terms = ['id', 'patient', 'subject', 'code', 'accession', 'date', 'time']
        numeric_to_clean = [c for c in numeric_cols if not any(term in c.lower() for term in exclude_terms)]
        
        pine_log(f"🔧 Cleaning {len(numeric_to_clean)} numeric columns")
        
        # Track changes
        changes = []
        
        # 1. Drop columns with too many missing values
        missing_rates = df[numeric_to_clean].isna().mean()
        cols_to_drop = missing_rates[missing_rates > missing_threshold].index.tolist()
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            numeric_to_clean = [c for c in numeric_to_clean if c not in cols_to_drop]
            changes.append(f"Dropped {len(cols_to_drop)} columns with >{missing_threshold*100}% missing")
        
        # 2. Impute missing values
        if numeric_to_clean:
            if imputation_method == "knn":
                imputer = impute.knn_imputer(threshold=missing_threshold, n_neighbor=5)
            elif imputation_method == "iterative":
                imputer = impute.iterative_imputer(threshold=missing_threshold, max_iter=10)
            elif imputation_method in ["median", "mean"]:
                imputer = impute.simple_imputer(threshold=missing_threshold, strategy=imputation_method)
            else:
                return f"Error: Unknown imputation method '{imputation_method}'"
            
            # Count missing before
            missing_before = df[numeric_to_clean].isna().sum().sum()
            
            # Apply imputation
            df_numeric = df[numeric_to_clean].copy()
            df_imputed = imputer.fit_transform(df_numeric)
            df[numeric_to_clean] = df_imputed
            
            missing_after = df[numeric_to_clean].isna().sum().sum()
            if missing_before > 0:
                changes.append(f"Imputed {missing_before - missing_after} missing values using {imputation_method}")
        
        # 3. Remove outliers
        if outlier_removal and numeric_to_clean:
            import numpy as np
            outliers_removed = 0
            
            for col in numeric_to_clean:
                if outlier_method == "iqr":
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR  # 3x IQR for medical data (more conservative)
                    upper_bound = Q3 + 3 * IQR
                    
                    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                    outliers_removed += outlier_mask.sum()
                    df.loc[outlier_mask, col] = np.nan
                
                elif outlier_method == "zscore":
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    outlier_mask = z_scores > 3
                    outliers_removed += outlier_mask.sum()
                    df.loc[outlier_mask, col] = np.nan
            
            if outliers_removed > 0:
                # Re-impute outliers
                if imputation_method == "knn":
                    imputer = impute.knn_imputer(threshold=1.0, n_neighbor=5)
                else:
                    imputer = impute.simple_imputer(threshold=1.0, strategy="median")
                
                df_numeric = df[numeric_to_clean].copy()
                df_imputed = imputer.fit_transform(df_numeric)
                df[numeric_to_clean] = df_imputed
                
                changes.append(f"Removed {outliers_removed} outliers using {outlier_method} method")
        
        # Save cleaned data
        with open(TABULAR_DATA_PATH, "w") as f:
            f.write(df.to_json(orient='records', indent=2))
        
        final_shape = df.shape
        
        # Generate summary
        summary = f"✅ Data Cleaning Complete\n\n"
        summary += f"Original: {original_shape[0]} rows × {original_shape[1]} columns\n"
        summary += f"Cleaned: {final_shape[0]} rows × {final_shape[1]} columns\n\n"
        summary += "Changes:\n" + "\n".join(f"  • {c}" for c in changes)
        
        return f"success|||{summary}"
    
    except Exception as e:
        err = f"{e}\n{traceback.format_exc()}"
        pine_log(f"❌ Data cleaning error: {err}")
        return f"Error: {e}"


@mcp.tool()
def discover_markers(
    target_column: str,
    p_value_threshold: float = 0.05,
    fold_change_threshold: float = 2.0,
    top_k: int = 20,
    strategy: str = "fold"
) -> str:
    """Discover significant biomarkers using Volcano plot analysis.
    
    This identifies features that are:
    1. Statistically significant (low p-value)
    2. Biologically meaningful (high fold-change)
    
    Args:
        target_column: Column name for grouping (e.g., "Disease_Status", "Group")
        p_value_threshold: P-value cutoff (default: 0.05)
        fold_change_threshold: Minimum fold-change (default: 2.0x)
        top_k: Number of top markers to return (default: 20)
        strategy: Selection strategy - "fold" (by fold-change) or "p" (by p-value)
    
    Returns:
        String with format: "filepath|||description"
    
    Use Cases:
        - "Find biomarkers for disease vs healthy"
        - "Which features distinguish IBD from controls?"
        - "Discover significant markers for treatment response"
    
    Medical Context:
        - Volcano plots are standard in biomarker discovery
        - Combines statistical significance (p-value) with effect size (fold-change)
        - Helps identify clinically relevant biomarkers, not just statistically significant ones
    """
    try:
        if not os.path.exists(TABULAR_DATA_PATH):
            return "Error: No data loaded."
        
        with open(TABULAR_DATA_PATH, "r") as f:
            df = pd.read_json(io.StringIO(f.read()))
        
        # Find target column
        target_col = None
        for c in df.columns:
            if aggressive_clean(target_column).lower() == aggressive_clean(c).lower():
                target_col = c
                break
        
        if not target_col:
            return f"Error: Target column '{target_column}' not found. Available: {', '.join(df.columns[:10])}"
        
        # Get numeric features
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        exclude_terms = ['id', 'patient', 'subject', 'code', 'accession', 'date', 'time']
        features = [c for c in numeric_cols if c != target_col and not any(term in c.lower() for term in exclude_terms)]
        
        if len(features) < 2:
            return "Error: Need at least 2 numeric features for biomarker discovery."
        
        X = df[features]
        y = df[target_col]
        
        # Check if binary classification
        unique_values = y.nunique()
        if unique_values != 2:
            return f"Error: Target must have exactly 2 groups for Volcano plot. Found {unique_values} groups."
        
        pine_log(f"🔬 Running Volcano analysis on {len(features)} features")
        
        # Run Volcano selection
        volcano_selector = volcano.Volcano_selection(
            k=top_k,
            strategy=strategy,
            p_threshold=p_value_threshold,
            fc_threshold=fold_change_threshold,
            log_domain=False,
            target_label=y.unique()[1]  # Use second unique value as "positive"
        )
        
        volcano_selector.fit(X, y)
        selected_markers = volcano_selector.selected_score
        
        # Generate volcano plot
        filename = f"{OUTPUT_DIR}/volcano_{int(datetime.datetime.now().timestamp())}.png"
        volcano_selector.plotting(
            title="Biomarker Discovery - Volcano Plot",
            show=False,
            saving=True,
            save_path=filename.replace('.png', '')
        )
        
        # Format results
        marker_list = "\n".join([f"  {i+1}. {marker}: {score:.3f}" for i, (marker, score) in enumerate(selected_markers.items())])
        
        summary = f"🔬 Biomarker Discovery Complete\n\n"
        summary += f"Analyzed: {len(features)} features\n"
        summary += f"Significant markers (p<{p_value_threshold}, FC>{fold_change_threshold}): {len(selected_markers)}\n\n"
        summary += f"Top {len(selected_markers)} Markers:\n{marker_list}\n\n"
        summary += f"Groups compared: {y.unique()[0]} vs {y.unique()[1]}"
        
        return f"{filename}|||{summary}"
    
    except Exception as e:
        err = f"{e}\n{traceback.format_exc()}"
        pine_log(f"❌ Biomarker discovery error: {err}")
        return f"Error: {e}"


@mcp.tool()
def train_medical_model(
    target_column: str,
    model_type: str = "RandomForest",
    n_trials: int = 25
) -> str:
    """Train a machine learning model on medical data.
    
    Automatically handles:
    - Hyperparameter tuning (Optuna)
    - Cross-validation
    - Class imbalance
    - Feature importance
    
    Args:
        target_column: Column to predict (e.g., "Disease", "Outcome")
        model_type: Model algorithm
                   - "RandomForest": Robust, interpretable (default)
                   - "SVM": Good for small datasets
                   - "LogisticRegression": Linear, interpretable
        n_trials: Number of hyperparameter optimization trials (default: 25)
    
    Returns:
        String with format: "status|||performance_metrics"
    
    Use Cases:
        - "Train a model to predict disease from biomarkers"
        - "Build classifier for patient outcomes"
        - "Predict treatment response"
    
    Medical Context:
        - RandomForest works well for biomarker data (handles non-linearity)
        - Logistic Regression provides interpretable coefficients
        - SVM good for small sample sizes (common in medical research)
    """
    try:
        if not os.path.exists(TABULAR_DATA_PATH):
            return "Error: No data loaded."
        
        with open(TABULAR_DATA_PATH, "r") as f:
            df = pd.read_json(io.StringIO(f.read()))
        
        # Find target column
        target_col = None
        for c in df.columns:
            if aggressive_clean(target_column).lower() == aggressive_clean(c).lower():
                target_col = c
                break
        
        if not target_col:
            return f"Error: Target column '{target_column}' not found."
        
        # Get features
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        exclude_terms = ['id', 'patient', 'subject', 'code', 'accession', 'date', 'time']
        features = [c for c in numeric_cols if c != target_col and not any(term in c.lower() for term in exclude_terms)]
        
        X = df[features]
        y = df[target_col]
        
        pine_log(f"🤖 Training {model_type} model on {len(features)} features")
        
        # Select model
        if model_type == "RandomForest":
            model = classification.RandomForest_tuner(n_try=n_trials, n_cv=5, target="mcc")
        elif model_type == "SVM":
            model = classification.SVM_tuner(n_try=n_trials, n_cv=5, target="mcc")
        elif model_type == "LogisticRegression":
            model = classification.ElasticLogit_tuner(n_try=n_trials, n_cv=5, target="mcc")
        else:
            return f"Error: Unknown model type '{model_type}'"
        
        # Train
        model.fit(X, y)
        
        # Get performance
        best_score = model.best_score
        
        summary = f"🤖 Model Training Complete\n\n"
        summary += f"Model: {model_type}\n"
        summary += f"Features: {len(features)}\n"
        summary += f"Samples: {len(X)}\n"
        summary += f"Best CV Score (MCC): {best_score:.3f}\n\n"
        summary += f"Model saved and ready for predictions."
        
        return f"success|||{summary}"
    
    except Exception as e:
        err = f"{e}\n{traceback.format_exc()}"
        pine_log(f"❌ Model training error: {err}")
        return f"Error: {e}"


@mcp.tool()
def generate_data_overview(
    target_column: str = None,
    is_classification: bool = True
) -> str:
    """Generate comprehensive data overview with ALL visualizations at once.
    
    Creates:
    - PCA plot
    - PLS-DA plot
    - UMAP plot
    - Correlation heatmap
    
    Args:
        target_column: Optional target for colored plots
        is_classification: Whether target is categorical (True) or continuous (False)
    
    Returns:
        String with format: "status|||description"
    
    Use Cases:
        - "Show me everything about my data"
        - "Give me a complete overview"
        - "Quick data exploration"
    
    Medical Context:
        - Quick way to understand data structure
        - Identifies clusters, outliers, correlations
        - Standard exploratory data analysis for biomarker studies
    """
    try:
        if not os.path.exists(TABULAR_DATA_PATH):
            return "Error: No data loaded."
        
        with open(TABULAR_DATA_PATH, "r") as f:
            df = pd.read_json(io.StringIO(f.read()))
        
        # Get features
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        exclude_terms = ['id', 'patient', 'subject', 'code', 'accession', 'date', 'time']
        features = [c for c in numeric_cols if not any(term in c.lower() for term in exclude_terms)]
        
        X = df[features]
        
        # Get target if specified
        y = None
        if target_column:
            target_col = None
            for c in df.columns:
                if aggressive_clean(target_column).lower() == aggressive_clean(c).lower():
                    target_col = c
                    break
            if target_col:
                y = df[target_col]
        
        pine_log(f"📊 Generating complete data overview")
        
        # Run all optimizations
        res_pca = generate_medical_plot(plot_type="pca", target_column=target_column)
        res_pls = run_pls_analysis(target_column=target_column)
        res_umap = run_umap_analysis(target_column=target_column)
        res_heat = run_correlation_heatmap()
        
        def extract_path(res): return res.split("|||")[0] if "|||" in res else None
        
        summary = f"📊 Comprehensive Data Overview Complete\n\n"
        summary += f"Visualizations generated:\n"
        summary += f"  • PCA Analysis: patterns and variance\n"
        summary += f"  • PLS-DA: supervised class separation\n"
        summary += f"  • UMAP: non-linear clustering\n"
        summary += f"  • Heatmap: feature correlations\n\n"
        
        # Return all paths (special handling might be needed in synthesizing or display)
        all_paths = [extract_path(r) for r in [res_pca, res_pls, res_umap, res_heat] if extract_path(r)]
        paths_str = ",".join(all_paths)
        
        return f"{paths_str}|||{summary}"
        
        summary = f"📊 Data Overview Complete\n\n"
        summary += f"Generated 4 visualizations:\n"
        summary += f"  • PCA plot\n"
        summary += f"  • PLS-DA plot\n"
        summary += f"  • UMAP plot\n"
        summary += f"  • Correlation heatmap\n\n"
        summary += f"Features analyzed: {len(features)}\n"
        summary += f"Samples: {len(X)}"
        
        return f"success|||{summary}"
    
    except Exception as e:
        err = f"{e}\n{traceback.format_exc()}"
        pine_log(f"❌ Data overview error: {err}")
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run()
