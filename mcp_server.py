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

import sys
import contextlib
import warnings

@contextlib.contextmanager
def suppress_output():
    """
    Redirects stdout and stderr to server_debug.log to prevent 
    library logs from breaking MCP JSON-RPC protocol.
    Also suppresses Python warnings.
    """
    with open("server_debug.log", "a") as logfile:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        original_showwarning = warnings.showwarning
        
        # Redirect output
        sys.stdout = logfile
        sys.stderr = logfile
        
        # Suppress warnings
        warnings.filterwarnings('ignore')
        
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            warnings.showwarning = original_showwarning

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
    
    try:
        with suppress_output():
            debug_log(f"Starting ingestion from {directory_path} ({doc_type})")
            
            all_docs = DocumentProcessor.load_directory(directory_path, doc_type=doc_type)
            
            if not all_docs:
                return "No valid documents found in the directory."
            
            # Store the first detected tabular data for plotting tools (Persistence to Disk)
            tabular_found = False
            for doc in all_docs:
                if "df_json" in doc.metadata:
                    with open(TABULAR_DATA_PATH, "w") as f:
                        f.write(doc.metadata["df_json"])
                    tabular_found = True
                    debug_log("Tabular data found and cached.")
                    break
                    
            rag_engine.ingest_documents(all_docs)
            
            summary = f"Successfully ingested {len(all_docs)} segments into {doc_type}."
            if tabular_found:
                summary += "\n(Tabular data cached for analysis)"
            
            return summary
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        debug_log(f"Ingestion CRASH: {error_detail}")
        return f"Ingestion failed: {str(e)}\n\nCheck server logs for details."

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
    try:
        schema = ""
        if os.path.exists(TABULAR_DATA_PATH):
            try:
                with open(TABULAR_DATA_PATH, "r") as f:
                    df_temp = pd.read_json(io.StringIO(f.read()))
                    schema = ", ".join([aggressive_clean(c) for c in df_temp.columns])
            except Exception as schema_err:
                debug_log(f"Schema extraction error: {schema_err}")
                pass

        debug_log(f"smart_intent_dispatch called: question='{question}', patient_filter='{patient_id_filter}'")
        
        answer, tool, tasks = rag_engine.smart_query(question, patient_id_filter, schema_context=schema)
        
        result = json.dumps({
            "answer": answer,
            "tool": tool,
            "tasks": tasks
        })
        
        debug_log(f"smart_intent_dispatch returning: {result[:200]}...")
        return result
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        debug_log(f"CRITICAL ERROR in smart_intent_dispatch:\n{error_trace}")
        
        # Return error as JSON so app.py doesn't crash
        return json.dumps({
            "answer": f"❌ Internal error: {str(e)}. Please check server logs.",
            "tool": "rag",
            "tasks": []
        })

@mcp.tool()
def query_medical_rag(question: str, patient_id_filter: str = None) -> str:
    """
    Queries the medical RAG system for patient info or medical guidelines.
    """
    with suppress_output():
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
    # ONLY auto-switch if user didn't explicitly specify a target_column
    if numeric_df.shape[1] == 1 and not target_column:
        single_col = numeric_df.columns[0]
        debug_log(f"Auto-switching to Distribution plot for single feature: {single_col}")
        plot_type = 'distribution'
        target_column = single_col 
    elif numeric_df.shape[1] == 1 and target_column:
        # User specified a column - just switch to distribution plot type, keep their column
        debug_log(f"User requested column '{target_column}', switching to distribution plot")
        plot_type = 'distribution'
    
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
    import uuid
    # Use unique ID to prevent overwrites in quick succession
    unique_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
    # Include column name in filename for debugging
    col_suffix = target_column.replace(' ', '_')[:20] if target_column else 'default'
    plot_path = f"plots/{plot_type}_{col_suffix}_{unique_id}.png"
    os.makedirs("plots", exist_ok=True)
    
    plt.close('all')
    interpretation = ""
    
    def format_for_display(text: str) -> str:
        """
        Convert technical column names to human-readable labels.
        This is for VISUALIZATION LABELS, not for filtering data.
        
        Examples:
        - 'age_years' -> 'Usia (Tahun)' or 'Age (Years)'
        - 'sp_mayo' -> 'Skor P-Mayo'
        - 'bmi_calc' -> 'Indeks Massa Tubuh (BMI)'
        """
        if not text: 
            return "General Analysis"
        
        # Comprehensive medical terminology mapping
        # Format: technical_name -> Human-Readable Label
        medical_terms = {
            # Mayo Score variations
            "sum pmayo": "Skor Total P-Mayo",
            "sum_pmayo": "Skor Total P-Mayo",
            "sp mayo": "Skor P-Mayo",
            "sp_mayo": "Skor P-Mayo", 
            "pmayo": "P-Mayo Score",
            "mayo score": "Mayo Score",
            "mayo_score": "Mayo Score",
            
            # Demographics
            "age": "Usia",
            "age_years": "Usia (Tahun)",
            "age years": "Usia (Tahun)",
            "patient age": "Usia Pasien",
            "gender": "Jenis Kelamin",
            "sex": "Jenis Kelamin",
            
            # Body measurements
            "bmi": "Indeks Massa Tubuh (BMI)",
            "bmi_calc": "Indeks Massa Tubuh",
            "weight": "Berat Badan (kg)",
            "height": "Tinggi Badan (cm)",
            "body weight": "Berat Badan",
            "body_weight": "Berat Badan",
            
            # Lab values
            "glucose": "Glukosa Darah",
            "blood glucose": "Glukosa Darah",
            "blood_glucose": "Glukosa Darah",
            "hba1c": "HbA1c",
            "cholesterol": "Kolesterol",
            "hdl": "HDL",
            "ldl": "LDL",
            "triglycerides": "Trigliserida",
            
            # Clinical measurements
            "blood pressure": "Tekanan Darah",
            "blood_pressure": "Tekanan Darah",
            "bp": "Tekanan Darah",
            "heart rate": "Detak Jantung",
            "heart_rate": "Detak Jantung",
            "hr": "Detak Jantung (HR)",
            
            # Diagnosis
            "diagnosis": "Diagnosis",
            "diagnosis_code": "Kode Diagnosis",
            "disease": "Penyakit",
            "condition": "Kondisi",
            
            # Generic technical terms
            "avg_intensity": "Intensitas Rata-rata",
            "mean_value": "Nilai Rata-rata",
            "std_dev": "Deviasi Standar",
            "count": "Jumlah",
            "frequency": "Frekuensi",
        }
        
        # Normalize text for matching
        text_lower = text.lower().strip()
        
        # Try exact match first
        if text_lower in medical_terms:
            return medical_terms[text_lower]
        
        # Try partial match
        for key, readable_label in medical_terms.items():
            if key in text_lower:
                return readable_label
        
        # If no match, clean up underscores and title case
        cleaned = text.replace('_', ' ').replace('.', ' ').title()
        return cleaned

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
        
        # VALIDATION: Ensure the resolved target exists in the dataframe
        # This prevents errors in plot generation and provides clear guidance
        if target_column and target_column != "General Sample":
            if target_column not in df.columns:
                # Column doesn't exist - provide fuzzy suggestions
                from difflib import get_close_matches
                suggestions = get_close_matches(target_column, df.columns.tolist(), n=3, cutoff=0.6)
                
                suggestion_text = ""
                if suggestions:
                    suggestion_text = f"**Did you mean**: {', '.join(suggestions)}\n\n"
                
                return (f"❌ Column '{target_column}' not found in dataset.\n\n"
                        f"{suggestion_text}"
                        f"**Available columns** (first 10): {', '.join(df.columns[:10].tolist())}\n\n"
                        f"**Tip**: Check spelling or try one of the suggested columns.")

        import seaborn as sns
        # 2. Plot Generation
        if plot_type.lower() == 'pca':
            from PineBioML.report.utils import pca_plot
            if numeric_df.shape[1] < 2 or numeric_df.shape[0] < 2:
                available_features = [format_for_display(c) for c in numeric_df.columns]
                return (f"⚠️ Data Tidak Cukup untuk PCA\n\n"
                        f"**Status**: {numeric_df.shape[0]} sampel, {numeric_df.shape[1]} fitur medis\n"
                        f"**Perlu**: Minimal 2 sampel dan 2 fitur\n"
                        f"**Fitur tersedia**: {', '.join(available_features) if available_features else 'Tidak ada'}\n\n"
                        f"**Saran**: Upload file dengan lebih banyak data medis (contoh: Usia, BMI, Glukosa)")
            
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
            top_markers_raw = pd.Series(abs(sk_pca.components_[0]), index=numeric_df.columns).sort_values(ascending=False).head(3).index.tolist()
            # Convert to readable labels
            top_markers = [format_for_display(m) for m in top_markers_raw]
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
            # CRITICAL VALIDATION: Ensure target_column survived the ambiguity filter
            if not target_column:
                return "Error: Distribution plot requires a valid target_column (e.g., 'Age')."
            
            debug_log(f"=== DISTRIBUTION PLOT DEBUG ===")
            debug_log(f"Requested target_column: '{target_column}'")
            debug_log(f"Available columns: {list(df.columns)}")
            
            # CASE-INSENSITIVE COLUMN MATCHING
            # rag_engine may return 'age_at_cpy' but df has 'Age At Cpy' after cleaning
            actual_column = None
            target_lower = target_column.lower().replace('_', ' ').replace('-', ' ')
            
            for col in df.columns:
                col_normalized = col.lower().replace('_', ' ').replace('-', ' ')
                if target_lower == col_normalized or target_lower in col_normalized or col_normalized in target_lower:
                    actual_column = col
                    debug_log(f"Matched '{target_column}' -> '{col}'")
                    break
            
            if not actual_column:
                return f"❌ Column '{target_column}' not found.\n\n**Available**: {list(df.columns[:10])}"
            
            # IMPORTANT: Clear any existing figure to prevent caching
            plt.clf()
            plt.figure(figsize=(10, 6))
            
            # Get series and try to convert to numeric (handles mixed types)
            series = df[actual_column].dropna()
            series_numeric = pd.to_numeric(series, errors='coerce')
            
            # Use numeric version if possible, otherwise original
            if series_numeric.notna().sum() > 0:
                series = series_numeric.dropna()
            
            # Update display_target for titles
            display_target = format_for_display(actual_column)
            
            # Safe range logging
            try:
                range_info = f"range: {series.min()} to {series.max()}"
            except:
                range_info = f"range: ({series.nunique()} unique values)"
            
            debug_log(f"Plotting column '{actual_column}' (display: '{display_target}') with {len(series)} values, {range_info}")
            
            # SMART CATEGORICAL DETECTION:
            # Even if numeric type, if < 10 unique values, treat as categorical
            n_unique = series.nunique()
            is_categorical = (not pd.api.types.is_numeric_dtype(series)) or (n_unique < 10)
            
            debug_log(f"Distribution plot for '{actual_column}': {n_unique} unique values, categorical={is_categorical}")
            
            if is_categorical:
                # BAR CHART for categorical data
                counts = series.value_counts().sort_index()
                
                # Convert numeric categories to readable labels if applicable
                if pd.api.types.is_numeric_dtype(series):
                    # For Sex (0/1), Age groups, etc.
                    labels = counts.index.astype(str)
                else:
                    labels = counts.index
                
                plt.bar(range(len(counts)), counts.values, color='skyblue', edgecolor='navy')
                plt.xticks(range(len(counts)), labels, rotation=45 if len(labels) > 5 else 0)
                plt.title(f"Frequency Distribution: {display_target}")
                plt.xlabel(display_target)
                plt.ylabel("Count")
                
                # Add value labels on bars
                for i, v in enumerate(counts.values):
                    plt.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
                
                # Interpretation
                most_common = counts.idxmax()
                interpretation = f"### Distribution: {display_target}\n"
                interpretation += f"- **Most Common**: {most_common} ({counts.max()} pasien)\n"
                interpretation += f"- **Categories**: {n_unique}\n"
                interpretation += f"- **Total**: {len(series)} pasien"
                
            else:
                # HISTOGRAM for continuous numeric data
                sns.histplot(series, kde=True, color='teal', bins=20)
                plt.title(f"Distribution Analysis: {display_target}")
                plt.xlabel(display_target)
                plt.ylabel("Frequency / Count")
                interpretation = f"### Distribution Analysis: {display_target}\n- **Mean**: {series.mean():.2f}\n- **Median**: {series.median():.2f}\n- **Range**: {series.min():.1f} - {series.max():.1f}"
            
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
    Trains ML model (RandomForest, XGBoost, SVM) with hyperparameter tuning.
    Auto-detects task type (Classification vs Regression).
    Auto-installs missing dependencies if needed.
    """
    if not os.path.exists(TABULAR_DATA_PATH):
        return "Error: No data ingested. Please upload data first."
    
    # AUTO-DEPENDENCY INSTALLER
    def auto_install(package_name):
        """Auto-install missing package"""
        try:
            import subprocess
            import sys
            debug_log(f"Auto-installing {package_name}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                debug_log(f"✅ Successfully installed {package_name}")
                return True
            else:
                debug_log(f"❌ Failed to install {package_name}: {result.stderr}")
                return False
        except Exception as e:
            debug_log(f"Installation error: {e}")
            return False
    
    # Check and install required packages
    required_packages = ['statsmodels', 'shap', 'catboost', 'optuna']
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            debug_log(f"Missing package: {pkg}. Auto-installing...")
            if not auto_install(pkg):
                return (f"⚠️ Auto-installation failed for '{pkg}'.\n\n"
                        f"Please manually install: `pip install {pkg}`\n\n"
                        f"Then try training again.")
    
    try:
        # Load data
        debug_log("Loading data for training...")
        with open(TABULAR_DATA_PATH, "r") as f:
            df = pd.read_json(io.StringIO(f.read()))
            
        # Target column discovery
        if not target_column or target_column not in df.columns:
            categorical = [c for c in df.columns if df[c].nunique() <= 5 and not pd.api.types.is_numeric_dtype(df[c])]
            target_column = categorical[0] if categorical else df.columns[-1]
            debug_log(f"Auto-selected target: {target_column}")
            
        # Prepare data: separate X and y
        y = df[target_column].dropna()
        x = df.loc[y.index].select_dtypes(include=['number']).drop(columns=[target_column], errors='ignore')
        
        # Remove rows with NaN in features
        x = x.dropna()
        y = y.loc[x.index]
        
        if len(x) < 10:
            return f"❌ Error: Insufficient data after cleaning.\n\nOnly {len(x)} samples available. Need at least 10 for training.\n\n**Suggestion**: Upload more data or check for missing values."

        # Auto Detect Task Type
        is_regression = pd.api.types.is_numeric_dtype(y) and y.nunique() > 10
        task_name = "Regression" if is_regression else "Classification"
        
        debug_log(f"Task: {task_name}, Samples: {len(x)}, Features: {len(x.columns)}")
        
        # Initialize tuner based on task and model type
        if is_regression:
            from PineBioML.model.supervised.Regression import RandomForest_tuner as RFR, XGBoost_tuner as XGBR
            if model_type.lower() == "xgboost": 
                tuner = XGBR(n_try=15, n_cv=3, target="r2")
            else: 
                tuner = RFR(n_try=15, n_cv=3, target="r2")
        else:
            from PineBioML.model.supervised.Classification import RandomForest_tuner, XGBoost_tuner
            if model_type.lower() == "xgboost": 
                tuner = XGBoost_tuner(n_try=15, n_cv=3, target="mcc")
            else: 
                tuner = RandomForest_tuner(n_try=15, n_cv=3, target="mcc")
        
        # Direct training (no Pine wrapper needed)
        debug_log(f"Starting {model_type} training with Optuna...")
        tuner.fit(x, y)
        debug_log("Training complete!")
        
        # Save the fitted tuner
        best_model_path = os.path.join(STATE_DIR, "best_model.joblib")
        import joblib
        joblib.dump(tuner, best_model_path)
        debug_log(f"Model saved to {best_model_path}")
        
        # Get metrics from tuner's Optuna study
        best_score = tuner.study.best_value
        n_trials = len(tuner.study.trials)
        
        # Get best parameters
        best_params = tuner.study.best_params
        params_str = "\n".join([f"  - {k}: {v}" for k, v in list(best_params.items())[:5]])
        
        return (f"### ✅ {task_name} Training Complete\n\n"
                f"**Model**: {model_type.replace('_', ' ').title()}\n"
                f"**Target Variable**: {target_column}\n"
                f"**Dataset**:\n"
                f"  - Samples: {len(x)}\n"
                f"  - Features: {len(x.columns)}\n\n"
                f"**Optimization Results**:\n"
                f"  - Best Score: {best_score:.4f}\n"
                f"  - Trials Completed: {n_trials}\n\n"
                f"**Top Parameters**:\n{params_str}\n\n"
                f"Model saved successfully. Use `predict_patient_outcome` for predictions!")
                
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        debug_log(f"Training Error:\n{error_detail}")
        return (f"❌ Training failed: {str(e)}\n\n"
                f"**Common causes**:\n"
                f"1. Insufficient data (need >= 10 samples)\n"
                f"2. Missing `optuna` package (run: pip install optuna)\n"
                f"3. Invalid target column\n\n"
                f"Check server logs for details.")

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
