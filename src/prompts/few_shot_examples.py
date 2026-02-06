"""Few-shot examples for orchestration to teach LLM proper tool selection."""

def get_few_shot_examples() -> str:
    """Returns comprehensive few-shot examples for agentic routing with reasoning."""
    return """
# FEW-SHOT EXAMPLES (Mental Models for the Agent):

## Example 1: Basic Distribution Plot (Styling Extraction)
User: "plot distribution of age with dark theme"
Output: {
  "answer": "Generating distribution plot for age using the dark theme.",
  "thoughts": "User wants a visualization. Detected 'age' as target and 'dark theme' as style.",
  "tasks": [
    {
      "tool": "generate_medical_plot",
      "args": {
        "plot_type": "distribution",
        "target_column": "age",
        "styling": { "theme": "dark" }
      }
    }
  ]
}

## Example 2: Indonesian + Specific ID Analysis (Hybrid Search)
User: "Coba analisis kenapa pasien ID 123 bisa sakit?"
Output: {
  "answer": "Saya akan mencari data klinis pasien 123 dan menganalisisnya berdasarkan referensi medis.",
  "thoughts": "User asking for specific patient (ID 123) analysis. Requires exact data lookup AND medical reasoning (RAG).",
  "tasks": [
    { "tool": "exact_identifier_search", "args": { "query": "123" } },
    { "tool": "query_medical_rag", "args": { "question": "Kenapa pasien 123 sakit? Analisis indikator klinisnya." } }
  ]
}

## Example 3: Internal Knowledge Query (SOP/Guidelines)
User: "what are the standard protocols for sample handling?"
Output: {
  "answer": "I will search our internal SOPs for sample handling protocols.",
  "thoughts": "Question is about general procedures/SOPs, not specific session data. Routing to RAG.",
  "tasks": [
    { "tool": "query_medical_rag", "args": { "question": "standard protocols for sample handling" } }
  ]
}

## Example 4: Session Data Overview (External Data)
User: "jelaskan isi file yang baru saya upload"
Output: {
  "answer": "Baik, saya akan membuat ringkasan statistik dari data yang Anda unggah.",
  "thoughts": "User refers to 'uploaded file' (session data). calling get_data_context for summary.",
  "tasks": [
    { "tool": "get_data_context", "args": {} }
  ]
}

## Example 5: Clustering Request (Implicit Keyword)
User: "bisa tolong kelompokkan pasien berdasarkan kemiripan?"
Output: {
  "answer": "Saya akan menjalankan analisis UMAP untuk melihat pola pengelompokan (clustering) pasien.",
  "thoughts": "User said 'kelompokkan' (grouping/clustering). The correct statistical tool for unsupervised clustering is UMAP.",
  "tasks": [
    { "tool": "run_umap_analysis", "args": {} }
  ]
}

## Example 6: Separation/Classification Request (Implicit Keyword)
User: "show me separation between healthy and sick groups"
Output: {
  "answer": "I'll run PLS-DA to visualize the separation between the healthy and sick classes.",
  "thoughts": "User asks for 'separation' between groups. This implies supervised analysis. PLS-DA is the correct tool.",
  "tasks": [
    { "tool": "run_pls_analysis", "args": {} }
  ]
}

## Example 7: Complex Styling (Custom JSON Construction)
User: "buat scatter plot Age vs BMI, warnanya merah, judul font 18"
Output: {
  "answer": "Siap, membuat scatter plot Age vs BMI dengan warna merah dan judul ukuran besar.",
  "thoughts": "Visualization request with multiple styles. Mapping 'merah'->'bar_color': 'red', 'font 18'->'title_size': 18.",
  "tasks": [
    {
      "tool": "generate_medical_plot",
      "args": {
        "plot_type": "scatter",
        "x": "Age",
        "y": "BMI",
        "styling": {
            "bar_color": "red",
            "title_size": 18
        }
      }
    }
  ]
}

## Example 8: Multi-Patient Specific Lookup (Not Statistics)
User: "bandingkan data klinis pasien 1, 2, dan 3"
Output: {
  "answer": "Saya akan mengambil data untuk pasien 1, 2, dan 3 untuk dibandingkan.",
  "thoughts": "User lists specific IDs (small number). This is a retrieval task, not a statistical group analysis.",
  "tasks": [
    { "tool": "query_medical_rag", "args": { "question": "Compare clinical data for patients 1, 2, and 3" } }
  ]
}

## Example 9: Confirmation / Follow-up (Context Aware)
User: "oke jalankan itu"
Output: {
  "answer": "Baik, segera saya proses.",
  "thoughts": "User confirmed previous suggestion. Checking history... Previous turn suggested PCA. Executing PCA.",
  "tasks": [
    { "tool": "generate_medical_plot", "args": { "plot_type": "pca" } }
  ]
}

## Example 10: General Chat (No Tool Needed)
User: "halo, kamu siapa?"
Output: {
  "answer": "Halo! Saya adalah asisten analisis medis PineBioML. Saya bisa membantu Anda menganalisis data pasien, visualisasi, atau mencari info SOP.",
  "thoughts": "General greeting. No data task required.",
  "tasks": []
}

## Example 11: Error Correction / Refinement
User: "bukan, maksud saya yang heatmap"
Output: {
  "answer": "Maaf atas kekeliruan sebelumnya. Saya akan buatkan Heatmap korelasinya.",
  "thoughts": "User corrected intent. Switching tool from previous (likely PCA/Scatter) to Heatmap.",
  "tasks": [
    { "tool": "run_correlation_heatmap", "args": {} }
  ]
}

## Example 12: Exact Code Search (Accession ID)
User: "cari sample dengan kode ACC-2024-X"
Output: {
  "answer": "Mencari sampel dengan kode spesifik ACC-2024-X.",
  "thoughts": "User provided a specific alphanumeric code. Using exact identifier search.",
  "tasks": [
    { "tool": "exact_identifier_search", "args": { "query": "ACC-2024-X" } }
  ]
}

## Example 13: Feature Discovery (Biomarkers)
User: "what are the most significant markers?"
Output: {
  "answer": "Running a Volcano Plot analysis to discover significant biomarkers in the dataset.",
  "thoughts": "User asking for 'significant markers'. The Volcano Plot tool (discover_markers) is designed for this.",
  "tasks": [
    { "tool": "discover_markers", "args": {} }
  ]
}

## Example 14: Data Cleaning Request
User: "datanya banyak yang kosong, tolong perbaiki"
Output: {
  "answer": "Saya akan melakukan pembersihan data (imputasi) untuk mengisi nilai yang kosong.",
  "thoughts": "User mentions missing data/empty values. Calling cleaning tool.",
  "tasks": [
    { "tool": "clean_medical_data", "args": {} }
  ]
}

## Example 15: Data Cleaning with RAG Extraction (Indonesian)
User: "Clean data pakai KNN imputation"
Output: {
  "answer": "Saya akan membersihkan data menggunakan KNN imputation untuk menangani missing values.",
  "thoughts": "User wants data cleaning in Indonesian. Mirroring language.",
  "tasks": [
    { "tool": "extract_data_from_rag", "args": {"query": "clinical data", "save_to_session": true} },
    { "tool": "clean_medical_data", "args": {"imputation_method": "knn", "outlier_removal": true} }
  ]
}

## Example 16: Data Overview (English)
User: "Show me a data overview"
Output: {
  "answer": "I will prepare a comprehensive data overview including PCA, PLS-DA, and UMAP analysis.",
  "thoughts": "User wants data overview in English. Mirroring language.",
  "tasks": [
    { "tool": "extract_data_from_rag", "args": {"query": "clinical data", "save_to_session": true} },
    { "tool": "generate_data_overview", "args": {"target_column": null, "is_classification": true} }
  ]
}

## Example 17: Biomarker Discovery (Indonesian)
User: "Cari biomarkers untuk Disease vs Healthy"
Output: {
  "answer": "Saya akan menjalankan analisis Volcano Plot untuk mengidentifikasi biomarker yang signifikan membedakan kelompok Disease dan Healthy.",
  "thoughts": "User wants biomarker discovery in Indonesian. Mirroring language.",
  "tasks": [
    { "tool": "extract_data_from_rag", "args": {"query": "clinical data"} },
    { "tool": "discover_markers", "args": {"target_column": "Disease", "p_value_threshold": 0.05} }
  ]
}

## Example 18: Model Training (English)
User: "Train a RandomForest model to predict Disease"
Output: {
  "answer": "I will train a RandomForest model to predict Disease using hyperparameter optimization.",
  "thoughts": "User wants ML training in English. Mirroring language.",
  "tasks": [
    { "tool": "extract_data_from_rag", "args": {} },
    { "tool": "train_medical_model", "args": {"target_column": "Disease", "model_type": "RandomForest"} }
  ]
}

## Example 19: PCA Plot (Indonesian)
User: "Buat PCA plot warnai Disease"
Output: {
  "answer": "Saya akan membuat PCA plot dengan pewarnaan berdasarkan Disease untuk melihat pola sebaran data.",
  "thoughts": "User wants PCA in Indonesian. Mirroring language.",
  "tasks": [
    { "tool": "extract_data_from_rag", "args": {"query": "clinical data"} },
    { "tool": "generate_medical_plot", "args": {"plot_type": "pca", "target_column": "Disease", "data_source": "session"} }
  ]
}

## Example 20: Medical Knowledge Query (RAG Tool - NOT PineBioML)
User: "What is diabetes?"
Output: {
  "answer": "I will search our medical knowledge base for information about diabetes.",
  "thoughts": "User asking for medical definition/knowledge. This is RAG query_medical_rag, NOT PineBioML.",
  "tasks": [
    {
      "tool": "query_medical_rag",
      "args": {
        "question": "What is diabetes? Provide definition and clinical overview."
      }
    }
  ]
}
"""