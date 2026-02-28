"""Few-shot examples for orchestration to teach LLM proper tool selection."""


def get_few_shot_examples() -> str:
    """Returns comprehensive few-shot examples for agentic routing with reasoning."""
    # NOTE: Using a regular string (not triple-quoted) to avoid f-string nesting issues
    # when this content is injected into orchestration.py's f-string.
    return (
        "# FEW-SHOT EXAMPLES (Mental Models for the Agent):\n\n"

        "## Example 1: Basic Distribution Plot (Styling Extraction)\n"
        'User: "plot distribution of age with dark theme"\n'
        "Output: {\n"
        '  "answer": "Generating distribution plot for age using the dark theme.",\n'
        '  "thoughts": "User wants a visualization. Detected \'age\' as target and \'dark theme\' as style.",\n'
        '  "tasks": [{ "tool": "generate_medical_plot", "args": { "plot_type": "distribution", "target_column": "age", "styling": "{\\"style\\": {\\"theme\\": \\"dark\\"}}" } }]\n'
        "}\n\n"

        "## Example 2: Indonesian + Specific ID Analysis (Hybrid Search)\n"
        'User: "Coba analisis kenapa pasien ID 123 bisa sakit?"\n'
        "Output: {\n"
        '  "answer": "Saya akan mencari data klinis pasien 123 dan menganalisisnya berdasarkan referensi medis.",\n'
        '  "thoughts": "User asking for specific patient (ID 123) analysis. Requires exact data lookup AND medical reasoning (RAG).",\n'
        '  "tasks": [\n'
        '    { "tool": "exact_identifier_search", "args": { "query": "123" } },\n'
        '    { "tool": "query_medical_rag", "args": { "question": "Kenapa pasien 123 sakit?" } }\n'
        "  ]\n"
        "}\n\n"

        "## Example 3: Internal SOP Query (ONLY for docs stored in the system)\n"
        'User: "what are the procedures in our internal SOP document?"\n'
        "Output: {\n"
        '  "answer": "I will search our internal SOP documents for the procedures.",\n'
        '  "thoughts": "Question is about OUR INTERNAL uploaded SOP docs. Use query_medical_rag for internal docs.",\n'
        '  "tasks": [{ "tool": "query_medical_rag", "args": { "question": "internal SOP procedures" } }]\n'
        "}\n\n"

        "## Example 4: Session Data Overview\n"
        'User: "jelaskan isi file yang baru saya upload"\n'
        "Output: {\n"
        '  "answer": "Baik, saya akan membuat ringkasan statistik dari data yang Anda unggah.",\n'
        '  "thoughts": "User refers to uploaded file (session data). Calling get_data_context.",\n'
        '  "tasks": [{ "tool": "get_data_context", "args": {} }]\n'
        "}\n\n"

        "## Example 5: Clustering Request\n"
        'User: "bisa tolong kelompokkan pasien berdasarkan kemiripan?"\n'
        "Output: {\n"
        '  "answer": "Saya akan menjalankan analisis UMAP untuk melihat pola pengelompokan pasien.",\n'
        '  "thoughts": "User said kelompokkan (clustering). UMAP is the correct tool.",\n'
        '  "tasks": [{ "tool": "run_umap_analysis", "args": {} }]\n'
        "}\n\n"

        "## Example 6: PLS-DA Separation\n"
        'User: "show me separation between healthy and sick groups"\n'
        "Output: {\n"
        '  "answer": "I\'ll run PLS-DA to visualize the separation between the healthy and sick classes.",\n'
        '  "thoughts": "User asks for separation between groups. PLS-DA is the correct tool.",\n'
        '  "tasks": [{ "tool": "run_pls_analysis", "args": {} }]\n'
        "}\n\n"

        "## Example 7: Biomarker Discovery\n"
        'User: "what are the most significant markers?"\n'
        "Output: {\n"
        '  "answer": "Running a Volcano Plot analysis to discover significant biomarkers.",\n'
        '  "thoughts": "User asking for significant markers. Volcano Plot (discover_markers) is designed for this.",\n'
        '  "tasks": [{ "tool": "discover_markers", "args": {} }]\n'
        "}\n\n"

        "## Example 8: Data Cleaning\n"
        'User: "Clean data pakai KNN imputation"\n'
        "Output: {\n"
        '  "answer": "Saya akan membersihkan data menggunakan KNN imputation.",\n'
        '  "thoughts": "User wants data cleaning in Indonesian.",\n'
        '  "tasks": [\n'
        '    { "tool": "extract_data_from_rag", "args": { "query": "clinical data", "save_to_session": true } },\n'
        '    { "tool": "clean_medical_data", "args": { "imputation_method": "knn", "outlier_removal": true } }\n'
        "  ]\n"
        "}\n\n"

        "## Example 9: Data Overview (English)\n"
        'User: "Show me a data overview"\n'
        "Output: {\n"
        '  "answer": "I will prepare a comprehensive data overview including PCA, PLS-DA, and UMAP analysis.",\n'
        '  "thoughts": "User wants data overview.",\n'
        '  "tasks": [\n'
        '    { "tool": "extract_data_from_rag", "args": { "query": "clinical data", "save_to_session": true } },\n'
        '    { "tool": "generate_data_overview", "args": { "is_classification": true } }\n'
        "  ]\n"
        "}\n\n"

        "## Example 10: General Chat\n"
        'User: "halo, kamu siapa?"\n'
        "Output: {\n"
        '  "answer": "Halo! Saya asisten analisis medis PineBioML.",\n'
        '  "thoughts": "General greeting. No data task required.",\n'
        '  "tasks": []\n'
        "}\n\n"

        "## Example 11: Exact Code Search\n"
        'User: "cari sample dengan kode ACC-2024-X"\n'
        "Output: {\n"
        '  "answer": "Mencari sampel dengan kode spesifik ACC-2024-X.",\n'
        '  "thoughts": "User provided a specific alphanumeric code. Using exact identifier search.",\n'
        '  "tasks": [{ "tool": "exact_identifier_search", "args": { "query": "ACC-2024-X" } }]\n'
        "}\n\n"

        "## Example 12: Clinical Experience Reasoning (EXPRAG Hybrid)\n"
        'User: "Pasien 60 tahun dengan Mayo Score 9 dan riwayat UC. Apa rekomendasi follow-up berdasarkan pengalaman pasien serupa?"\n'
        "Output: {\n"
        '  "answer": "Saya akan mencari pasien dengan profil klinis serupa untuk membandingkan tindakan medis.",\n'
        '  "thoughts": "User asks for treatment based on similar patients. Priority for query_exprag_hybrid.",\n'
        '  "tasks": [{ "tool": "query_exprag_hybrid", "args": { "question": "rekomendasi follow-up", "patient_data": "{\\"age\\": 60, \\"sum_pmayo\\": 9}" } }]\n'
        "}\n\n"

        "## Example 13: PCA Plot\n"
        'User: "Buat PCA plot warnai Disease"\n'
        "Output: {\n"
        '  "answer": "Saya akan membuat PCA plot dengan pewarnaan berdasarkan Disease.",\n'
        '  "thoughts": "User wants PCA in Indonesian.",\n'
        '  "tasks": [{ "tool": "generate_medical_plot", "args": { "plot_type": "pca", "target_column": "Disease" } }]\n'
        "}\n\n"

        "## Example 14: Model Training\n"
        'User: "Train a RandomForest model to predict Disease"\n'
        "Output: {\n"
        '  "answer": "I will train a RandomForest model to predict Disease.",\n'
        '  "thoughts": "User wants ML training.",\n'
        '  "tasks": [\n'
        '    { "tool": "extract_data_from_rag", "args": {} },\n'
        '    { "tool": "train_medical_model", "args": { "target_column": "Disease", "model_type": "RandomForest" } }\n'
        "  ]\n"
        "}\n\n"

        "## Example 15: Grouped Distribution\n"
        'User: "tampilkan distribusi umur berdasarkan jenis kelamin"\n'
        "Output: {\n"
        '  "answer": "Saya akan membuat plot distribusi umur yang dikelompokkan berdasarkan jenis kelamin.",\n'
        '  "thoughts": "User wants distribution by category. Using hue_column for grouping.",\n'
        '  "tasks": [{ "tool": "generate_medical_plot", "args": { "plot_type": "distribution", "target_column": "age", "hue_column": "sex" } }]\n'
        "}\n\n"

        "## Example 16: Box Plot for Comparison\n"
        'User: "compare CRP levels between disease groups using a box plot"\n'
        "Output: {\n"
        '  "answer": "I will generate a box plot to compare CRP levels across different disease groups.",\n'
        '  "thoughts": "User explicitly requested a box plot for comparison.",\n'
        '  "tasks": [{ "tool": "generate_medical_plot", "args": { "plot_type": "box", "target_column": "crp", "hue_column": "disease" } }]\n'
        "}\n\n"

        "## ==================================================================\n"
        "## CRITICAL: EXTERNAL GUIDELINES ROUTING (Examples 17-21)\n"
        "## RULE: If user mentions ANY guideline body OR asks for treatment protocol/escalation,\n"
        "## ALWAYS use query_external_guidelines — NEVER query_medical_rag for these!\n"
        "## ==================================================================\n\n"

        "## Example 17: External Guideline — ACG Named\n"
        'User: "Patient P-045 has MES 3 Severe Activity. What is the recommended escalation per ACG guidelines?"\n'
        "Output: {\n"
        '  "answer": "I will fetch the latest ACG guidelines for severe ulcerative colitis (MES 3) and provide the recommended escalation protocol.",\n'
        '  "thoughts": "User explicitly mentions ACG (external guideline body). MUST use query_external_guidelines. query_medical_rag only searches internal docs and would return nothing for ACG web content.",\n'
        '  "tasks": [{ "tool": "query_external_guidelines", "args": { "question": "recommended escalation for severe ulcerative colitis MES 3 per ACG guidelines", "patient_context": "MES 3, Severe Activity" } }]\n'
        "}\n\n"

        "## Example 18: External Guideline — ECCO Named\n"
        'User: "What are ECCO recommendations for induction therapy in moderate-to-severe UC?"\n'
        "Output: {\n"
        '  "answer": "I will retrieve the ECCO guidelines for induction therapy in moderate-to-severe ulcerative colitis from the web.",\n'
        '  "thoughts": "ECCO is an external guideline body. Must use query_external_guidelines.",\n'
        '  "tasks": [{ "tool": "query_external_guidelines", "args": { "question": "ECCO recommendations induction therapy moderate-to-severe ulcerative colitis" } }]\n'
        "}\n\n"

        "## Example 19: External Guideline — ADA Diabetes\n"
        'User: "What protocol should I follow for a patient with HbA1c 11.2? ADA guidelines?"\n'
        "Output: {\n"
        '  "answer": "I will look up the ADA guidelines for managing HbA1c 11.2 and provide the recommended treatment protocol.",\n'
        '  "thoughts": "ADA is an external guideline body for diabetes. Must use query_external_guidelines.",\n'
        '  "tasks": [{ "tool": "query_external_guidelines", "args": { "question": "ADA guidelines management HbA1c 11.2 diabetes treatment protocol", "patient_context": "HbA1c 11.2" } }]\n'
        "}\n\n"

        "## Example 20: External Guideline — Indonesian\n"
        'User: "Panduan tatalaksana kolitis ulseratif berat berdasarkan ECCO dan ACG?"\n'
        "Output: {\n"
        '  "answer": "Saya akan mengambil panduan tatalaksana kolitis ulseratif berat dari ECCO dan ACG melalui sumber eksternal.",\n'
        '  "thoughts": "User mentions ECCO and ACG (both external bodies). Language is Indonesian. Must use query_external_guidelines.",\n'
        '  "tasks": [{ "tool": "query_external_guidelines", "args": { "question": "severe ulcerative colitis ECCO ACG guidelines treatment escalation", "patient_context": "severe activity" } }]\n'
        "}\n\n"

        "## Example 21: External Guideline — No Named Authority, But Treatment Protocol\n"
        'User: "What is the standard treatment escalation for a patient with severe IBD not responding to steroids?"\n'
        "Output: {\n"
        '  "answer": "I will retrieve current international guidelines for rescue therapy in steroid-refractory severe IBD.",\n'
        '  "thoughts": "User asks treatment escalation — this is external knowledge. Must use query_external_guidelines, not internal docs.",\n'
        '  "tasks": [{ "tool": "query_external_guidelines", "args": { "question": "treatment escalation steroid-refractory severe IBD rescue therapy guidelines" } }]\n'
        "}\n\n"

        "## ROUTING DECISION TREE:\n"
        "- User mentions ACG, ECCO, WHO, NICE, ESC, AHA, IDSA, ADA, ASCO, ESMO, GOLD, KDIGO, EULAR, AAN, ACOG... -> ALWAYS use query_external_guidelines\n"
        "- User asks for treatment escalation/protocol/recommendation for a clinical severity -> use query_external_guidelines\n"
        "- User asks about docs stored in our INTERNAL uploaded files -> use query_medical_rag\n"
        "- User asks for similar patient historical experience -> use query_exprag_hybrid\n"
    )
