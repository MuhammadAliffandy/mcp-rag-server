@mcp.tool()
def clean_medical_data(
    imputation_method: str = "knn",
    outlier_removal: bool = True,
    outlier_method: str = "iqr",
    missing_threshold: float = 0.33
) -> str:
    """
    Clean medical data by imputing missing values and removing outliers.
    
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
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
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
            from PineBioML.preprocessing.impute import knn_imputer, simple_imputer, iterative_imputer
            
            if imputation_method == "knn":
                imputer = knn_imputer(threshold=missing_threshold, n_neighbor=5)
            elif imputation_method == "iterative":
                imputer = iterative_imputer(threshold=missing_threshold, max_iter=10)
            elif imputation_method in ["median", "mean"]:
                imputer = simple_imputer(threshold=missing_threshold, strategy=imputation_method)
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
                    
                    # Replace outliers with NaN, then impute again
                    df.loc[outlier_mask, col] = np.nan
                
                elif outlier_method == "zscore":
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    outlier_mask = z_scores > 3
                    outliers_removed += outlier_mask.sum()
                    df.loc[outlier_mask, col] = np.nan
            
            if outliers_removed > 0:
                # Re-impute outliers
                if imputation_method == "knn":
                    imputer = knn_imputer(threshold=1.0, n_neighbor=5)
                else:
                    imputer = simple_imputer(threshold=1.0, strategy="median")
                
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


