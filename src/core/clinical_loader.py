import pandas as pd
from typing import List
from src.core.patient_profile import PatientProfile

import pandas as pd
import os
from typing import List, Optional
from src.core.patient_profile import PatientProfile

def load_patient_data(file_path: str) -> List[PatientProfile]:
    """
    Unified loader for clinical data (Excel or CSV).
    Automatically detects format and maps to PatientProfile.
    """
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        return []

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.xlsx' or ext == '.xls':
            df = pd.read_excel(file_path)
        elif ext == '.csv':
            df = pd.read_csv(file_path)
        else:
            print(f"⚠️ Unsupported format: {ext}")
            return []

        profiles = []
        for _, row in df.iterrows():
            # Flexible Mapping Logic
            def get_val(keys, default=None):
                for k in keys:
                    if k in row and not pd.isna(row[k]):
                        return row[k]
                return default

            case_id = str(get_val(['patient', 'hadm_id', 'case_id'], 'unknown'))
            if case_id == 'unknown': continue

            # Extract fields with multi-key support (mapping standard EHR keys)
            profile = PatientProfile(
                case_id=case_id,
                indication=str(get_val(['indication', 'diagnosis'], 'screening')).lower(),
                age=float(get_val(['age', 'age_at_cpy', 'usia'])) if get_val(['age', 'age_at_cpy', 'usia']) else None,
                sum_pmayo=float(get_val(['sum_pmayo', 'mayo', 'mayo_score'])) if get_val(['sum_pmayo', 'mayo', 'mayo_score']) else None,
                dz_location=str(get_val(['dz_location', 'dz_location_active only', 'location', 'lokasi'])) if get_val(['dz_location', 'dz_location_active only', 'location', 'lokasi']) else None,
                hb=float(get_val(['hb', 'hemoglobin', 'hbg'])) if get_val(['hb', 'hemoglobin', 'hbg']) else None,
                rectal_bleed=float(get_val(['rectal_bleed', 'bleeding'])) if get_val(['rectal_bleed', 'bleeding']) else None,
                polyp_history=str(get_val(['polyp_history', 'polyp'], '')).split(',') if get_val(['polyp_history', 'polyp']) else [],
                procedures=str(get_val(['procedures', 'procedure'], 'colonoscopy')).split(',')
            )
            profiles.append(profile)

        print(f"✅ Successfully loaded {len(profiles)} clinical profiles from {file_path}")
        return profiles

    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return []

def load_clinical_excel(file_path: str) -> List[PatientProfile]:
    """Legacy wrapper for backward compatibility."""
    return load_patient_data(file_path)
