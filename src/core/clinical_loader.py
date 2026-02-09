import pandas as pd
from typing import List
from src.core.patient_profile import PatientProfile

def load_clinical_excel(file_path: str) -> List[PatientProfile]:
    """
    Load clinical data from Excel and convert to PatientProfile objects.
    
    Mapping Strategy:
    - patient -> case_id
    - age_at_cpy -> age
    - sum_pMayo -> sum_pmayo
    - dz_location_active only -> dz_location
    - hb -> hb
    """
    try:
        df = pd.read_excel(file_path)
        profiles = []
        
        for _, row in df.iterrows():
            # Basic validation: must have patient ID
            if pd.isna(row.get('patient')):
                continue
                
            profile = PatientProfile(
                case_id=str(row.get('patient')),
                indication="ibd",  # Default for this dataset
                age=float(row.get('age_at_cpy')) if not pd.isna(row.get('age_at_cpy')) else None,
                sum_pmayo=float(row.get('sum_pMayo')) if not pd.isna(row.get('sum_pMayo')) else None,
                dz_location=str(row.get('dz_location_active only')) if not pd.isna(row.get('dz_location_active only')) else None,
                hb=float(row.get('hb')) if not pd.isna(row.get('hb')) else None,
                # Polyp history and procedures can be inferred or extracted if needed
                polyp_history=[],
                procedures=["colonoscopy"]
            )
            profiles.append(profile)
            
        print(f"✅ Successfully loaded {len(profiles)} clinical profiles from {file_path}")
        return profiles
        
    except Exception as e:
        print(f"❌ Error loading Excel: {e}")
        return []
