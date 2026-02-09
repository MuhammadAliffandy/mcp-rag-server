"""
Similarity Ranking Functions for EXPRAG Workflow (Phase 2)

This module implements Jaccard similarity-based ranking for patient cohort selection.
"""

from typing import List, Tuple, Dict
from src.core.patient_profile import PatientProfile


def calculate_similarity(profile_a: PatientProfile, profile_b: PatientProfile) -> float:
    """
    Calculate mixed similarity between two patient profiles.
    
    Combines Jaccard similarity for categorical data with 
    Gaussian similarity for numerical clinical metrics.
    
    Weights:
    - Categorical (70%): Indication, Polyps, Procedures, Dz Location
    - Numerical (30%): Age, sum_pMayo, Hb, Rectal Bleed
    """
    dict_a = profile_a.to_comparison_dict()
    dict_b = profile_b.to_comparison_dict()
    
    # 1. Categorical Similarity (Jaccard)
    cat_keys = ['indication', 'polyp_history', 'procedures', 'dz_location']
    all_a = set()
    all_b = set()
    for key in cat_keys:
        all_a.update(dict_a.get(key, set()))
        all_b.update(dict_b.get(key, set()))
    
    cat_sim = 0.0
    union = len(all_a | all_b)
    if union > 0:
        cat_sim = len(all_a & all_b) / union
        
    # 2. Numerical Similarity (Gaussian)
    # Define scale (sigma) for each metric
    num_metrics = {
        'age': 15.0,        # 15 years difference is significant
        'sum_pmayo': 2.0,   # 2 points difference is significant
        'hb': 2.0,          # 2 g/dL difference is significant
        'rectal_bleed': 1.0 # 1 point difference is significant
    }
    
    num_sims = []
    import math
    
    for key, sigma in num_metrics.items():
        val_a = dict_a.get(key)
        val_b = dict_b.get(key)
        
        if val_a is not None and val_b is not None:
            diff = abs(float(val_a) - float(val_b))
            sim = math.exp(-(diff**2) / (2 * sigma**2))
            num_sims.append(sim)
            
    num_sim = sum(num_sims) / len(num_sims) if num_sims else 1.0
    
    # Combined Weighted Similarity
    return (0.7 * cat_sim) + (0.3 * num_sim)


def rank_cohort(
    input_profile: PatientProfile,
    database: List[PatientProfile],
    top_k: int = 10,
    min_threshold: float = 0.1
) -> List[Tuple[str, float]]:
    """
    Rank all profiles in database by similarity to input profile.
    
    This function simulates the coarse ranking step that would work against
    a real patient database (SQL/NoSQL). It returns the Top-K most similar
    case IDs with their similarity scores.
    
    Args:
        input_profile: The query patient profile
        database: List of historical patient profiles to compare against
        top_k: Maximum number of similar cases to return
        min_threshold: Minimum similarity score to include (filter weak matches)
    
    Returns:
        List of tuples (case_id, similarity_score) sorted by score descending
        
    Example:
        >>> input_patient = PatientProfile(case_id="new", indication="IBD", ...)
        >>> db = [profile1, profile2, profile3]
        >>> cohort = rank_cohort(input_patient, db, top_k=5)
        >>> # Returns: [("analysis_id_1", 0.85), ("analysis_id_2", 0.42), ...]
    
    Critical Note:
        Each returned tuple preserves the case_id so downstream systems
        can retrieve the full patient record from the database.
    """
    # Calculate similarity for each profile in database
    scores = []
    
    for db_profile in database:
        similarity = calculate_similarity(input_profile, db_profile)
        
        # Only include if above threshold
        if similarity >= min_threshold:
            scores.append((db_profile.case_id, similarity))
    
    # Sort by similarity descending
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top K
    return scores[:top_k]


def get_cohort_ids(
    input_profile: PatientProfile,
    database: List[PatientProfile],
    top_k: int = 10,
    min_threshold: float = 0.1
) -> List[str]:
    """
    Convenience function to get just the case IDs (without scores).
    
    Args:
        input_profile: The query patient profile
        database: List of historical patient profiles
        top_k: Maximum number of IDs to return
        min_threshold: Minimum similarity threshold
    
    Returns:
        List of case_ids suitable for Pinecone $in filter
        
    Example:
        >>> ids = get_cohort_ids(input_patient, db)
        >>> # Returns: ["analysis_id_1", "analysis_id_2", "analysis_id_3"]
        >>> # Can be used directly: filter={'case_id': {'$in': ids}}
    """
    ranked = rank_cohort(input_profile, database, top_k, min_threshold)
    return [case_id for case_id, score in ranked]
