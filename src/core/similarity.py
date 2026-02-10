"""
Advanced Similarity Ranking Engine (EXPRAG Full Power)
Implements Multi-Axis scoring and Quintile Binning for experience diversity.
"""

from typing import List, Tuple, Dict, Any
import math
from math import ceil
from src.core.patient_profile import PatientProfile


def calculate_jaccard(set_a: set, set_b: set) -> float:
    """Calculate standard Jaccard similarity between two sets."""
    union = len(set_a | set_b)
    if union == 0: return 0.0
    return len(set_a & set_b) / union


def calculate_gaussian(val_a: float, val_b: float, sigma: float) -> float:
    """Calculate Gaussian similarity for numerical values."""
    if val_a is None or val_b is None: return 1.0 
    diff = abs(float(val_a) - float(val_b))
    return math.exp(-(diff**2) / (2 * sigma**2))


def get_axis_scores(profile_a: PatientProfile, profile_b: PatientProfile) -> Dict[str, float]:
    """
    Calculate similarity across independent medical dimensions.
    Matching the granularity of original EXPRAG: diagnoses, procedures, and prescriptions.
    """
    dict_a = profile_a.to_comparison_dict()
    dict_b = profile_b.to_comparison_dict()
    
    # Use standard EXPRAG axis mapping
    # Note: prescription logic is usually categorical in these datasets
    return {
        'diagnoses': calculate_jaccard(dict_a.get('polyp_history', set()) | dict_a.get('indication', set()), 
                                     dict_b.get('polyp_history', set()) | dict_b.get('indication', set())),
        'procedures': calculate_jaccard(dict_a.get('procedures', set()), dict_b.get('procedures', set())),
        # We use a mix for prescriptions if available, or fallback to clinical metrics
        'clinical': calculate_gaussian(dict_a.get('age'), dict_b.get('age'), 15.0) * 0.5 + 
                   calculate_gaussian(dict_a.get('sum_pmayo'), dict_b.get('sum_pmayo'), 2.0) * 0.5
    }


def rank_cohort_with_diversity(
    input_profile: PatientProfile,
    database: List[PatientProfile],
    top_n_per_bin: int = 2,
    min_threshold: float = 0.05
) -> List[str]:
    """
    Experimental Diversity Selection (Quintile Binning).
    Steps:
    1. Calculate similarities for all candidates.
    2. Sort them into 5 bins based on score (0-20, 20-40, ..., 80-100 quantile).
    3. Select Top-N from EACH bin.
    """
    scored_candidates = []
    for db_profile in database:
        axis_scores = get_axis_scores(input_profile, db_profile)
        # Combine axes into a single score for binning (equal weight by default)
        total_score = sum(axis_scores.values()) / len(axis_scores)
        
        if total_score >= min_threshold:
            scored_candidates.append((db_profile.case_id, total_score))
    
    if not scored_candidates: return []
    
    # Sort by score descending
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Split into Quintile Bins
    m = len(scored_candidates)
    bin_size = ceil(m / 5)
    bins = []
    for i in range(5):
        start = i * bin_size
        end = m if i == 4 else (i + 1) * bin_size
        bins.append(scored_candidates[start:end])
    
    # Select Top-N from each bin to ensure diversity
    diverse_cohort = []
    for bin_list in bins:
        diverse_cohort.extend([item[0] for item in bin_list[:top_n_per_bin]])
        
    return diverse_cohort


def get_cohort_ids(
    input_profile: PatientProfile,
    database: List[PatientProfile],
    top_k: int = 10,
    use_diversity: bool = False
) -> List[str]:
    """Unified entry point for cohort identification."""
    if use_diversity:
        # For diversity, top_k is approximate (5 bins * N per bin)
        n_per_bin = ceil(top_k / 5)
        return rank_cohort_with_diversity(input_profile, database, top_n_per_bin=n_per_bin)
    
    # Standard ranking logic
    all_scores = []
    for db_profile in database:
        axis_scores = get_axis_scores(input_profile, db_profile)
        total_score = sum(axis_scores.values()) / len(axis_scores)
        all_scores.append((db_profile.case_id, total_score))
        
    all_scores.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in all_scores[:top_k]]
