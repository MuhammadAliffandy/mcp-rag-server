# New Knowledge & Task Directives

This document encapsulates the latest task directives and knowledge updates provided by the user modifying the ColonoSense system on March 24, 2026.

## 1. Core Logic & Data
Calculations are strictly bound to Excel spreadsheet extractions:
- **Total Mayo Score** = `Partial Mayo Score (UC_baseline)` + `MAX(MES segments in UC_cpy)`.
- **Prognostic Factors**: Explicitly checks `age < 40`, `extent=3`, `MES=3`, `CRP > 1`, `Albumin < 3.5`, `Steroid Use`, and `PSC` status from `UC_baseline`.

## 2. Updated Hierarchy of Evidence
The system adheres to a strict Guard RAG logic when referencing Medical SOPs. The explicit priority order is:
1. **[Tier 1]** Global guidelines.
2. **[Tier 2]** Local guidelines.
3. **[Tier 4]** Pivotal trials.
4. **[Tier 3]** Meta-analyses.
*Note: This specific order (Tier 1 -> Tier 2 -> Tier 4 -> Tier 3) was a direct update from the user, elevating Pivotal Trial importance above Meta-analyses in the output format.*

## 3. Medication Adjustment Logic Update
Medication adjustment evaluation uses an **"OR"** condition for escalating therapy:
- **Adjustment Needed**: If the patient is ONLY in Clinical/Biochemical remission, **OR** if the Medication Range `>` Expected Time from SOPs (despite reaching remission).

## 4. Strict Response Templates
Outputs for Categories 1 and 2 must follow exact structural templates, including:
- Distinct sections for **Patient ID**, **Colonoscopy Dates**, and **Score Breakdowns**.
- An explicit table for **Remission Status** using boolean visual indicators (**✅ or ❌**) for Clinical, Bio-chemical, Endoscopic, and Histologic criteria.

## 5. Offline-First Mandate
The `query_guard_rag` strictly defaults to internal SOPs. Web searches are ONLY triggered as a fallback if zero internal matches are retrieved.

## 6. End-to-End Visual QA Validation
On March 25, 2026, the application underwent an End-to-End QA browser test using the Streamlit UI, yielding the following validated outcomes:
- **Scenario 1 (Severity & Remission)**: Successfully rendered the ✅/❌ icons for the Remission Checklist. (Verified missing sub-scores correctly compute to 0).
- **Scenario 2 (Medication Adjustment)**: Successfully flagged an "Adjustment Status: YES" utilizing the new `OR` logic condition, and appropriately cited `[Tier 1]` guidelines.
- **Scenario 3 (Cancer Surveillance)**: Accurately computed a 10-year disease duration and rendered the 8-year surveillance recommendation timeline based on Tier 1 criteria.
These verified screenshots have replaced the mock artifacts in the `QA_Report.md`.
