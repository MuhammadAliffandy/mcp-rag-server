# ColonoSense QA Technical & Evaluation Summary

This document provides a technical overview of the Large Language Model (LLM) infrastructure and the mathematical metrics used to evaluate the ColonoSense QA system.

## 1. LLM Architecture & Infrastructure

ColonoSense employs a hybrid LLM strategy to balance performance, privacy, and clinical accuracy.

| Component | Specification | Primary Use Case |
| :--- | :--- | :--- |
| **Local LLM** | **Llama 3.1 (70B)** via Ollama | Main clinical reasoning engine on DGX server for data privacy. |
| **Cloud LLM** | **GPT-4o-mini** (OpenAI) | Fallback for high-speed synthesis and structural template matching. |

### Resource Requirements (Per Question)
- **Input Context (RAG + SOPs):** ~2,500 tokens
- **Output Generation:** ~400 tokens
- **Total Footprint:** ~2,900 tokens per evaluation query.
- **Total Patient Report (18 Questions):** ~52,200 tokens total.

---

## 2. Evaluation Methodology (Mathematics)

To ensure clinical safety and trial protocol adherence, each response is graded against four core metrics:

### A. Data Retrieval Accuracy
Measures the system's ability to extract numeric "Anchors" from the patient database.
$$ \text{Retrieval} = \frac{\text{Correctly Extracted Anchors}}{\text{Total Expected Anchors}} $$

### B. Correctness (Truthfulness)
Ensures the AI output matches the ground truth without hallucinations.
$$ \text{Correctness} = 1 - \left( \frac{\text{Hallucinations} + \text{Contradictions}}{\text{Total Facts Stated}} \right) $$

### C. Concordance (Guideline Adherence)
Measures alignment with global clinical protocols (e.g., STRIDE-II or trial rubrics).
$$ \text{Concordance} = \frac{\text{Rules Adhered To}}{\text{Total Applicable Clinical Rules}} $$

### D. Completeness
Ensures all required fields in the forced response template are populated.
$$ \text{Completeness} = \frac{\text{Filled Template Fields}}{\text{Total Required Fields}} $$

---

---

## 4. Recent Updates & Knowledge Base (April 2026)

### A. Dataset & Patient Migration
- **New Data Source:** Migrated system-wide to `AI_UC_20260304(follow_up_20260211)_long.xlsx`.
- **Default Patient:** Updated system default to **Patient 4** for focused clinical validation.

### B. Remission Logic Fix (Critical)
- **Problem:** The pipeline previously fetched the absolute latest colonoscopy (often a flare visit), causing "False Negative" remission results.
- **Solution:** Implemented **Baseline-Linked Selection**. The system now prioritizes the colonoscopy record that matches the date in `UC_baseline` or selects the closest record on/before the evaluation date (2026-02-11).

### C. Synthesis Template Robustness
- **Medication Mapping:** Added direct mapping for numeric `med_class` (e.g., 3.0) and `med_name` in synthesis prompts to prevent LLM confusion.
- **Expanded Guidelines:** Updated STRIDE-II expected evaluation timelines for **Risankizumab** and **Ustekinumab** in the prompt context.

### D. Automated Reporting
- **PPTX Generation:** Implemented `generate_detailed_ppt_bs4.py` to automate the creation of 16:9 Widescreen technical reports, including automated element-level screenshots for each of the 18 QA categories.
