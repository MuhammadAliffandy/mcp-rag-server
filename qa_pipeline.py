"""
ColonoSense End-to-End QA Pipeline
====================================
This script is the self-contained QA runner. It:
  1. Extracts ground truth from the raw Excel file.
  2. Calls the live ColonoSense agent with the correct clinical prompts.
  3. Evaluates the agent response against the MD specs via an LLM judge.
  4. Outputs a structured JSON QA Report.

Usage:
  python qa_pipeline.py --patient_id 1
  python qa_pipeline.py --patient_id 2999892 --category Q1.1
  python qa_pipeline.py --patient_id 1 --category all
"""

import os
import sys
import json
import argparse
import math
import datetime
import pandas as pd
from PineBioML.model.llm_factory import get_llm
from dotenv import load_dotenv

# ── path setup ───────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
load_dotenv()

# ── constants ─────────────────────────────────────────────────────────────────
EXCEL_FILE = "internal_docs/AI_UC_20260304(follow_up_20260211)_long.xlsx"
EVAL_DATE  = datetime.datetime(2026, 2, 11)

SHEET_HEADER = {
    "UC_baseline": 1,
    "UC_cpy"     : 0,
    "UC_lab"     : 0,
    "UC_histo"   : 0,
    "UC_med"     : 1,
}
# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: GROUND TRUTH EXTRACTOR
# ═════════════════════════════════════════════════════════════════════════════
def extract_ground_truth(pid) -> dict:
    """
    Reads the raw Excel file and computes all clinical ground-truth values.
    Returns a dict with pre-calculated expected values.
    """
    gt = {"patient_id": pid, "errors": []}

    try:
        # --- BASELINE ---
        df_b = pd.read_excel(EXCEL_FILE, sheet_name="UC_baseline", header=SHEET_HEADER["UC_baseline"])
        try:
            pid_int = int(pid)
            b_rows = df_b[df_b["id"].apply(lambda x: int(x) if pd.notnull(x) else -1) == pid_int]
        except (ValueError, TypeError):
            b_rows = df_b[df_b["id"].astype(str) == str(pid)]
        if b_rows.empty:
            available = df_b["id"].dropna().head(5).tolist()
            gt["errors"].append(f"Patient {pid} not found in UC_baseline. Available IDs (first 5): {available}")
        else:
            b = b_rows.sort_values("date_onset").iloc[-1]
            gt["bl_mayo_total"] = float(b["bl_mayo_total"]) if pd.notnull(b.get("bl_mayo_total")) else 0.0
            gt["bl_mayo_s"]     = float(b["bl_mayo_s"])     if pd.notnull(b.get("bl_mayo_s"))     else 0.0
            gt["bl_mayo_b"]     = float(b["bl_mayo_b"])     if pd.notnull(b.get("bl_mayo_b"))     else 0.0
            gt["bl_mayo_p"]     = float(b["bl_mayo_p"])     if pd.notnull(b.get("bl_mayo_p"))     else 0.0
            gt["extent"]        = float(b["extent"])         if pd.notnull(b.get("extent"))         else None
            gt["birthday"]      = str(b["birthday"])
            gt["date_onset"]    = str(b["date_onset"])
            try:
                onset = pd.to_datetime(b["date_onset"])
                bday  = pd.to_datetime(b["birthday"])
                gt["age_at_diagnosis"] = round((onset - bday).days / 365.25, 1)
            except Exception:
                gt["age_at_diagnosis"] = None

        # --- COLONOSCOPY (MES) ---
        df_c   = pd.read_excel(EXCEL_FILE, sheet_name="UC_cpy", header=SHEET_HEADER["UC_cpy"])
        try:
            pid_int = int(pid)
            c_rows = df_c[df_c["id"].apply(lambda x: int(x) if pd.notnull(x) else -1) == pid_int]
        except (ValueError, TypeError):
            c_rows = df_c[df_c["id"].astype(str) == str(pid)]
        if not c_rows.empty:
            latest_c = c_rows.sort_values("date_cpy").iloc[-1]
            gt["last_cpy_date"] = str(latest_c["date_cpy"])
            mes_cols = ["mes_a", "mes_t", "mes_d", "mes_s", "mes_r"]
            mes_vals = [float(latest_c[k]) for k in mes_cols if pd.notnull(latest_c.get(k))]
            gt["mes_values"] = {k: (float(latest_c[k]) if pd.notnull(latest_c.get(k)) else None) for k in mes_cols}
            gt["max_mes"]    = max(mes_vals) if mes_vals else 0.0
        else:
            gt["max_mes"]       = 0.0
            gt["last_cpy_date"] = None
            gt["mes_values"]    = {}

        # --- LAB ---
        df_l   = pd.read_excel(EXCEL_FILE, sheet_name="UC_lab", header=SHEET_HEADER["UC_lab"])
        try:
            pid_int = int(pid)
            l_rows = df_l[df_l["id"].apply(lambda x: int(x) if pd.notnull(x) else -1) == pid_int]
        except (ValueError, TypeError):
            l_rows = df_l[df_l["id"].astype(str) == str(pid)]
        gt["crp"] = gt["fc"] = gt["alb"] = None
        if not l_rows.empty:
            for item, key in [("crp", "crp"), ("fc", "fc"), ("alb", "alb")]:
                rows = l_rows[l_rows["lab_item"].str.lower() == item].sort_values("lab_date")
                if not rows.empty:
                    gt[key] = float(rows.iloc[-1]["lab_value"])

        # --- HISTOLOGY (Nancy) ---
        df_h   = pd.read_excel(EXCEL_FILE, sheet_name="UC_histo", header=SHEET_HEADER["UC_histo"])
        try:
            pid_int = int(pid)
            h_rows = df_h[df_h["id"].apply(lambda x: int(x) if pd.notnull(x) else -1) == pid_int]
        except (ValueError, TypeError):
            h_rows = df_h[df_h["id"].astype(str) == str(pid)]
        if not h_rows.empty:
            latest_h  = h_rows.sort_values("date_cpy").iloc[-1]
            nancy_cols = ["nancy_a", "nancy_t", "nancy_d", "nancy_s", "nancy_r"]
            nancy_vals = [float(latest_h[k]) for k in nancy_cols if pd.notnull(latest_h.get(k))]
            gt["nancy_values"] = {k: (float(latest_h[k]) if pd.notnull(latest_h.get(k)) else None) for k in nancy_cols}
            gt["max_nancy"]    = max(nancy_vals) if nancy_vals else 0.0
        else:
            gt["max_nancy"]    = 0.0
            gt["nancy_values"] = {}

        # --- MEDICATIONS ---
        df_m   = pd.read_excel(EXCEL_FILE, sheet_name="UC_med", header=SHEET_HEADER["UC_med"])
        try:
            pid_int = int(pid)
            m_rows = df_m[df_m["id"].apply(lambda x: int(x) if pd.notnull(x) else -1) == pid_int].copy()
        except (ValueError, TypeError):
            m_rows = df_m[df_m["id"].astype(str) == str(pid)].copy()
        gt["active_meds"] = []
        if not m_rows.empty:
            m_rows["start_date"] = pd.to_datetime(m_rows["start_date"], errors="coerce")
            m_rows["end_date"]   = pd.to_datetime(m_rows["end_date"],   errors="coerce")
            for _, row in m_rows.iterrows():
                st, en = row["start_date"], row["end_date"]
                if pd.notnull(st) and st <= EVAL_DATE:
                    if pd.isnull(en) or en >= EVAL_DATE:
                        dur_w = round((EVAL_DATE - st).days / 7.0, 1)
                        gt["active_meds"].append({
                            "med_name":      str(row.get("med_name", "")),
                            "med_class":     row.get("med_class"),
                            "start_date":    str(st.date()),
                            "end_date":      str(en.date()) if pd.notnull(en) else None,
                            "duration_weeks": dur_w,
                        })
            # Index drug = latest start_date
            if gt["active_meds"]:
                gt["active_meds"].sort(key=lambda x: x["start_date"], reverse=True)
                gt["index_drug"] = gt["active_meds"][0]

        # ── DERIVED BOOLEAN FLAGS ─────────────────────────────────────────
        partial_mayo = gt.get("bl_mayo_total", 0.0)
        max_mes      = gt.get("max_mes", 0.0)
        max_nancy    = gt.get("max_nancy", 0.0)
        crp          = gt.get("crp")
        fc           = gt.get("fc")
        alb          = gt.get("alb")
        extent       = gt.get("extent")
        age_dx       = gt.get("age_at_diagnosis")

        gt["total_mayo_score"] = partial_mayo + max_mes
        score = gt["total_mayo_score"]
        if   score <= 2:  gt["expected_severity"] = "Remission"
        elif score <= 5:  gt["expected_severity"] = "Mild"
        elif score <= 10: gt["expected_severity"] = "Moderate"
        else:             gt["expected_severity"] = "Severe"

        gt["clinical_remission"]     = partial_mayo < 3 and all(
            gt.get(k, 0.0) <= 1 for k in ["bl_mayo_s", "bl_mayo_b", "bl_mayo_p"]
        )
        gt["biochemical_remission"]  = (crp  is not None and crp  < 1.0) and \
                                       (fc   is not None and fc   < 100.0)
        gt["endoscopic_remission"]   = max_mes   <= 1.0
        gt["histologic_remission"]   = max_nancy <= 1.0

        # Poor prognosis factors
        steroid_use = any(
            m["med_class"] == 2 and m["med_name"] != "Cortiment MMX"
            for m in gt.get("active_meds", [])
        )
        poor_factors = []
        if age_dx is not None and age_dx < 40:                    poor_factors.append(f"Age at diagnosis {age_dx:.1f} yrs (< 40)")
        if extent is not None and extent == 3:                     poor_factors.append("Extensive colitis (extent=3)")
        if max_mes >= 3:                                           poor_factors.append(f"MES={max_mes:.0f} (severe endoscopic activity)")
        if crp is not None and crp > 1.0:                         poor_factors.append(f"Elevated CRP={crp} mg/dL (>1)")
        if alb is not None and alb < 3.5:                         poor_factors.append(f"Low Albumin={alb} g/dL (<3.5)")
        if steroid_use:                                            poor_factors.append("Steroid use (med_class=2, not Cortiment MMX)")

        gt["poor_factors"]            = poor_factors
        gt["expected_poor_prognosis"] = len(poor_factors) > 0

    except FileNotFoundError:
        gt["errors"].append(f"Excel file not found: {EXCEL_FILE}")
    except Exception as e:
        gt["errors"].append(f"Extraction error: {str(e)}")

    return gt


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: AGENT RESPONSE GENERATOR
# ═════════════════════════════════════════════════════════════════════════════
def _build_anchor_block(pid, gt: dict) -> str:
    """
    Builds a structured numeric anchor block from pre-computed ground truth.
    This is injected into the LLM prompt so it copies values instead of hallucinating.
    """
    index_drug = gt.get("index_drug", {})
    mes_vals   = gt.get("mes_values", {})
    nancy_vals = gt.get("nancy_values", {})
    poor_factors = gt.get("poor_factors", [])

    anchor = f"""
╔══════════════════════════════════════════════════════════════════╗
   STRUCTURED PATIENT ANCHOR — USE THESE VALUES EXACTLY
   DO NOT calculate or infer — copy directly from this block.
╚══════════════════════════════════════════════════════════════════╝
Patient ID            : {pid}

── MAYO SCORES ──────────────────────────────────────────────────
bl_mayo_total (Partial): {gt.get('bl_mayo_total', 'N/A')}
  bl_mayo_s (stool)    : {gt.get('bl_mayo_s', 'N/A')}
  bl_mayo_b (bleeding) : {gt.get('bl_mayo_b', 'N/A')}
  bl_mayo_p (physician): {gt.get('bl_mayo_p', 'N/A')}
max_mes (MES)          : {gt.get('max_mes', 'N/A')}
mes_values             : {mes_vals}
Total Mayo Score       : {gt.get('total_mayo_score', 'N/A')}
Expected Severity      : {gt.get('expected_severity', 'N/A')}

── LABS ──────────────────────────────────────────────────────────
crp_value              : {gt.get('crp', 'N/A')} mg/dL
fc_value               : {gt.get('fc', 'N/A')} ug/g
albumin                : {gt.get('alb', 'N/A')} g/dL

── HISTOLOGY (NANCY) ────────────────────────────────────────────
nancy_values           : {nancy_vals}
max_nancy              : {gt.get('max_nancy', 'N/A')}

── COLONOSCOPY ──────────────────────────────────────────────────
last_cpy_date          : {gt.get('last_cpy_date', 'N/A')}

── REMISSION FLAGS (pre-computed, copy exactly) ─────────────────
clinical_remission     : {'✅ YES' if gt.get('clinical_remission') else '❌ NO'}
biochemical_remission  : {'✅ YES' if gt.get('biochemical_remission') else '❌ NO'}
endoscopic_remission   : {'✅ YES' if gt.get('endoscopic_remission') else '❌ NO'}
histologic_remission   : {'✅ YES' if gt.get('histologic_remission') else '❌ NO'}

── DEMOGRAPHICS ─────────────────────────────────────────────────
age_at_diagnosis       : {gt.get('age_at_diagnosis', 'N/A')} years
extent                 : {gt.get('extent', 'N/A')}
date_onset             : {gt.get('date_onset', 'N/A')}

── PROGNOSIS ────────────────────────────────────────────────────
expected_poor_prognosis: {'✅ YES — POOR PROGNOSIS' if gt.get('expected_poor_prognosis') else '❌ NO poor factors'}
poor_factors           : {poor_factors if poor_factors else 'None'}

── MEDICATION (INDEX DRUG) ──────────────────────────────────────
index_drug_name        : {index_drug.get('med_name', 'N/A')}
index_drug_class       : {index_drug.get('med_class', 'N/A')}
index_drug_start_date  : {index_drug.get('start_date', 'N/A')}
index_drug_duration_wk : {index_drug.get('duration_weeks', 'N/A')} weeks
expected_adjustment    : {gt.get('expected_adjustment', 'See STRIDE-II logic in prompt')}
════════════════════════════════════════════════════════════════
"""
    return anchor


def generate_agent_response(pid, category: str, gt: dict = None) -> dict:
    """
    Calls the live ColonoSense agent via mcp_server tools and returns responses
    for each clinical question.
    gt (ground_truth dict): if provided, a structured patient anchor is injected
    into the synthesis prompt so the LLM uses pre-computed values directly.
    """
    # Ensure project root is importable
    project_root = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.api.mcp_server import query_core_rag, query_guard_rag, synthesize_medical_results

    prompts = {
        "Q1.1": f"What is the disease severity for patient {pid}? Provide Q1.1 Disease Severity Status.",
        "Q1.2": f"What is the remission status for patient {pid}? Provide Q1.2 Remission Status Assessment and Q2.1 Recommended Targets.",
        "Q1.3": f"What are the prognostic factors for patient {pid}? Provide Q1.3 Prognostic Factor Assessment.",
        "Q2.2": f"Should the medication be adjusted for patient {pid}? Provide Q2.2 Medication Adjustment Status.",
    }

    categories_to_run = list(prompts.keys()) if category == "all" else [category.upper()]
    responses = {}

    # Import synthesis prompt for category-forced strict output
    try:
        from PineBioML.prompts.synthesis import get_synthesis_prompt
        from PineBioML.model.llm_factory import get_llm
        synth_llm = get_llm(model_name="gpt-4o-mini", temperature=0)
        use_strict_synth = True
    except Exception:
        use_strict_synth = False

    for cat in categories_to_run:
        q = prompts.get(cat)
        if not q:
            responses[cat] = f"Unknown category: {cat}"
            continue

        print(f"\n[QA] Calling ColonoSense for {cat} — Patient {pid}...")

        # Step 1: Core RAG — get patient data (includes STRUCTURED PATIENT ANCHOR from Excel)
        raw_patient = query_core_rag(str(pid), q)

        # Step 2: Guard RAG — get clinical SOPs
        sop_context = query_guard_rag(q)

        # No ground truth injection — LLM extracts values from RAG context autonomously
        tool_outputs = f"Core RAG:\n{raw_patient}\n\nGuard RAG:\n{sop_context}"

        # Step 3: Use strict category-aware synthesis prompt if available
        if use_strict_synth:
            synth_prompt = get_synthesis_prompt(
                language="English",
                question=q,
                rag_context=raw_patient,
                tool_outputs=tool_outputs,
                category_id=cat,
            )
            resp = synth_llm.invoke([
                ("system", synth_prompt),
                ("human", "Please answer the clinical question. Extract all numeric values from the STRUCTURED PATIENT ANCHOR in TECHNICAL FINDINGS. Do NOT guess values.")
            ])
            final_answer = resp.content
        else:
            final_answer = synthesize_medical_results(q, tool_outputs, raw_patient)

        responses[cat] = final_answer

    return responses


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3A: DETERMINISTIC PYTHON EVALUATOR (Pilar 3)
# ═════════════════════════════════════════════════════════════════════════════
import re

def _parse_float(text: str, label: str):
    """Extract first float after a label keyword in agent response text."""
    pattern = rf"{re.escape(label)}[^\d-]*(\d+\.?\d*)"
    m = re.search(pattern, text, re.IGNORECASE)
    return float(m.group(1)) if m else None

def _check_numeric_close(agent_val, gt_val, tol=0.5) -> bool:
    if agent_val is None or gt_val is None:
        return False
    return abs(float(agent_val) - float(gt_val)) <= tol

def evaluate_deterministic_python(ground_truth: dict, agent_response: str, category: str) -> dict:
    """
    Pilar 3: Evaluate metrics that are mathematically verifiable WITHOUT calling an LLM.
    Returns partial metric dict:
      - internal_rag_extraction_pass
      - deterministic_math_pass
    """
    pid  = str(ground_truth.get("patient_id", ""))
    text = agent_response
    errors = []

    # ── 1. internal_rag_extraction_pass ─────────────────────────────────────
    # Check that agent mentions the correct Patient ID
    extraction_ok = pid in text
    if not extraction_ok:
        errors.append(f"[EXTRACTION] Patient ID '{pid}' not found in response.")

    # Check key date if available
    cpy_date = ground_truth.get("last_cpy_date", "")
    if cpy_date and cpy_date[:7] not in text:  # match YYYY-MM
        errors.append(f"[EXTRACTION] Last colonoscopy date '{cpy_date}' not found in response.")
        extraction_ok = False

    # ── 2. deterministic_math_pass ──────────────────────────────────────────
    math_ok = True

    if category == "Q1.1":
        gt_total = ground_truth.get("total_mayo_score")
        gt_sev   = ground_truth.get("expected_severity", "").lower()
        gt_mes   = ground_truth.get("max_mes")

        # Check Total Mayo Score appears correctly
        agent_total = _parse_float(text, "Total Mayo")
        if not _check_numeric_close(agent_total, gt_total):
            errors.append(f"[MATH] Total Mayo: expected {gt_total}, found {agent_total} in response.")
            math_ok = False

        # ── FUZZY severity check: allow synonyms (e.g. 'endoscopic remission' counts as 'remission') ──
        SEVERITY_SYNONYMS = {
            "remission": ["remission", "in remission", "disease remission", "clinical remission",
                          "endoscopic remission", "mucosal healing"],
            "mild":      ["mild", "mild-moderate", "mild to moderate"],
            "moderate":  ["moderate", "moderately active", "mild-moderate", "moderate to severe"],
            "severe":    ["severe", "severely active", "fulminant"],
        }
        text_lower = text.lower()
        synonyms = SEVERITY_SYNONYMS.get(gt_sev, [gt_sev])
        sev_found = any(s in text_lower for s in synonyms)
        if gt_sev and not sev_found:
            errors.append(f"[MATH] Severity label '{gt_sev}' (or synonyms) not found in response.")
            math_ok = False

        # Check MES max
        agent_mes = _parse_float(text, "MES max")
        if not _check_numeric_close(agent_mes, gt_mes):
            errors.append(f"[MATH] MES max: expected {gt_mes}, found {agent_mes}.")
            math_ok = False

    elif category in ("Q1.2", "Q2.1", "Q2.2"):
        # Check remission flags appear as ✅/❌
        remission_map = {
            "clinical_remission":    ground_truth.get("clinical_remission"),
            "biochemical_remission": ground_truth.get("biochemical_remission"),
            "endoscopic_remission":  ground_truth.get("endoscopic_remission"),
            "histologic_remission":  ground_truth.get("histologic_remission"),
        }
        for key, expected in remission_map.items():
            label = key.replace("_", " ").title()
            yes_found = "✅" in text or "YES" in text.upper()
            no_found  = "❌" in text or "NO" in text.upper()
            if expected is True and not yes_found:
                errors.append(f"[MATH] {label} should be YES but ✅ not found.")
                math_ok = False
            elif expected is False and not no_found:
                errors.append(f"[MATH] {label} should be NO but ❌ not found.")
                math_ok = False

        # For Q2.2: verify duration_weeks if index drug exists
        if category == "Q2.2":
            idx = ground_truth.get("index_drug", {})
            if idx:
                expected_dur = idx.get("duration_weeks")
                agent_dur    = _parse_float(text, "Duration")
                if not _check_numeric_close(agent_dur, expected_dur, tol=1.5):
                    errors.append(f"[MATH] Duration: expected {expected_dur}w, found {agent_dur}w.")
                    math_ok = False

    elif category == "Q1.3":
        # Check poor prognosis verdict matches
        gt_poor = ground_truth.get("expected_poor_prognosis", False)
        if gt_poor and "POOR PROGNOSIS" not in text.upper():
            errors.append("[MATH] Expected POOR PROGNOSIS but not found in response.")
            math_ok = False
        elif not gt_poor and "POOR PROGNOSIS" in text.upper():
            errors.append("[MATH] No poor prognosis expected but POOR PROGNOSIS keyword found.")
            math_ok = False

    elif category == "Q2.3":
        # If patient is in remission, response must recommend "optimize" not escalation
        in_remission = ground_truth.get("endoscopic_remission") and ground_truth.get("clinical_remission")
        text_lower = text.lower()
        if in_remission:
            OPTIMIZE_KEYS = ["optimize", "continue current", "no escalation", "no adjustment", "maintain"]
            ESCALATE_KEYS = ["escalate", "switch", "add-on immunomodulators"]
            optimize_found = any(k in text_lower for k in OPTIMIZE_KEYS)
            escalate_found = any(k in text_lower for k in ESCALATE_KEYS)
            if not optimize_found:
                errors.append("[MATH] Q2.3: Patient in remission — expected 'optimize/continue' recommendation not found.")
                math_ok = False
            if escalate_found and not optimize_found:
                errors.append("[MATH] Q2.3: Escalation recommended despite patient being in remission.")
                math_ok = False

    elif category == "Q5.1":
        # Dietary: check that response references patient's actual disease status
        in_remission = ground_truth.get("clinical_remission")
        text_lower = text.lower()
        REMISSION_DIET_KEYS = ["mediterranean", "whole grain", "omega-3", "fresh vegetable",
                                "remission diet", "balanced diet", "fiber-rich"]
        ACTIVE_DIET_KEYS    = ["low-residue", "cooked vegetable", "white rice", "low-fiber"]
        if in_remission:
            found = any(k in text_lower for k in REMISSION_DIET_KEYS)
            if not found:
                errors.append("[MATH] Q5.1: Patient in remission — Mediterranean/balanced diet recommendation not found.")
                math_ok = False
        else:
            found = any(k in text_lower for k in ACTIVE_DIET_KEYS)
            if not found:
                errors.append("[MATH] Q5.1: Active disease — low-residue diet recommendation not found.")
                math_ok = False

    elif category == "Q5.3":
        # Lifestyle: must mention at least 3 of the core UC lifestyle keywords
        text_lower = text.lower()
        LIFESTYLE_KEYS = ["smoking", "physical activity", "exercise", "stress",
                          "alcohol", "weight", "bmi", "mindfulness", "cessation"]
        found_keys = [k for k in LIFESTYLE_KEYS if k in text_lower]
        if len(found_keys) < 3:
            errors.append(f"[MATH] Q5.3: Only {len(found_keys)} lifestyle keywords found (need ≥3). Found: {found_keys}")
            math_ok = False

    elif category == "Q6.2":
        # Maternal risk: must reference at least 2 known maternal risk terms
        text_lower = text.lower()
        MATERNAL_RISK_KEYS = ["preeclampsia", "flare", "gestational", "vte",
                               "thromboembolism", "infection", "preterm", "comparable", "increased"]
        found_keys = [k for k in MATERNAL_RISK_KEYS if k in text_lower]
        if len(found_keys) < 2:
            errors.append(f"[MATH] Q6.2: Only {len(found_keys)} maternal risk keywords found (need ≥2). Found: {found_keys}")
            math_ok = False

    elif category == "Q6.3":
        # Fetal/neonatal risk: must reference at least 2 known neonatal risk terms
        text_lower = text.lower()
        NEONATAL_RISK_KEYS = ["preterm", "birth weight", "sga", "small for gestational",
                               "neonatal", "live vaccine", "congenital", "comparable", "increased",
                               "immunosuppression", "placental"]
        found_keys = [k for k in NEONATAL_RISK_KEYS if k in text_lower]
        if len(found_keys) < 2:
            errors.append(f"[MATH] Q6.3: Only {len(found_keys)} neonatal risk keywords found (need ≥2). Found: {found_keys}")
            math_ok = False

    return {
        "internal_rag_extraction_pass": extraction_ok and len([e for e in errors if "EXTRACTION" in e]) == 0,
        "deterministic_math_pass":      math_ok,
        "_python_errors":               errors,
    }


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3B: LLM JUDGE — Format & SOP only (Hybrid Judge)
# ═════════════════════════════════════════════════════════════════════════════
JUDGE_SYSTEM = """
You are Anti-Gravity, the strict QA Medical Auditor for ColonoSense.
You are given:
  1. AGENT_RESPONSE: The text generated by the ColonoSense RAG agent.
  2. CATEGORY: The clinical question category being evaluated.
  3. PYTHON_ERRORS: Deterministic errors already identified by the Python evaluator.

DOCTOR FORMAT RULES (the gold standard is concise sentences, NOT step-by-step blocks):

Q1.1: Response MUST contain the sentence: "The patient is in [Remission/Mild/Moderate/Severe] because total Mayo score was [X]. (partial Mayo score [X], MES [X])."
Q1.2: Response MUST list all achieved/not-achieved remission types with their values.
Q1.3: Response MUST state "The patient has [the below poor prognostic factors: ...] / no poor prognostic factors."
Q2.1: Response MUST state "Yes the patient had achieved [short/intermediate/long-term] treatment target ([description])."
Q2.2: Response MUST contain exactly ONE of:
      - "No." (if no adjustment needed)
      - "Continue and reassess in [X] weeks."
      - "Yes, ... the current medication should be adjusted."
      AND at least one [Tier X] citation.
Q3.1: MUST start with "[Tier 1] Since the patient belongs to [low/intermediate/high] risk group..."
Q3.2: MUST start with "[Tier 1] Based on the patient's sex, age, underlying disease..."
Q4.1–Q6.3: MUST include at least one [Tier X] citation in the response.

Your job is to evaluate 2 metrics:
  c) guard_rag_logic_pass — Did the agent follow the correct clinical reasoning per the ANCHOR/STRIDE-II?
                             Is there a [Tier X] citation in the response?
  d) template_formatting_pass — Does the response use the CONCISE DOCTOR SENTENCE format?
                                  NOT verbose Step 1/Step 2 blocks. The sentence must match the template above.

Return ONLY a JSON object, no markdown:
{
  "metrics": {
    "guard_rag_logic_pass": true/false,
    "template_formatting_pass": true/false
  },
  "error_logs": ["...specific format or logic discrepancy details..."],
  "engineer_action_plan": "Concise fix or 'None — all checks passed.'"
}
"""

def evaluate_with_llm(ground_truth: dict, agent_response: str, category: str) -> dict:
    """
    Hybrid evaluator: Python handles deterministic metrics, LLM handles format & SOP.
    Falls back to LLM-only if Python check fails to import.
    """
    pid = str(ground_truth.get("patient_id", "N/A"))

    # ── Step A: Python deterministic check ──────────────────────────────────
    python_result = evaluate_deterministic_python(ground_truth, agent_response, category)
    python_errors = python_result.get("_python_errors", [])

    # ── Step B: LLM judge for format & SOP only ─────────────────────────────
    llm = get_llm(model_name="gpt-4o", temperature=0)
    user_msg = f"""
CATEGORY: {category}
PATIENT_ID: {pid}

PYTHON_ERRORS (already identified — do NOT re-evaluate these):
{json.dumps(python_errors, indent=2)}

AGENT_RESPONSE:
{agent_response[:6000]}\n[...truncated if long...]
"""
    try:
        result  = llm.invoke([("system", JUDGE_SYSTEM), ("human", user_msg)])
        content = result.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        llm_result = json.loads(content)
    except Exception as e:
        llm_result = {
            "metrics": {"guard_rag_logic_pass": False, "template_formatting_pass": False},
            "error_logs": [f"LLM Judge error: {str(e)}"],
            "engineer_action_plan": "Fix LLM judge invocation."
        }

    # ── Step C: Merge results ────────────────────────────────────────────────
    all_errors = python_errors + llm_result.get("error_logs", [])
    merged_metrics = {
        "internal_rag_extraction_pass": python_result.get("internal_rag_extraction_pass", False),
        "deterministic_math_pass":      python_result.get("deterministic_math_pass", False),
        "guard_rag_logic_pass":         llm_result.get("metrics", {}).get("guard_rag_logic_pass", False),
        "template_formatting_pass":     llm_result.get("metrics", {}).get("template_formatting_pass", False),
    }
    overall = "PASS" if all(merged_metrics.values()) else "FAIL"

    return {
        "qa_session": {
            "patient_id_tested":    pid,
            "category_tested":      category,
            "evaluation_timestamp": "2026-02-11"
        },
        "ground_truth_used": {k: v for k, v in ground_truth.items() if k not in ("errors", "active_meds")},
        "metrics":           merged_metrics,
        "overall_status":    overall,
        "error_logs":        all_errors,
        "engineer_action_plan": llm_result.get("engineer_action_plan", "N/A")
    }


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: REPORT PRINTER
# ═════════════════════════════════════════════════════════════════════════════
EMOJI_MAP = {True: "🟢 PASS", False: "🔴 FAIL"}

def print_qa_report(report: dict, category: str):
    print("\n" + "="*60)
    print(f"🛡️  ANTI-GRAVITY QA REPORT — {category}")
    print("="*60)
    status = report.get("overall_status", "FAIL")
    icon   = "🟢" if status == "PASS" else "🔴"
    print(f"Overall Status : {icon} {status}")
    print(f"Patient ID     : {report.get('qa_session', {}).get('patient_id_tested', 'N/A')}")
    print(f"Eval Date      : {report.get('qa_session', {}).get('evaluation_timestamp', 'N/A')}")
    print()
    metrics = report.get("metrics", {})
    for key, val in metrics.items():
        label = key.replace("_", " ").title()
        print(f"  {'✅' if val else '❌'} {label}")
    print()
    logs = report.get("error_logs", [])
    if logs:
        print("📋 Error Logs:")
        for log in logs:
            print(f"  • {log}")
    print()
    fix = report.get("engineer_action_plan", "N/A")
    print(f"🔧 Engineer Action Plan:\n  {fix}")
    print("="*60)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="ColonoSense End-to-End QA Pipeline")
    parser.add_argument("--patient_id", required=True, help="Patient ID to test (e.g. 1 or 2999892)")
    parser.add_argument("--category",   default="all",  help="Category: Q1.1 | Q1.2 | Q1.3 | Q2.2 | all")
    args = parser.parse_args()

    # Resolve patient ID type
    try:
        pid = int(args.patient_id)
    except ValueError:
        pid = args.patient_id

    print(f"\n{'='*60}")
    print(f"ColonoSense QA Pipeline")
    print(f"Patient ID : {pid}")
    print(f"Category   : {args.category}")
    print(f"Eval Date  : {EVAL_DATE.date()}")
    print(f"{'='*60}")

    # STEP 1: Extract ground truth
    print("\n[1/3] Extracting ground truth from Excel...")
    gt = extract_ground_truth(pid)
    if gt.get("errors"):
        print("⚠️  Errors during extraction:")
        for e in gt["errors"]:
            print(f"   • {e}")
        if "not found" in str(gt["errors"]):
            sys.exit(1)

    print(f"      Total Mayo  = {gt.get('bl_mayo_total', 0)} + {gt.get('max_mes', 0)} = {gt.get('total_mayo_score', 0)} → {gt.get('expected_severity')}")
    print(f"      Clinical Remission   : {gt.get('clinical_remission')}")
    print(f"      Biochemical Remission: {gt.get('biochemical_remission')}")
    print(f"      Endoscopic Remission : {gt.get('endoscopic_remission')}")
    print(f"      Histologic Remission : {gt.get('histologic_remission')}")
    print(f"      Poor Prognosis       : {gt.get('expected_poor_prognosis')} — {gt.get('poor_factors', [])}")
    if gt.get("index_drug"):
        print(f"      Index Drug           : {gt['index_drug']['med_name']} ({gt['index_drug']['duration_weeks']} weeks)")

    # STEP 2: Generate agent responses (pass gt so anchor is injected)
    print("\n[2/3] Generating ColonoSense agent responses (with Patient Anchor)...")
    agent_responses = generate_agent_response(pid, args.category, gt=gt)

    # STEP 3: Evaluate each category
    print("\n[3/3] Evaluating responses against ground truth...")
    all_reports = {}
    for cat, response in agent_responses.items():
        print(f"\n      Evaluating {cat}...")
        report = evaluate_with_llm(gt, response, cat)
        all_reports[cat] = report
        print_qa_report(report, cat)

    # Save full report to JSON
    output_path = f"qa_report_patient_{pid}.json"
    with open(output_path, "w") as f:
        json.dump({
            "run_timestamp": datetime.datetime.now().isoformat(),
            "patient_id"   : str(pid),
            "ground_truth" : {k: v for k, v in gt.items() if k != "errors"},
            "agent_responses": agent_responses,
            "qa_reports"   : all_reports,
        }, f, indent=2, default=str)
    print(f"\n✅ Full QA report saved to: {output_path}")


if __name__ == "__main__":
    main()
