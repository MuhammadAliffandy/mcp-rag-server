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
EXCEL_FILE = "internal_docs/4DEADFE0FD06EA10E459256A2E85237AB43BD9EB_UC_20260304(follow_up_20260211)_long.xlsx"
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
def generate_agent_response(pid, category: str) -> dict:
    """
    Calls the live ColonoSense agent via mcp_server tools and returns responses
    for each clinical question.
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

        # Step 1: Core RAG — get patient data
        raw_patient = query_core_rag(str(pid), q)

        # Step 2: Guard RAG — get clinical SOPs
        sop_context = query_guard_rag(q)

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
                ("human", "Please generate the clinical answer based on the instructions.")
            ])
            final_answer = resp.content
        else:
            final_answer = synthesize_medical_results(q, tool_outputs, raw_patient)

        responses[cat] = final_answer

    return responses


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: LLM EVALUATOR (ANTI-GRAVITY JUDGE)
# ═════════════════════════════════════════════════════════════════════════════
JUDGE_SYSTEM = """
You are Anti-Gravity, the strict QA Medical Auditor for ColonoSense.
You are given:
  1. GROUND_TRUTH: The mathematically expected values computed from raw Excel data.
  2. AGENT_RESPONSE: The text generated by the ColonoSense RAG agent.
  3. CATEGORY: The clinical question category being evaluated.

Your job:
- Compare AGENT_RESPONSE to GROUND_TRUTH strictly.
- Evaluate 4 metrics as true/false:
    a) internal_rag_extraction_pass   — Did agent extract correct values (Patient ID, dates, scores) from data?
    b) deterministic_math_pass        — Are all computed values (Total Mayo, MES, duration_weeks) numerically correct?
    c) guard_rag_logic_pass           — Did agent follow STRIDE-II adjustment logic correctly? 
                                        Is there a numbered [Tier X] citation in "Medical SOP" section?
    d) template_formatting_pass       — Does response use the exact numbered template, ✅/❌ emojis, ∆ symbols?

STRICT RULES:
- If any value is wrong by any amount, set that metric to false.
- For Q1.1: Must have exactly 3 numbered points, correct severity label, correct total mayo math.
- For Q1.2: Must have 7 numbered points. Point 7 must have 4 remission lines each with ✅/❌.
- For Q1.3: Must have 11 numbered points. If poor factors, Point 11 must say "∆ POOR PROGNOSIS".
- For Q2.2: Must have 11 numbered points. Point 11 must be "Medical SOP" with [Tier X] citations. If no citations → FAIL.

Return ONLY a JSON object, no markdown:
{
  "qa_session": {
    "patient_id_tested": "...",
    "category_tested": "...",
    "evaluation_timestamp": "2026-02-11"
  },
  "ground_truth_used": { ... summarize key values ... },
  "metrics": {
    "internal_rag_extraction_pass": true/false,
    "deterministic_math_pass": true/false,
    "guard_rag_logic_pass": true/false,
    "template_formatting_pass": true/false
  },
  "overall_status": "PASS or FAIL",
  "error_logs": ["...specific discrepancy details..."],
  "engineer_action_plan": "Concise fix instruction or 'None — all checks passed.'"
}
"""

def evaluate_with_llm(ground_truth: dict, agent_response: str, category: str) -> dict:
    """Sends ground truth + agent response to LLM judge and returns QA report."""
    llm = get_llm(model_name="gpt-4o", temperature=0)

    gt_summary = {k: v for k, v in ground_truth.items() if k != "errors"}
    user_msg = f"""
CATEGORY: {category}
PATIENT_ID: {ground_truth.get('patient_id')}

GROUND_TRUTH:
{json.dumps(gt_summary, indent=2, default=str)}

AGENT_RESPONSE:
{agent_response}
"""
    try:
        result = llm.invoke([("system", JUDGE_SYSTEM), ("human", user_msg)])
        content = result.content.strip()
        # Strip markdown fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception as e:
        return {
            "qa_session": {"patient_id_tested": str(ground_truth.get("patient_id")), "category_tested": category},
            "overall_status": "FAIL",
            "error_logs": [f"LLM Judge error: {str(e)}"],
            "engineer_action_plan": "Fix the LLM judge invocation."
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

    # STEP 2: Generate agent responses
    print("\n[2/3] Generating ColonoSense agent responses...")
    agent_responses = generate_agent_response(pid, args.category)

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
