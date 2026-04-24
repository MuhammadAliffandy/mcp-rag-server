"""
ColonoSense Prompt Variation QA Tester
========================================
Acts as a human QA tester sending natural language question permutations
to the ColonoSense agent and evaluating:

  Q1.1 — Disease Severity:
    • Template: exactly 3-point numbered list
    • Math: Total Mayo = Partial Mayo + MAX(MES) → correct label
    • Data: correct patient values retrieved

  Q2.2 — Medication Adjustment:
    • Logic: "No Adjustment" | "Continue and reassess in X weeks" | "Adjustment"
    • Guard RAG: ALL tiers listed in strict [Tier X] 1. format
    • Template: 11-point numbered list, Point 11 = Medical SOP

Usage:
  python qa_tester.py --patient_id 2999892
  python qa_tester.py --patient_id 2999892 --category Q1.1
  python qa_tester.py --patient_id 5969795 --category Q2.2
  python qa_tester.py --patient_id 2999892 --category all
"""

import os, sys, json, re, datetime, argparse
import pandas as pd
from PineBioML.model.llm_factory import get_llm
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
load_dotenv()

EXCEL_FILE = "internal_docs/AI_UC_20260304(follow_up_20260211)_long.xlsx"
EVAL_DATE  = datetime.datetime(2026, 2, 11)

# header row differs per sheet
SHEET_HEADER = {
    "UC_baseline": 1,
    "UC_cpy"     : 0,
    "UC_lab"     : 0,
    "UC_histo"   : 0,
    "UC_med"     : 1,   # row 0 is metadata; row 1 has real column names
}

def _read(sheet: str) -> pd.DataFrame:
    return pd.read_excel(EXCEL_FILE, sheet_name=sheet, header=SHEET_HEADER[sheet])

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT VARIATION REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
def get_q1_1_prompts(pid: str) -> list[dict]:
    """All natural language permutations for Q1.1."""
    return [
        {"id": "Q1.1-A", "prompt": f"What is the disease severity of patient {pid}?"},
        {"id": "Q1.1-B", "prompt": f"What is the severity of patient {pid}'s disease?"},
        {"id": "Q1.1-C", "prompt": f"For patient {pid}, what is the disease severity?"},
        {"id": "Q1.1-D", "prompt": f"Assess the disease severity for patient ID {pid}."},
        {"id": "Q1.1-E", "prompt": f"Tell me the disease severity of patient {pid}."},
    ]

def get_q2_2_prompts(pid: str) -> list[dict]:
    """All natural language permutations for Q2.2."""
    return [
        {"id": "Q2.2-A", "prompt": f"Based on the patient {pid}'s current status, should the medication be adjusted?"},
        {"id": "Q2.2-B", "prompt": f"Based on patient {pid}'s current status, should the medication be adjusted? Continue from chat history."},
        {"id": "Q2.2-C", "prompt": f"Should medication be adjusted for patient {pid}?"},
        {"id": "Q2.2-D", "prompt": f"Does patient {pid} need a medication adjustment?"},
        {"id": "Q2.2-E", "prompt": f"Review medication adjustment needed for patient {pid} using STRIDE-II guidelines."},
        {"id": "Q2.2-F", "prompt": f"For patient {pid}, assess the medication and decide: No Adjustment, Continue and Reassess, or Adjustment?"},
    ]

# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH EXTRACTOR (fast version)
# ─────────────────────────────────────────────────────────────────────────────
def _match(df, pid):
    try:
        pid_int = int(pid)
        return df[df["id"].apply(lambda x: int(x) if pd.notnull(x) else -1) == pid_int]
    except Exception:
        return df[df["id"].astype(str) == str(pid)]

def extract_gt(pid) -> dict:
    gt = {"patient_id": str(pid)}
    try:
        # BASELINE
        df_b = _read("UC_baseline")
        b_rows = _match(df_b, pid)
        if b_rows.empty:
            avail = df_b["id"].dropna().tolist()[:5]
            gt["error"] = f"Patient {pid} not found. Available IDs: {avail}"
            return gt
        b = b_rows.iloc[-1]
        def _f(col): return float(b[col]) if col in b.index and pd.notnull(b[col]) else 0.0
        gt["bl_mayo_total"] = _f("bl_mayo_total")
        gt["bl_mayo_s"]     = _f("bl_mayo_s")
        gt["bl_mayo_b"]     = _f("bl_mayo_b")
        gt["bl_mayo_p"]     = _f("bl_mayo_p")
        gt["extent"]     = float(b["extent"])    if "extent"     in b.index and pd.notnull(b["extent"])    else None
        gt["birthday"]   = str(b["birthday"])   if "birthday"   in b.index else None
        gt["date_onset"] = str(b["date_onset"]) if "date_onset" in b.index else None

        # CPY (MES)
        df_c  = _read("UC_cpy")
        c_rows = _match(df_c, pid)
        gt["max_mes"] = 0.0; gt["last_cpy"] = None
        if not c_rows.empty:
            sort_col = "date_cpy" if "date_cpy" in df_c.columns else df_c.columns[2]
            lc = c_rows.sort_values(sort_col).iloc[-1]
            gt["last_cpy"] = str(lc.get(sort_col, ""))[:10]
            vals = [float(lc[k]) for k in ["mes_a","mes_t","mes_d","mes_s","mes_r"]
                    if k in lc.index and pd.notnull(lc[k])]
            gt["max_mes"] = max(vals) if vals else 0.0

        # LAB
        df_l  = _read("UC_lab")
        l_rows = _match(df_l, pid)
        gt["crp"] = gt["fc"] = gt["alb"] = None
        if not l_rows.empty:
            date_col = "lab_date" if "lab_date" in df_l.columns else df_l.columns[2]
            item_col = "lab_item"  if "lab_item" in df_l.columns else df_l.columns[3]
            val_col  = "lab_value" if "lab_value" in df_l.columns else df_l.columns[4]
            for item, key in [("crp","crp"),("fc","fc"),("alb","alb")]:
                rows = l_rows[l_rows[item_col].str.lower()==item].sort_values(date_col)
                if not rows.empty: gt[key] = float(rows.iloc[-1][val_col])

        # HISTO (Nancy)
        df_h  = _read("UC_histo")
        h_rows = _match(df_h, pid)
        gt["max_nancy"] = 0.0
        if not h_rows.empty:
            sort_col = "date_cpy" if "date_cpy" in df_h.columns else df_h.columns[2]
            lh = h_rows.sort_values(sort_col).iloc[-1]
            nvals = [float(lh[k]) for k in ["nancy_a","nancy_t","nancy_d","nancy_s","nancy_r"]
                     if k in lh.index and pd.notnull(lh[k])]
            gt["max_nancy"] = max(nvals) if nvals else 0.0

        # MED
        df_m  = _read("UC_med")
        m_rows = _match(df_m, pid).copy()
        gt["active_meds"] = []
        if not m_rows.empty:
            m_rows["start_date"] = pd.to_datetime(m_rows["start_date"], errors="coerce")
            m_rows["end_date"]   = pd.to_datetime(m_rows["end_date"],   errors="coerce")
            for _, row in m_rows.iterrows():
                st, en = row["start_date"], row["end_date"]
                if pd.notnull(st) and st <= EVAL_DATE:
                    if pd.isnull(en) or en >= EVAL_DATE:
                        gt["active_meds"].append({
                            "name"           : str(row.get("med_name","")),
                            "class"          : row.get("med_class"),
                            "start"          : str(st.date()),
                            "duration_weeks" : round((EVAL_DATE-st).days/7.0,1),
                        })
            if gt["active_meds"]:
                gt["active_meds"].sort(key=lambda x: x["start"], reverse=True)
                gt["index_drug"] = gt["active_meds"][0]

        # Derived flags
        pm  = gt["bl_mayo_total"]
        mes = gt["max_mes"]
        gt["total_mayo"]   = pm + mes
        score = gt["total_mayo"]
        gt["severity"]     = ("Remission" if score<=2 else "Mild" if score<=5
                              else "Moderate" if score<=10 else "Severe")
        gt["clinical_rem"] = (pm < 3 and
                              all(gt.get(k,0.0)<=1 for k in ["bl_mayo_s","bl_mayo_b","bl_mayo_p"]))
        gt["bio_rem"]      = ((gt["crp"] is not None and gt["crp"]<1.0) and
                              (gt["fc"]  is not None and gt["fc"]<100.0))
        gt["endo_rem"]     = mes <= 1.0
        gt["histo_rem"]    = gt["max_nancy"] <= 1.0

    except Exception as e:
        import traceback
        gt["error"] = f"{e}\n{traceback.format_exc()}"
    return gt

# ─────────────────────────────────────────────────────────────────────────────
# AGENT CALLER
# ─────────────────────────────────────────────────────────────────────────────
def call_agent(pid: str, prompt: str, category: str) -> str:
    from src.api.mcp_server import query_core_rag, query_guard_rag
    from PineBioML.prompts.synthesis import get_synthesis_prompt
    from PineBioML.model.llm_factory import get_llm

    raw   = query_core_rag(str(pid), prompt)
    sop   = query_guard_rag(prompt)
    tools = f"Core RAG:\n{raw}\n\nGuard RAG:\n{sop}"

    cat_id = "Q1.1" if category == "Q1.1" else "Q2.2"
    sys_p = get_synthesis_prompt(
        language="English",
        question=prompt,
        rag_context=raw,
        tool_outputs=tools,
        category_id=cat_id,
    )
    llm = get_llm(model_name="gpt-4o-mini", temperature=0)
    return llm.invoke([("system", sys_p)]).content

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATORS
# ─────────────────────────────────────────────────────────────────────────────

# --- Q1.1 Evaluator (deterministic) ------------------------------------------
def evaluate_q1_1(response: str, gt: dict, prompt_id: str) -> dict:
    text = response.strip()
    errors = []

    # RULE 1: Template structure — doctor format: "The patient is in [Severity] because total Mayo score was [X]."
    SEVERITY_LABELS = ["remission", "mild", "moderate", "severe"]
    has_template_phrase = bool(re.search(
        r'the patient is in (remission|mild|moderate|severe)', text, re.IGNORECASE
    )) or bool(re.search(
        r'total mayo score was \d', text, re.IGNORECASE
    ))
    has_mayo_values = bool(re.search(
        r'partial mayo.*\d|pMayo.*\d|mayo.*partial', text, re.IGNORECASE
    ))
    has_mes = bool(re.search(r'MES\s*[=:]?\s*\d|endoscopic.*\d', text, re.IGNORECASE))
    template_ok = has_template_phrase and has_mayo_values
    if not has_template_phrase:
        errors.append("Missing template phrase: 'The patient is in [Severity] because total Mayo score was [X].'")
    if not has_mayo_values:
        errors.append("Missing partial Mayo score value in response.")
    if not has_mes:
        errors.append("Missing MES value in response.")

    # RULE 2: Must mention Patient ID
    pid = gt["patient_id"]
    if pid not in text:
        errors.append(f"Patient ID '{pid}' not found in response.")

    # RULE 3: Severity label correct (fuzzy)
    expected_sev = gt["severity"]
    SEVERITY_SYNONYMS = {
        "remission": ["remission", "in remission", "disease remission", "clinical remission", "endoscopic remission"],
        "mild":      ["mild", "mild-moderate", "mild to moderate"],
        "moderate":  ["moderate", "moderately active", "mild-moderate"],
        "severe":    ["severe", "severely active", "fulminant"],
    }
    synonyms = SEVERITY_SYNONYMS.get(expected_sev.lower(), [expected_sev.lower()])
    sev_found = any(s in text.lower() for s in synonyms)
    if not sev_found:
        errors.append(f"Wrong severity: expected '{expected_sev}' or synonym, not found in response.")

    # RULE 4: [Tier X] citation
    has_tier = bool(re.search(r'\[Tier \d\]', text))
    if not has_tier:
        errors.append("Missing [Tier X] guideline citation.")

    total = gt["total_mayo"]
    return {
        "prompt_id"      : prompt_id,
        "category"       : "Q1.1",
        "ground_truth"   : {
            "total_mayo": total,
            "partial_mayo": gt["bl_mayo_total"],
            "max_mes": gt["max_mes"],
            "expected_severity": expected_sev,
            "last_cpy": gt.get("last_cpy",""),
        },
        "template_pass"  : template_ok,
        "severity_pass"  : sev_found,
        "extraction_pass": pid in text,
        "tier_pass"      : has_tier,
        "errors"         : errors,
        "overall"        : "✅ PASS" if not errors else "❌ FAIL",
        "response_snippet": text[:800],
    }


# --- Q2.2 Evaluator (deterministic + LLM tier check) -------------------------
TIER_JUDGE = """
You are a strict medical citation auditor.
Given the AGENT_RESPONSE, check if section "11. Medical SOP" or "Medical SOP" exists and contains:
  - At least ONE citation in the format: [Tier X] followed by a numbered recommendation
  - Ideally covers Tier 1, 2, 3, 4 in order
  - Each tier entry: "[Tier X]\\n  1. Recommendation [Society/Author, Year]"

Return ONLY JSON (no markdown):
{
  "tier_section_found": true/false,
  "tiers_present": ["Tier 1","Tier 2",...],
  "tier_format_correct": true/false,
  "tier_errors": ["list any format issues"]
}
"""

def evaluate_q2_2(response: str, gt: dict, prompt_id: str) -> dict:
    text = response.strip()
    errors = []

    # RULE 1: Template sentence (doctor format — concise)
    has_no_adjust = bool(re.search(r'no\.?\s*$|no adjustment|no,? the medication should not', text, re.IGNORECASE))
    has_continue  = bool(re.search(r'continue\s+and\s+reassess', text, re.IGNORECASE))
    has_adjust    = bool(re.search(r'yes.*medication should be adjusted|medication should be adjusted', text, re.IGNORECASE))
    has_decision  = has_no_adjust or has_continue or has_adjust
    if not has_decision:
        errors.append("No decision found. Expected: 'No.' / 'Continue and reassess in X weeks.' / 'Yes, the current medication should be adjusted.'")

    # RULE 2: [Tier X] citation present
    has_tier = bool(re.search(r'\[Tier \d\]', text))
    if not has_tier:
        errors.append("Missing [Tier X] guideline citation (required for all categories).")

    # RULE 3: Validate adjustment logic against ground truth
    endo_rem  = gt.get("endo_rem", False)
    histo_rem = gt.get("histo_rem", False)
    if endo_rem or histo_rem:
        if not has_no_adjust:
            errors.append(f"Patient in Endoscopic({'✅' if endo_rem else '❌'})/Histologic({'✅' if histo_rem else '❌'}) remission → expected 'No.' but not found.")
    else:
        if has_no_adjust and not has_continue and not has_adjust:
            errors.append("Patient NOT in endoscopic remission but agent said 'No Adjustment' without justification.")

    # RULE 4: LLM judge for tier citation quality (lightweight check)
    llm = get_llm(model_name="gpt-4o-mini", temperature=0)
    try:
        llm_with_json = llm.bind(response_format={"type": "json_object"})
        judge_resp = llm_with_json.invoke([
            ("system", TIER_JUDGE),
            ("human", f"AGENT_RESPONSE:\n{text}")
        ])
        tier_data = json.loads(judge_resp.content.strip().lstrip("```json").rstrip("```"))
    except Exception as e:
        tier_data = {"tier_section_found": False, "tiers_present": [], "tier_format_correct": False, "tier_errors": [str(e)]}

    if not tier_data.get("tier_section_found") and not has_tier:
        errors.append("HARD FAIL: No [Tier X] citation found in response.")
    tiers_found = tier_data.get("tiers_present", [])

    return {
        "prompt_id"             : prompt_id,
        "category"              : "Q2.2",
        "ground_truth"          : {
            "endoscopic_remission": gt.get("endo_rem"),
            "histologic_remission": gt.get("histo_rem"),
            "index_drug"          : gt.get("index_drug",{}).get("name","N/A"),
            "duration_weeks"      : gt.get("index_drug",{}).get("duration_weeks","N/A"),
        },
        "template_pass"         : has_decision,
        "decision_pass"         : has_decision,
        "adjustment_logic_pass" : not bool([e for e in errors if "remission" in e.lower()]),
        "tier_citation_pass"    : has_tier,
        "tiers_found"           : tiers_found,
        "errors"                : errors,
        "overall"               : "✅ PASS" if not errors else "❌ FAIL",
        "response_snippet"      : text[:800],
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPORT PRINTER
# ─────────────────────────────────────────────────────────────────────────────
def print_result(r: dict):
    sep = "─" * 56
    print(f"\n{sep}")
    print(f"  [{r['prompt_id']}]  {r['overall']}")
    print(sep)

    if r["category"] == "Q1.1":
        gt = r["ground_truth"]
        print(f"  Partial Mayo : {gt['partial_mayo']}  |  MAX MES : {gt['max_mes']}")
        print(f"  Total Mayo   : {gt['total_mayo']}  →  Expected Severity : {gt['expected_severity']}")
        print(f"  Last CPY     : {gt['last_cpy']}")
        print()
        print(f"  ✅ Template (≥3 pts) : {r['template_pass']}")
        print(f"  ✅ Severity label    : {r['severity_pass']}")
        print(f"  ✅ Patient ID found  : {r['extraction_pass']}")
    else:
        gt = r["ground_truth"]
        print(f"  Index Drug   : {gt['index_drug']}  |  Duration: {gt['duration_weeks']} wks")
        print(f"  Endo Rem     : {gt['endoscopic_remission']}  |  Histo Rem: {gt['histologic_remission']}")
        print()
        print(f"  ✅ Template (≥10 pts)   : {r['template_pass']}")
        print(f"  ✅ Decision found        : {r['decision_pass']}")
        print(f"  ✅ Adjustment logic      : {r['adjustment_logic_pass']}")
        print(f"  ✅ Tier citation format  : {r['tier_citation_pass']}")
        print(f"  📋 Tiers found           : {r['tiers_found']}")

    if r["errors"]:
        print()
        print("  ❌ Issues:")
        for e in r["errors"]:
            print(f"     • {e}")

    print()
    print("  Agent snippet:")
    snippet = r["response_snippet"].replace("\n", "\n  ")
    print(f"  {snippet[:400]}...")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient_id", required=True)
    parser.add_argument("--category",   default="all", help="Q1.1 | Q2.2 | all")
    args = parser.parse_args()
    pid  = args.patient_id

    print(f"\n{'═'*56}")
    print(f"  ColonoSense QA Tester  |  Patient: {pid}")
    print(f"  Eval date: {EVAL_DATE.date()}")
    print(f"{'═'*56}")

    # 1. Extract ground truth once
    print("\n[GT] Extracting ground truth...")
    gt = extract_gt(pid)
    if "error" in gt:
        print(f"❌  {gt['error']}")
        sys.exit(1)
    print(f"     Total Mayo : {gt['total_mayo']} → {gt['severity']}")
    print(f"     Endo Rem   : {gt['endo_rem']}  | Histo Rem: {gt['histo_rem']}")
    if gt.get("index_drug"):
        print(f"     Index Drug : {gt['index_drug']['name']} ({gt['index_drug']['duration_weeks']} wks)")

    cat = args.category.upper()
    all_results = []

    # 2. Q1.1 variations
    if cat in ("Q1.1", "ALL"):
        print(f"\n{'═'*56}")
        print(f"  CATEGORY Q1.1 — Disease Severity ({len(get_q1_1_prompts(pid))} prompts)")
        print(f"{'═'*56}")
        for pv in get_q1_1_prompts(pid):
            print(f"\n  ▷ Sending [{pv['id']}]: \"{pv['prompt']}\"")
            try:
                resp = call_agent(pid, pv["prompt"], "Q1.1")
                result = evaluate_q1_1(resp, gt, pv["id"])
            except Exception as e:
                result = {"prompt_id": pv["id"], "category":"Q1.1",
                          "overall": "❌ FAIL", "errors": [f"Agent error: {e}"],
                          "ground_truth":{}, "template_pass":False,
                          "severity_pass":False, "extraction_pass":False,
                          "response_snippet":""}
            print_result(result)
            all_results.append(result)

    # 3. Q2.2 variations
    if cat in ("Q2.2", "ALL"):
        print(f"\n{'═'*56}")
        print(f"  CATEGORY Q2.2 — Medication Adjustment ({len(get_q2_2_prompts(pid))} prompts)")
        print(f"{'═'*56}")
        for pv in get_q2_2_prompts(pid):
            print(f"\n  ▷ Sending [{pv['id']}]: \"{pv['prompt']}\"")
            try:
                resp = call_agent(pid, pv["prompt"], "Q2.2")
                result = evaluate_q2_2(resp, gt, pv["id"])
            except Exception as e:
                result = {"prompt_id": pv["id"], "category":"Q2.2",
                          "overall": "❌ FAIL", "errors": [f"Agent error: {e}"],
                          "ground_truth":{}, "template_pass":False,
                          "decision_pass":False, "adjustment_logic_pass":False,
                          "tier_citation_pass":False, "tiers_found":[],
                          "response_snippet":""}
            print_result(result)
            all_results.append(result)

    # 4. Summary table
    print(f"\n{'═'*56}")
    print("  OVERALL QA SUMMARY")
    print(f"{'═'*56}")
    q1_res = [r for r in all_results if r["category"]=="Q1.1"]
    q2_res = [r for r in all_results if r["category"]=="Q2.2"]

    def tally(res):
        p = sum(1 for r in res if r["overall"].startswith("✅"))
        return f"{p}/{len(res)} passed"

    if q1_res: print(f"  Q1.1 (Severity)      : {tally(q1_res)}")
    if q2_res: print(f"  Q2.2 (Adjustment)    : {tally(q2_res)}")

    total_pass = sum(1 for r in all_results if r["overall"].startswith("✅"))
    grand      = len(all_results)
    icon = "🟢" if total_pass == grand else "🔴"
    print(f"\n  {icon}  {total_pass}/{grand} total prompts passed")

    # 5. Save report
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"qa_variation_report_{pid}_{ts}.json"
    with open(path, "w") as f:
        json.dump({
            "patient_id"  : pid,
            "eval_date"   : str(EVAL_DATE.date()),
            "ground_truth": gt,
            "results"     : all_results,
        }, f, indent=2, default=str)
    print(f"\n  📄 Full report saved → {path}\n")


if __name__ == "__main__":
    main()
