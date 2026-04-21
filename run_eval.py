"""
ColonoSense — Standalone Evaluation Runner  (run_eval.py)
==========================================================
Runs a full quantitative evaluation for one patient × one category.
No browser needed — outputs terminal table + JSON report.

Evaluation dimensions (matches the clinical trial rubric image):
  1. Data Retrieval Accuracy     — LLM judge checks field-level correctness
  2. Output Correctness          — LLM judge: Correct / Partially Correct / Incorrect
  3. Guideline Concordance       — LLM judge: Correct / Partially Correct / Incorrect
  4. Output Completeness         — LLM judge: Complete / Partially Complete / Incomplete
  5. Output Helpfulness          — LLM judge: Helpful / Partially Helpful / Not Helpful
  [6] Retrieval Trace Block      — checks presence of retrieval_trace + guideline_trace JSON

Usage:
  python run_eval.py --patient_id 1 --category Q1.1
  python run_eval.py --patient_id 1 --category all
  python run_eval.py --patient_id 1 --category Q2.3,Q3.1,Q4.1
"""

import os, sys, json, re, datetime, argparse, textwrap
import pandas as pd
import numpy as np
import itertools
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
load_dotenv()

EXCEL_FILE = "internal_docs/4DEADFE0FD06EA10E459256A2E85237AB43BD9EB_UC_20260304(follow_up_20260211)_long.xlsx"
EVAL_DATE  = datetime.datetime(2026, 2, 11)

# Correct header rows per sheet (confirmed by inspection)
SHEET_HEADER = {
    "UC_baseline": 1,   # row 0 = repeated header
    "UC_cpy":      0,   # row 0 = real column names
    "UC_lab":      0,
    "UC_histo":    0,
    "UC_med":      1,   # row 0 = repeated header
}

ALL_CATEGORIES = [
    "Q1.1","Q1.2","Q1.3",
    "Q2.1","Q2.2","Q2.3",
    "Q3.1","Q3.2",
    "Q4.1","Q4.2","Q4.3","Q4.4",
    "Q5.1","Q5.2","Q5.3",
    "Q6.1","Q6.2","Q6.3",
]

CATEGORY_PROMPTS = {
    "Q1.1": "What is the disease severity for patient {pid}?",
    "Q1.2": "What is the remission status for patient {pid}?",
    "Q1.3": "What are the prognostic factors for patient {pid}?",
    "Q2.1": "What treat-to-target status has patient {pid} achieved?",
    "Q2.2": "Should the medication be adjusted for patient {pid}?",
    "Q2.3": "What is the recommended next treatment option for patient {pid}?",
    "Q3.1": "What is the colorectal cancer screening plan for patient {pid}?",
    "Q3.2": "What other cancer screenings should patient {pid} receive?",
    "Q4.1": "What non-invasive monitoring exams are needed for patient {pid}?",
    "Q4.2": "Is therapeutic drug monitoring recommended for patient {pid}?",
    "Q4.3": "What medication-specific monitoring is required for patient {pid}?",
    "Q4.4": "What opportunistic infection screenings and vaccinations are required for patient {pid}?",
    "Q5.1": "What dietary recommendations are needed for patient {pid}?",
    "Q5.2": "What nutritional supplements or deficiency screenings are needed for patient {pid}?",
    "Q5.3": "What lifestyle modifications are recommended for patient {pid}?",
    "Q6.1": "Which medications for patient {pid} are safe during pregnancy?",
    "Q6.2": "What maternal risks does patient {pid} face from disease activity or medications?",
    "Q6.3": "What fetal risks does patient {pid} face from disease activity or medications?",
}

# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL COLOURS
# ─────────────────────────────────────────────────────────────────────────────
R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"; B = "\033[94m"
C = "\033[96m"; W = "\033[97m"; DIM = "\033[2m"; RST = "\033[0m"
BOLD = "\033[1m"

def col(text, c): return f"{c}{text}{RST}"
def rate_col(r):
    if r is None: return col("—", DIM)
    if r >= 0.80: return col(f"{r*100:.1f}%", G)
    if r >= 0.50: return col(f"{r*100:.1f}%", Y)
    return col(f"{r*100:.1f}%", R)

def verdict_col(v):
    if not v: return col("—", DIM)
    if v in ("Correct","Complete","Helpful"):           return col(v, G)
    if v in ("Partially Correct","Partially Complete","Partially Helpful"): return col(v, Y)
    return col(v, R)

# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────
def _read(sheet):
    return pd.read_excel(EXCEL_FILE, sheet_name=sheet, header=SHEET_HEADER[sheet])

def _match(df, pid):
    """Match patient rows using numeric or string ID."""
    try:
        pid_int = int(pid)
        return df[df["id"].apply(lambda x: int(float(x)) if pd.notnull(x) and str(x).replace('.','',1).isdigit() else -1) == pid_int]
    except Exception:
        return df[df["id"].astype(str).str.strip() == str(pid)]

def extract_gt(pid) -> dict:
    gt = {"patient_id": str(pid)}
    try:
        # ── UC_baseline ──────────────────────────────────────────────────────
        df_b  = _read("UC_baseline")
        b_rows = _match(df_b, pid)
        if b_rows.empty:
            avail = df_b["id"].dropna().head(8).tolist()
            return {"error": f"Patient {pid} not found in UC_baseline. Available IDs: {avail}"}
        b = b_rows.iloc[-1]

        def _f(col, default=0.0):
            val = b.get(col, None)
            return float(val) if val is not None and pd.notnull(val) else default
        def _s(col):
            val = b.get(col, None)
            return str(val).strip() if val is not None and pd.notnull(val) else None

        gt.update({
            "bl_mayo_total": _f("bl_mayo_total"),
            "bl_mayo_s":     _f("bl_mayo_s"),
            "bl_mayo_b":     _f("bl_mayo_b"),
            "bl_mayo_p":     _f("bl_mayo_p"),
            "extent":        _f("extent"),
            "birthday":      _s("birthday"),
            "date_onset":    _s("date_onset"),
            "sex":           _s("sex"),
            "age":           _f("age"),
            "psc":           str(int(_f("psc"))) if _s("psc") else "0",
            "smoking":       _s("smoking"),
            "family_hx_crc": str(int(_f("family_hx_crc"))) if _s("family_hx_crc") else "0",
            "duration":      _f("duration") if "duration" in b.index else None,
        })
        try:
            gt["age_at_dx"] = round(
                (pd.to_datetime(gt["date_onset"]) - pd.to_datetime(gt["birthday"])).days / 365.25, 1)
        except: gt["age_at_dx"] = None

        # ── UC_cpy (MES) ─────────────────────────────────────────────────────
        df_c  = _read("UC_cpy"); c_rows = _match(df_c, pid)
        gt.update({"max_mes": 0.0, "last_cpy": None, "mes_values": {}})
        if not c_rows.empty:
            sc = "date_cpy" if "date_cpy" in df_c.columns else df_c.columns[1]
            lc = c_rows.sort_values(sc).iloc[-1]
            gt["last_cpy"] = str(lc.get(sc, ""))[:10]
            mes_seg = {k: float(lc[k]) for k in ["mes_a","mes_t","mes_d","mes_s","mes_r"]
                       if k in lc.index and pd.notnull(lc[k])}
            gt["mes_values"] = mes_seg
            gt["max_mes"]    = max(mes_seg.values()) if mes_seg else 0.0

        # ── UC_histo (Nancy) ─────────────────────────────────────────────────
        df_h  = _read("UC_histo"); h_rows = _match(df_h, pid)
        gt.update({"max_nancy": 0.0, "nancy_values": {}})
        if not h_rows.empty:
            sc = "date_cpy" if "date_cpy" in df_h.columns else df_h.columns[1]
            lh = h_rows.sort_values(sc).iloc[-1]
            nancy_seg = {k: float(lh[k]) for k in ["nancy_a","nancy_t","nancy_d","nancy_s","nancy_r"]
                         if k in lh.index and pd.notnull(lh[k])}
            gt["nancy_values"] = nancy_seg
            gt["max_nancy"]    = max(nancy_seg.values()) if nancy_seg else 0.0

        # ── UC_lab ───────────────────────────────────────────────────────────
        df_l  = _read("UC_lab"); l_rows = _match(df_l, pid)
        gt.update({"crp": None, "fc": None, "alb": None,
                   "crp_date": None, "fc_date": None, "alb_date": None})
        if not l_rows.empty:
            dc = "lab_date"  if "lab_date"  in df_l.columns else df_l.columns[1]
            ic = "lab_item"  if "lab_item"  in df_l.columns else df_l.columns[2]
            vc = "lab_value" if "lab_value" in df_l.columns else df_l.columns[3]
            for item, key in [("crp","crp"),("fc","fc"),("alb","alb")]:
                rows = l_rows[l_rows[ic].astype(str).str.lower() == item].sort_values(dc)
                if not rows.empty:
                    gt[key]           = float(rows.iloc[-1][vc])
                    gt[f"{key}_date"] = str(rows.iloc[-1][dc])[:10]

        # ── UC_med ───────────────────────────────────────────────────────────
        df_m  = _read("UC_med"); m_rows = _match(df_m, pid).copy()
        gt["active_meds"] = []; gt["past_meds"] = []
        if not m_rows.empty:
            m_rows["start_date"] = pd.to_datetime(m_rows["start_date"], errors="coerce")
            m_rows["end_date"]   = pd.to_datetime(m_rows["end_date"],   errors="coerce")
            for _, row in m_rows.iterrows():
                st_, en = row["start_date"], row["end_date"]
                if pd.isnull(st_) or st_ > EVAL_DATE: continue
                entry = {
                    "name":           str(row.get("med_name","")),
                    "class":          int(float(row["med_class"])) if pd.notnull(row.get("med_class")) else None,
                    "dose":           str(row.get("dose","")),
                    "route":          str(row.get("route","")),
                    "interval":       str(row.get("interval","")),
                    "start":          str(st_.date()),
                    "end":            str(en.date()) if pd.notnull(en) else None,
                    "duration_weeks": round((EVAL_DATE - st_).days / 7.0, 1),
                }
                if pd.isnull(en) or en >= EVAL_DATE:
                    gt["active_meds"].append(entry)
                else:
                    entry["months_since_stopped"] = round((EVAL_DATE - en).days / 30.0, 1)
                    gt["past_meds"].append(entry)

            gt["active_meds"].sort(key=lambda x: x["start"], reverse=True)
            if gt["active_meds"]:
                gt["index_drug"] = gt["active_meds"][0]

        # ── DERIVED FLAGS ────────────────────────────────────────────────────
        pm, mes, nancy = gt["bl_mayo_total"], gt["max_mes"], gt["max_nancy"]
        crp, fc, alb   = gt.get("crp"), gt.get("fc"), gt.get("alb")

        gt["total_mayo"] = pm + mes
        score = gt["total_mayo"]
        gt["severity"] = ("Remission" if score<=2 else "Mild" if score<=5
                           else "Moderate" if score<=10 else "Severe")

        gt["clinical_rem"]     = pm < 3 and all(gt.get(k,0.0)<=1 for k in ["bl_mayo_s","bl_mayo_b","bl_mayo_p"])
        gt["bio_rem"]          = (crp is not None and crp < 1.0) and (fc is not None and fc < 100.0)
        gt["endo_rem"]         = mes <= 1.0
        gt["histo_rem"]        = nancy <= 1.0
        gt["full_remission"]   = gt["clinical_rem"] and gt["bio_rem"] and gt["endo_rem"]

        # steroid dependency
        steroid_meds = [m for m in gt["active_meds"]
                        if m["class"] == 2 and "cortiment" not in m["name"].lower()]
        gt["steroid_dependent"] = any(m["duration_weeks"] > 12 for m in steroid_meds)

        # poor prognosis factors
        pf = []
        if gt.get("age_at_dx") and gt["age_at_dx"] < 40: pf.append(f"Age<40 at Dx ({gt['age_at_dx']:.1f}y)")
        if gt["extent"] == 3:    pf.append("Extensive colitis (extent=3)")
        if mes >= 3:             pf.append(f"MES={mes:.0f} (severe)")
        if crp and crp > 1.0:   pf.append(f"CRP={crp} mg/dL (>1)")
        if alb and alb < 3.5:   pf.append(f"Albumin={alb} g/dL (<3.5)")
        if steroid_meds:         pf.append("Steroid use (non-Cortiment MMX)")
        gt["poor_factors"]    = pf
        gt["poor_prognosis"]  = len(pf) > 0

    except Exception as e:
        import traceback
        gt["error"] = f"{e}\n{traceback.format_exc()}"
    return gt

# ─────────────────────────────────────────────────────────────────────────────
# AGENT CALLER
# ─────────────────────────────────────────────────────────────────────────────
def call_agent(pid: str, category: str) -> str:
    question = CATEGORY_PROMPTS.get(category, f"Answer {category} for patient {pid}").format(pid=pid)
    try:
        from src.api.mcp_server import query_core_rag, query_guard_rag
        from PineBioML.prompts.synthesis import get_synthesis_prompt
        from PineBioML.model.llm_factory import get_llm

        print(f"  {DIM}[Core RAG]  fetching patient data...{RST}")
        raw  = query_core_rag(str(pid), question)

        print(f"  {DIM}[Guard RAG] fetching guidelines...{RST}")
        sop  = query_guard_rag(question)

        tools = f"Core RAG:\n{raw}\n\nGuard RAG:\n{sop}"
        prompt = get_synthesis_prompt("English", question, raw, tools, category_id=category)

        print(f"  {DIM}[Synthesis] running LLM...{RST}")
        llm = get_llm(model_name="gpt-4o-mini", temperature=0)
        return llm.invoke([("system", prompt)]).content

    except Exception as e:
        import traceback
        return f"[Agent Error] {e}\n{traceback.format_exc()}"

# ─────────────────────────────────────────────────────────────────────────────
# LLM JUDGES
# ─────────────────────────────────────────────────────────────────────────────
def _judge(system_prompt: str, gt: dict, response: str, category: str) -> dict:
    try:
        from PineBioML.model.llm_factory import get_llm
        llm = get_llm(model_name="gpt-4o-mini", temperature=0)

        # Clean ground truth: remove heavy list fields
        gt_clean = {k: v for k, v in gt.items() if k not in ("error", "mes_values", "nancy_values")}
        user_msg = (
            f"CATEGORY: {category}\n"
            f"GROUND_TRUTH:\n{json.dumps(gt_clean, indent=2, default=str)[:3000]}\n\n"
            f"AGENT_RESPONSE:\n{response[:5000]}"
        )
        llm_with_json = llm.bind(response_format={"type": "json_object"})
        res     = llm_with_json.invoke([("system", system_prompt), ("human", user_msg)])
        content = res.content.strip()
        # Strip markdown fences
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'```\s*$', '', content).strip()
        return json.loads(content)
    except Exception as e:
        return {"error": str(e), "verdict": "Incorrect", "accuracy_rate": 0.0}


J_DATA = """\
You are a strict clinical data auditor checking DATA RETRIEVAL ACCURACY.

Compare AGENT_RESPONSE against GROUND_TRUTH.

FIRST, identify WHICH fields are actually REQUIRED to answer the clinical question for this specific CATEGORY.
For example:
- Q1.x requires Mayo sub-scores, MES, Nancy, etc.
- Q4.x requires medications, CRP, FC, etc.
- Q5.x requires BMI, smoking, albumin, extent, etc.
DO NOT penalize the agent for omitting data fields that are IRRELEVANT to the current category.

For each REQUIRED field, score 1 (correct) or 0 (incorrect/missing).
A value is "correct" if it appears in the response and matches the ground truth within ±0.1 tolerance.
IMPORTANT: The response may include a PATIENT ANCHOR/DATA RETRIEVAL section — values from there count as correctly retrieved.

Return ONLY valid JSON (no markdown, no prose) with dynamically generated keys based on the required fields:
{
  "required_fields_identified": ["field1", "field2"],
  "field_scores": {"field1": 1, "field2": 0},
  "correct_count": 1,
  "total_fields": 2,
  "accuracy_rate": 0.50,
  "incorrect_fields": ["field2"],
  "verdict": "Partially Correct"
}
Verdict rule: accuracy_rate >= 0.8 → "Correct", 0.5–0.79 → "Partially Correct", < 0.5 → "Incorrect".
"""

J_CORRECTNESS = """\
You are a senior IBD physician judging OUTPUT CORRECTNESS.

Check whether:
1. The final conclusion sentence exactly matches the required fill-in-the-blank template for this category.
2. The key clinical decision (severity / remission label / adjustment / screening interval) is medically CORRECT per the ground truth values.
3. All cited numeric values are factually accurate.

Verdict:
  "Correct"           — template sentence present + decision medically correct + values accurate
  "Partially Correct" — decision correct but template format deviated, OR minor value error
  "Incorrect"         — wrong decision OR major factual error

Return ONLY valid JSON:
{
  "template_sentence_found": true,
  "decision_clinically_correct": true,
  "value_errors": [],
  "verdict": "Correct",
  "accuracy_rate": 1.0
}
"""

J_CONCORDANCE = """\
You are a senior IBD physician judging GUIDELINE CONCORDANCE.

Check whether the agent's recommendations align with current IBD guidelines (ECCO, ACG, STRIDE-II, BSG, AGA).
Look for: correct guideline citations in [Tier X] format, correct recommendation per guideline, no contradictions.

Verdict:
  "Correct"           — recommendations align with ≥1 named guideline correctly cited
  "Partially Correct" — partially aligned or citation missing but logic correct
  "Incorrect"         — contradicts guidelines or no citation at all

Return ONLY valid JSON:
{
  "guideline_citations_present": true,
  "guidelines_cited": ["ECCO 2023","ACG 2021"],
  "major_concordance_errors": [],
  "verdict": "Correct",
  "concordance_rate": 1.0
}
"""

J_COMPLETENESS = """\
You are a clinical completeness auditor.

Check that ALL required template sections are present for this clinical category.

IMPORTANT: Accept EITHER format as equivalent and complete:
  Format A: "Step 1 — DATA RETRIEVAL" / "Step 2 — GUARD RAG LOGIC" / "Final Clinical Conclusion"
  Format B: "DATA RETRIEVAL" / "GUARD RAG LOGIC" / "Final Clinical Conclusion" (without Step labels)
  Format C: Numbered list (1-N points) + "Final Clinical Conclusion" sentence
  Format D: FORCE ACTION block + Final ANSWER sentence + retrieval_trace JSON

Section equivalences (mark as FOUND if ANY equivalent heading appears):
  "Step 1" or "DATA RETRIEVAL" or "FORCE ACTION" or a numbered intro = Data section
  "Step 2" or "GUARD RAG LOGIC" or "MONITORING RULES" or "SCREENING RULES" = Logic section
  "Final Clinical Conclusion" or any sentence starting with:
    "Based on", "Yes,", "No,", "This patient", "Screening for", "For patients under" = Conclusion present

Also check:
  - A ```json retrieval_trace block OR "retrieval_trace" keyword present
  - A "guideline_trace" keyword present

DO NOT mark as "Incomplete" just because "Step 1"/"Step 2" labels are missing.
DO NOT mark as "Incomplete" if response has a correct FINAL ANSWER sentence plus trace block.

Verdict:
  "Complete"           — data section present + conclusion sentence present + retrieval_trace present
  "Partially Complete" — conclusion present OR trace present but not both; or one section missing
  "Incomplete"         — no conclusion sentence AND no retrieval_trace

Return ONLY valid JSON:
{
  "sections_found": ["DATA RETRIEVAL","GUARD RAG LOGIC","Final Clinical Conclusion","retrieval_trace"],
  "sections_missing": [],
  "retrieval_trace_present": true,
  "guideline_trace_present": true,
  "verdict": "Complete",
  "complete_rate": 1.0
}
"""

J_HELPFULNESS = """\
You are a junior gastroenterologist evaluating clinical HELPFULNESS of an AI-generated response.

Judge whether this response would be helpful in your daily clinical decision-making:
  "Helpful"           — immediately actionable, clear recommendation, would reinforce or change your decision
  "Partially Helpful" — provides some useful info but missing context or clarity
  "Not Helpful"       — wrong, incomplete, or would NOT influence your clinical decision

Return ONLY valid JSON:
{
  "actionable": true,
  "clear_recommendation": true,
  "missing_elements": [],
  "verdict": "Helpful",
  "helpfulness_rate": 1.0
}
"""

JUDGES = [
    ("Data Retrieval Accuracy",  J_DATA,         "accuracy_rate",    "verdict"),
    ("Output Correctness",       J_CORRECTNESS,  "accuracy_rate",    "verdict"),
    ("Guideline Concordance",    J_CONCORDANCE,  "concordance_rate", "verdict"),
    ("Output Completeness",      J_COMPLETENESS, "complete_rate",    "verdict"),
    ("Output Helpfulness",       J_HELPFULNESS,  "helpfulness_rate", "verdict"),
]

# ─────────────────────────────────────────────────────────────────────────────
# KRIPPENDORFF'S ALPHA  (ordinal, simplified)
# ─────────────────────────────────────────────────────────────────────────────
_ORD = {"Correct":1.0, "Partially Correct":0.5, "Incorrect":0.0,
        "Complete":1.0, "Partially Complete":0.5, "Incomplete":0.0,
        "Helpful":1.0, "Partially Helpful":0.5, "Not Helpful":0.0}

def krippendorff_alpha(ratings: list) -> float:
    """ratings: list of numeric values (one per simulated rater). Returns α."""
    try:
        arr = np.array(ratings, dtype=float)
        n   = len(arr)
        if n < 2: return float("nan")
        Do = sum((a-b)**2 for a,b in itertools.combinations(arr, 2)) / len(list(itertools.combinations(arr, 2)))
        mu = arr.mean()
        De = sum((v - mu)**2 for v in arr) / n
        if De == 0: return 1.0
        return round(1.0 - Do / De, 3)
    except: return float("nan")

def simulate_raters(base_verdict: str, n: int = 5, noise_sd: float = 0.15) -> list:
    """Simulate n rater scores around the LLM-judge verdict (for demo IRV)."""
    base = _ORD.get(base_verdict, 0.5)
    rng  = np.random.default_rng(seed=42)
    return list(np.clip(np.round(rng.normal(base, noise_sd, n), 2), 0, 1))

# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH PRINTER
# ─────────────────────────────────────────────────────────────────────────────
def print_gt_summary(gt: dict):
    print(f"\n{BOLD}{C}━━━  GROUND TRUTH  ━━━{RST}")
    print(f"  Patient ID    : {col(gt['patient_id'], W)}")
    print(f"  Severity      : {col(gt.get('severity','?'), W)}  "
          f"(Total Mayo {gt.get('total_mayo',0):.1f} = "
          f"PM {gt.get('bl_mayo_total',0)} + MES {gt.get('max_mes',0)})")
    print(f"  sub-scores    : S={gt.get('bl_mayo_s')}, B={gt.get('bl_mayo_b')}, P={gt.get('bl_mayo_p')}")
    print(f"  MES max       : {col(str(gt.get('max_mes',0)), W)}  "
          f"Nancy max: {col(str(gt.get('max_nancy',0)), W)}")
    print(f"  CRP           : {gt.get('crp','—')} mg/dL   FC: {gt.get('fc','—')} µg/g   Alb: {gt.get('alb','—')} g/dL")
    rem = (f"Clinical={'✅' if gt.get('clinical_rem') else '❌'}  "
           f"Bio={'✅' if gt.get('bio_rem') else '❌'}  "
           f"Endo={'✅' if gt.get('endo_rem') else '❌'}  "
           f"Histo={'✅' if gt.get('histo_rem') else '❌'}")
    print(f"  Remission     : {rem}")
    print(f"  Poor Prognosis: {'∆ YES' if gt.get('poor_prognosis') else 'No'}"
          + (f"  ({', '.join(gt['poor_factors'])})" if gt.get('poor_factors') else ""))
    if gt.get("active_meds"):
        idx = gt["active_meds"][0]
        print(f"  Index Drug    : {col(idx['name'], W)}  class={idx['class']}  "
              f"{idx['duration_weeks']}w  route={idx['route']}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION RUNNER (single category)
# ─────────────────────────────────────────────────────────────────────────────
def run_category(pid: str, category: str, gt: dict) -> dict:
    print(f"\n{BOLD}{B}{'═'*64}{RST}")
    print(f"{BOLD}{B}  EVALUATING  {W}{category}{RST}  —  Patient {pid}")
    print(f"{B}{'═'*64}{RST}")

    # 1. Generate agent response
    print(f"\n{C}[1/3] Generating ColonoSense response...{RST}")
    response = call_agent(pid, category)

    if response.startswith("[Agent Error]"):
        print(col(f"  ✗ Agent failed: {response[:200]}", R))
        return {"category": category, "error": response, "overall": "FAIL"}

    print(f"  {G}✓ Response generated ({len(response)} chars){RST}")
    print(f"\n{DIM}  Response preview:{RST}")
    for line in response[:600].splitlines():
        print(f"    {DIM}{line}{RST}")
    if len(response) > 600:
        print(f"    {DIM}[... +{len(response)-600} chars]{RST}")

    # 2. Run all 5 judges
    print(f"\n{C}[2/3] Running LLM judges...{RST}")
    judge_results = {}
    for dim_name, sys_prompt, rate_key, verdict_key in JUDGES:
        print(f"  {DIM}→ {dim_name}...{RST}", end="", flush=True)
        jr = _judge(sys_prompt, gt, response, category)
        judge_results[dim_name] = jr
        rate    = jr.get(rate_key, jr.get("accuracy_rate"))
        verdict = jr.get(verdict_key, "—")
        print(f"  {rate_col(rate)}  {verdict_col(verdict)}")

    # 3. Compute IRV (simulated 5 physician raters per dimension)
    print(f"\n{C}[3/3] Computing inter-rater variability (Krippendorff's α)...{RST}")
    irv_results = {}
    for dim_name, _, rate_key, verdict_key in JUDGES[1:]:  # Dims 2-5 for IRV
        jr      = judge_results[dim_name]
        verdict = jr.get(verdict_key, "Incorrect")
        sims    = simulate_raters(verdict, n=5)
        alpha   = krippendorff_alpha(sims)
        irv_results[dim_name] = {"simulated_ratings": sims, "alpha": alpha}
        print(f"  {dim_name[:30]:30s}  alpha={col(f'{alpha:.3f}', C)}  "
              f"rater_scores={[round(s,2) for s in sims]}")

    # 4. Check retrieval_trace block
    has_trace = bool(re.search(r'retrieval_trace', response))
    has_gtrace= bool(re.search(r'guideline_trace', response))
    print(f"\n  Retrieval trace block: {'✅' if has_trace else '❌'}  "
          f"Guideline trace: {'✅' if has_gtrace else '❌'}")

    # 5. Build final result dict
    result = {
        "patient_id": pid,
        "category":   category,
        "timestamp":  datetime.datetime.now().isoformat(),
        "response_length": len(response),
        "retrieval_trace_present":  has_trace,
        "guideline_trace_present":  has_gtrace,
        "judges": {dim: jr for dim, jr in judge_results.items()},
        "irv":    irv_results,
        "agent_response": response,
    }

    # Extract key rates for summary table
    result["data_accuracy"]   = judge_results["Data Retrieval Accuracy"].get("accuracy_rate")
    result["correctness"]     = judge_results["Output Correctness"].get("accuracy_rate")
    result["concordance"]     = judge_results["Guideline Concordance"].get("concordance_rate")
    result["completeness"]    = judge_results["Output Completeness"].get("complete_rate")
    result["helpfulness"]     = judge_results["Output Helpfulness"].get("helpfulness_rate")
    result["correctness_v"]   = judge_results["Output Correctness"].get("verdict","—")
    result["concordance_v"]   = judge_results["Guideline Concordance"].get("verdict","—")
    result["completeness_v"]  = judge_results["Output Completeness"].get("verdict","—")
    result["helpfulness_v"]   = judge_results["Output Helpfulness"].get("verdict","—")
    result["overall"]         = "PASS" if all([
        (result["data_accuracy"]  or 0) >= 0.6,
        result["correctness_v"]  in ("Correct","Partially Correct"),
        result["concordance_v"]  in ("Correct","Partially Correct"),
        result["completeness_v"] in ("Complete","Partially Complete"),
    ]) else "FAIL"

    return result

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(all_results: list, gt: dict):
    print(f"\n\n{BOLD}{W}{'━'*96}{RST}")
    print(f"{BOLD}{W}  EVALUATION SUMMARY   Patient: {gt['patient_id']}    {EVAL_DATE.date()}{RST}")
    print(f"{BOLD}{W}{'━'*96}{RST}")

    HDR = (f"{'Cat':5}  {'Data Acc':9}  {'Correctness':13}  {'Concordance':13}  "
           f"{'Completeness':13}  {'Helpfulness':12}  {'IRV α':7}  {'Status':6}")
    print(f"{DIM}{HDR}{RST}")
    print(f"{DIM}{'─'*96}{RST}")

    for r in all_results:
        if "error" in r:
            print(f"  {r['category']:5}  {col('ERROR: ' + str(r['error'])[:70], R)}")
            continue

        irv_vals = [v["alpha"] for v in r.get("irv",{}).values() if isinstance(v.get("alpha"), float)]
        mean_irv = np.nanmean(irv_vals) if irv_vals else float("nan")

        status_icon = col("PASS ✓", G) if r["overall"] == "PASS" else col("FAIL ✗", R)
        row = (
            f"  {W}{r['category']:5}{RST}  "
            f"{rate_col(r.get('data_accuracy')):9}  "
            f"{verdict_col(r.get('correctness_v','—')):13}  "
            f"{verdict_col(r.get('concordance_v','—')):13}  "
            f"{verdict_col(r.get('completeness_v','—')):13}  "
            f"{verdict_col(r.get('helpfulness_v','—')):12}  "
            f"{col(f'{mean_irv:.3f}' if not np.isnan(mean_irv) else '—', C):7}  "
            f"{status_icon}"
        )
        print(row)

    print(f"{DIM}{'─'*96}{RST}")

    # Aggregate
    valid = [r for r in all_results if "error" not in r]
    def _avg(key): return np.nanmean([r.get(key,np.nan) for r in valid])
    pass_n = sum(1 for r in valid if r["overall"] == "PASS")

    print(f"\n  {BOLD}Aggregate (n={len(valid)} categories):{RST}")
    print(f"    Data Retrieval Accuracy : {rate_col(_avg('data_accuracy'))}")
    print(f"    Output Correctness      : {rate_col(_avg('correctness'))}")
    print(f"    Guideline Concordance   : {rate_col(_avg('concordance'))}")
    print(f"    Output Completeness     : {rate_col(_avg('completeness'))}")
    print(f"    Output Helpfulness      : {rate_col(_avg('helpfulness'))}")
    icon = col("🟢 PASS", G) if pass_n == len(valid) else (col("🟡 PARTIAL", Y) if pass_n > 0 else col("🔴 FAIL", R))
    print(f"\n    {BOLD}Overall: {icon}  {pass_n}/{len(valid)} categories passed{RST}")
    print(f"{BOLD}{W}{'━'*96}{RST}\n")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ColonoSense Evaluation Runner")
    parser.add_argument("--patient_id", required=True, help="Patient ID (e.g. 1)")
    parser.add_argument("--category",   default="Q1.1",
                        help="Category: Q1.1 | Q2.2 | all | Q1.1,Q2.2,Q3.1")
    parser.add_argument("--out",        default=None, help="Output JSON path (optional)")
    args = parser.parse_args()
    pid  = args.patient_id

    # Resolve categories
    raw_cat = args.category.strip()
    if raw_cat.lower() == "all":
        cats = ALL_CATEGORIES
    else:
        cats = [c.strip().upper() for c in raw_cat.split(",") if c.strip()]

    print(f"\n{BOLD}{W}{'━'*64}{RST}")
    print(f"{BOLD}{W}  ColonoSense Clinical Evaluation Runner{RST}")
    print(f"  Patient   : {col(pid, C)}")
    print(f"  Categories: {col(', '.join(cats), C)}")
    print(f"  Eval Date : {EVAL_DATE.date()}")
    print(f"{BOLD}{W}{'━'*64}{RST}\n")

    # 1. Extract ground truth once
    print(f"{C}[STEP 1] Extracting ground truth from Excel...{RST}")
    gt = extract_gt(pid)
    if "error" in gt:
        print(col(f"  ✗ {gt['error']}", R))
        sys.exit(1)
    print_gt_summary(gt)

    # 2. Run each category
    all_results = []
    for cat in cats:
        r = run_category(pid, cat, gt)
        all_results.append(r)

    # 3. Print summary table
    print_summary(all_results, gt)

    # 4. Save JSON report
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"eval_report_{pid}_{ts}.json"
    report   = {
        "run_timestamp": datetime.datetime.now().isoformat(),
        "patient_id":    pid,
        "categories":    cats,
        "eval_date":     str(EVAL_DATE.date()),
        "ground_truth":  {k: v for k, v in gt.items() if k not in ("error",)},
        "results":       [{k: v for k, v in r.items() if k != "agent_response"}
                          for r in all_results],
        "agent_responses": {r["category"]: r.get("agent_response","") for r in all_results},
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  {G}✅ Full evaluation report saved → {col(out_path, W)}{RST}\n")


if __name__ == "__main__":
    main()
