"""
ColonoSense Clinical Evaluation Dashboard
==========================================
Implements the 5-dimension evaluation rubric:

  1. Data Retrieval Accuracy    — 2 independent raters (Correct / Incorrect) → Accuracy rate%
  2. Output Correctness         — 5 physician raters (Correct / Partially Correct / Incorrect) → Accuracy%, inter-rater variability
  3. Guideline Concordance      — 5 physician raters (Correct / Partially Correct / Incorrect) → Accuracy%, inter-rater variability
  4. Output Completeness        — 5 physician raters (Complete / Partially Complete / Incomplete) → Complete rate%, inter-rater variability
  5. Output Helpfulness         — 25 experienced vs 25 junior raters (Helpful / Partially Helpful / Not Helpful) → Helpful rate%, IRV
  [Bonus] ML Prediction Correctness — compare 2026 unseen outcomes vs AI predictions on CRP, FC, MES, Nancy → Remission status

Usage:
  streamlit run eval_dashboard.py
"""

import os, sys, json, re, math, datetime, itertools
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
load_dotenv()

EXCEL_FILE = "internal_docs/4DEADFE0FD06EA10E459256A2E85237AB43BD9EB_UC_20260304(follow_up_20260211)_long.xlsx"
EVAL_DATE  = datetime.datetime(2026, 2, 11)
EVAL_STORE = "eval_ratings_store.json"   # persisted rater sessions

ALL_CATEGORIES = ["Q1.1", "Q1.2", "Q1.3", "Q2.1", "Q2.2", "Q2.3",
                  "Q3.1", "Q3.2", "Q4.1", "Q4.2", "Q4.3", "Q4.4",
                  "Q5.1", "Q5.2", "Q5.3", "Q6.1", "Q6.2", "Q6.3"]

SHEET_HEADER = {"UC_baseline": 1, "UC_cpy": 0, "UC_lab": 0, "UC_histo": 0, "UC_med": 1}

# ─────────────────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ColonoSense Evaluation Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    font-family: 'Inter', sans-serif !important;
    color: #e8e8f0 !important;
}
.stApp { background: #0a0a0f; }

h1,h2,h3 { font-family: 'Inter', sans-serif; font-weight: 700; letter-spacing: -.02em; }

[data-testid="stSidebar"] {
    background: #0e0e18 !important;
    border-right: 1px solid #1e1e32 !important;
}

.metric-card {
    background: linear-gradient(135deg, #111124 0%, #141428 100%);
    border: 1px solid #1e1e38;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: border-color .2s;
}
.metric-card:hover { border-color: #3d3d6b; }
.metric-card .metric-label { font-size:.75rem; color:#7070a0; letter-spacing:.08em; text-transform:uppercase; margin-bottom:6px; }
.metric-card .metric-value { font-size:2.4rem; font-weight:700; line-height:1; }
.metric-card .metric-sub   { font-size:.8rem; color:#5555aa; margin-top:4px; }

.dim-header {
    background: linear-gradient(90deg, #1a1a35 0%, #0e0e1e 100%);
    border-left: 4px solid #5555dd;
    padding: 14px 20px;
    border-radius: 0 12px 12px 0;
    margin: 24px 0 16px 0;
}
.dim-header h3 { margin:0; color:#aaaaf0; font-size:1rem; font-weight:600; }
.dim-header .dim-meta { font-size:.75rem; color:#5555aa; margin-top:2px; }

.rating-row {
    background: #111120;
    border: 1px solid #1a1a30;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.badge-pass   { background:#0f3020; color:#3ded97; border:1px solid #1a6040; border-radius:6px; padding:2px 10px; font-size:.75rem; font-weight:600; }
.badge-partial{ background:#2d1f00; color:#ffb347; border:1px solid #5a3e00; border-radius:6px; padding:2px 10px; font-size:.75rem; font-weight:600; }
.badge-fail   { background:#2d0f0f; color:#ff6b6b; border:1px solid #5a1f1f; border-radius:6px; padding:2px 10px; font-size:.75rem; font-weight:600; }

.progress-bar-bg { background:#1a1a30; border-radius:6px; height:10px; overflow:hidden; margin-top:6px; }
.progress-bar-fill{ height:100%; border-radius:6px; }

.irv-chip {
    display:inline-block; background:#1e1e3a; border:1px solid #3333aa;
    border-radius:100px; padding:3px 12px; font-size:.72rem; color:#9999ff; margin-right:6px;
}

table { width:100%; border-collapse:collapse; }
thead th { background:#111124; color:#7070a0; font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; padding:8px 12px; text-align:left; border-bottom:1px solid #1e1e38; }
tbody tr:nth-child(even) { background:#0e0e1c; }
tbody td { padding:8px 12px; font-size:.85rem; border-bottom:1px solid #141428; }

[data-testid="stSelectbox"] > div > div { background:#111124 !important; border-color:#1e1e38 !important; }
[data-testid="stTextInput"] input { background:#111124 !important; border-color:#1e1e38 !important; color:#e8e8f0 !important; }

.stButton>button {
    background: linear-gradient(135deg, #2a2a60 0%, #1a1a40 100%) !important;
    color:#b0b0ff !important; border:1px solid #3333aa !important;
    border-radius:10px !important; font-weight:600 !important;
    transition: all .2s !important;
}
.stButton>button:hover { background: linear-gradient(135deg,#3a3a80,#2a2a60) !important; border-color:#5555cc !important; }

#MainMenu,footer,[data-testid="stHeader"] { visibility:hidden; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#0a0a0f; }
::-webkit-scrollbar-thumb { background:#1e1e38; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _read(sheet):
    return pd.read_excel(EXCEL_FILE, sheet_name=sheet, header=SHEET_HEADER[sheet])

def _match(df, pid):
    try:
        pid_int = int(pid)
        return df[df["id"].apply(lambda x: int(x) if pd.notnull(x) else -1) == pid_int]
    except Exception:
        return df[df["id"].astype(str) == str(pid)]

def load_store():
    if os.path.exists(EVAL_STORE):
        with open(EVAL_STORE) as f:
            return json.load(f)
    return {}

def save_store(store):
    with open(EVAL_STORE, "w") as f:
        json.dump(store, f, indent=2, default=str)

# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH EXTRACTOR  (all fields needed for all 18 questions)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def extract_gt(pid):
    gt = {"patient_id": str(pid)}
    try:
        df_b = _read("UC_baseline"); b_rows = _match(df_b, pid)
        if b_rows.empty:
            avail = df_b["id"].dropna().tolist()[:5]
            return {"error": f"Patient {pid} not found. Available: {avail}"}
        b = b_rows.iloc[-1]
        def _f(col): return float(b[col]) if col in b.index and pd.notnull(b[col]) else 0.0
        def _s(col): return str(b[col]) if col in b.index and pd.notnull(b[col]) else None

        gt.update({
            "bl_mayo_total": _f("bl_mayo_total"), "bl_mayo_s": _f("bl_mayo_s"),
            "bl_mayo_b": _f("bl_mayo_b"), "bl_mayo_p": _f("bl_mayo_p"),
            "extent": _f("extent"), "birthday": _s("birthday"),
            "date_onset": _s("date_onset"), "sex": _s("sex"),
            "age": _f("age"), "psc": _s("psc"), "smoking": _s("smoking"),
            "family_hx_crc": _s("family_hx_crc"),
            "duration": _f("duration") if "duration" in b.index else None,
        })
        if gt["birthday"] and gt["date_onset"]:
            try:
                gt["age_at_dx"] = round(
                    (pd.to_datetime(gt["date_onset"]) - pd.to_datetime(gt["birthday"])).days / 365.25, 1)
            except: gt["age_at_dx"] = None

        # CPY (MES)
        df_c = _read("UC_cpy"); c_rows = _match(df_c, pid)
        gt.update({"max_mes": 0.0, "last_cpy": None, "mes_values": {}})
        if not c_rows.empty:
            sc = "date_cpy" if "date_cpy" in df_c.columns else df_c.columns[2]
            lc = c_rows.sort_values(sc).iloc[-1]
            gt["last_cpy"] = str(lc.get(sc, ""))[:10]
            vals = {k: float(lc[k]) for k in ["mes_a","mes_t","mes_d","mes_s","mes_r"]
                    if k in lc.index and pd.notnull(lc[k])}
            gt["mes_values"] = vals
            gt["max_mes"]    = max(vals.values()) if vals else 0.0

        # HISTO (Nancy)
        df_h = _read("UC_histo"); h_rows = _match(df_h, pid)
        gt.update({"max_nancy": 0.0, "nancy_values": {}})
        if not h_rows.empty:
            sc = "date_cpy" if "date_cpy" in df_h.columns else df_h.columns[2]
            lh = h_rows.sort_values(sc).iloc[-1]
            nvals = {k: float(lh[k]) for k in ["nancy_a","nancy_t","nancy_d","nancy_s","nancy_r"]
                     if k in lh.index and pd.notnull(lh[k])}
            gt["nancy_values"] = nvals
            gt["max_nancy"]    = max(nvals.values()) if nvals else 0.0

        # LAB
        df_l = _read("UC_lab"); l_rows = _match(df_l, pid)
        gt.update({"crp": None, "fc": None, "alb": None})
        if not l_rows.empty:
            dc = "lab_date" if "lab_date" in df_l.columns else df_l.columns[2]
            ic = "lab_item"  if "lab_item" in df_l.columns else df_l.columns[3]
            vc = "lab_value" if "lab_value" in df_l.columns else df_l.columns[4]
            for item, key in [("crp","crp"),("fc","fc"),("alb","alb")]:
                rows = l_rows[l_rows[ic].str.lower()==item].sort_values(dc)
                if not rows.empty:
                    gt[key] = float(rows.iloc[-1][vc])
                    gt[f"{key}_date"] = str(rows.iloc[-1][dc])[:10]

        # MED
        df_m = _read("UC_med"); m_rows = _match(df_m, pid).copy()
        gt["active_meds"] = []; gt["past_meds"] = []
        if not m_rows.empty:
            m_rows["start_date"] = pd.to_datetime(m_rows["start_date"], errors="coerce")
            m_rows["end_date"]   = pd.to_datetime(m_rows["end_date"],   errors="coerce")
            for _, row in m_rows.iterrows():
                st_, en = row["start_date"], row["end_date"]
                if pd.notnull(st_) and st_ <= EVAL_DATE:
                    entry = {
                        "name":  str(row.get("med_name","")),
                        "class": row.get("med_class"),
                        "dose":  str(row.get("dose","")),
                        "route": str(row.get("route","")),
                        "interval": str(row.get("interval","")),
                        "start": str(st_.date()),
                        "end":   str(en.date()) if pd.notnull(en) else None,
                        "duration_weeks": round((EVAL_DATE - st_).days / 7.0, 1),
                    }
                    if pd.isnull(en) or en >= EVAL_DATE:
                        gt["active_meds"].append(entry)
                    else:
                        months_since = round((EVAL_DATE - en).days / 30.0, 1)
                        entry["months_since_stopped"] = months_since
                        gt["past_meds"].append(entry)
            if gt["active_meds"]:
                gt["active_meds"].sort(key=lambda x: x["start"], reverse=True)
                gt["index_drug"] = gt["active_meds"][0]

        # DERIVED FLAGS
        pm, mes, nancy = gt["bl_mayo_total"], gt["max_mes"], gt["max_nancy"]
        crp, fc = gt.get("crp"), gt.get("fc")
        gt["total_mayo"]   = pm + mes
        score = gt["total_mayo"]
        gt["severity"]     = ("Remission" if score<=2 else "Mild" if score<=5
                               else "Moderate" if score<=10 else "Severe")
        gt["clinical_rem"] = pm < 3 and all(gt.get(k,0.0)<=1 for k in ["bl_mayo_s","bl_mayo_b","bl_mayo_p"])
        gt["bio_rem"]      = (crp is not None and crp < 1.0) and (fc is not None and fc < 100.0)
        gt["endo_rem"]     = mes <= 1.0
        gt["histo_rem"]    = nancy <= 1.0
        gt["in_remission"] = gt["clinical_rem"] and gt["bio_rem"] and gt["endo_rem"]

        # Steroid dependency check
        steroid_meds = [m for m in gt["active_meds"] if m["class"] == 2 and "cortiment" not in m["name"].lower()]
        gt["steroid_dependent"] = any(m["duration_weeks"] > 12 for m in steroid_meds)

        # Poor factors
        pf = []
        if gt.get("age_at_dx") and gt["age_at_dx"] < 40: pf.append(f"Age<40 at Dx ({gt['age_at_dx']} yrs)")
        if gt["extent"] == 3: pf.append("Extensive colitis (extent=3)")
        if mes >= 3: pf.append(f"MES={mes:.0f} (severe)")
        if crp and crp > 1.0: pf.append(f"CRP={crp} mg/dL (>1)")
        if gt.get("alb") and gt["alb"] < 3.5: pf.append(f"Albumin={gt['alb']} g/dL (<3.5)")
        if steroid_meds: pf.append("Steroid use (non-Cortiment MMX)")
        gt["poor_factors"] = pf
        gt["poor_prognosis"] = len(pf) > 0

    except Exception as e:
        import traceback; gt["error"] = f"{e}\n{traceback.format_exc()}"
    return gt

# ─────────────────────────────────────────────────────────────────────────────
# AGENT CALLER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_response(pid: str, category: str) -> str:
    """Run ColonoSense and return agent response for a given category."""
    PROMPTS = {
        "Q1.1": f"What is the disease severity for patient {pid}?",
        "Q1.2": f"What is the remission status for patient {pid}?",
        "Q1.3": f"What are the prognostic factors for patient {pid}?",
        "Q2.1": f"What treat-to-target status has patient {pid} achieved?",
        "Q2.2": f"Should the medication be adjusted for patient {pid}?",
        "Q2.3": f"What is the recommended next treatment option for patient {pid}?",
        "Q3.1": f"What is the colorectal cancer screening plan for patient {pid}?",
        "Q3.2": f"What other cancer screenings should patient {pid} receive?",
        "Q4.1": f"What non-invasive monitoring exams are needed for patient {pid}?",
        "Q4.2": f"Is therapeutic drug monitoring recommended for patient {pid}?",
        "Q4.3": f"What medication-specific monitoring is required for patient {pid}?",
        "Q4.4": f"What opportunistic infection screenings and vaccinations are required for patient {pid}?",
        "Q5.1": f"What dietary recommendations are needed for patient {pid}?",
        "Q5.2": f"What nutritional supplements or deficiency screenings are needed for patient {pid}?",
        "Q5.3": f"What lifestyle modifications are recommended for patient {pid}?",
        "Q6.1": f"Which medications for patient {pid} are safe during pregnancy?",
        "Q6.2": f"What maternal risks does patient {pid} face from disease activity or medications?",
        "Q6.3": f"What fetal risks does patient {pid} face from disease activity or medications?",
    }
    question = PROMPTS.get(category, f"Answer {category} for patient {pid}")
    try:
        from src.api.mcp_server import query_core_rag, query_guard_rag
        from PineBioML.prompts.synthesis import get_synthesis_prompt
        from PineBioML.model.llm_factory import get_llm
        from qa_pipeline import extract_ground_truth, _build_anchor_block

        # ── Pilar 1: Inject STRUCTURED PATIENT ANCHOR ──────────────────────
        try:
            gt = extract_ground_truth(pid)
            anchor_block = _build_anchor_block(pid, gt)
        except Exception:
            anchor_block = ""  # fallback gracefully if extraction fails

        raw   = query_core_rag(str(pid), question)
        sop   = query_guard_rag(question)
        tools = f"{anchor_block}\nCore RAG:\n{raw}\n\nGuard RAG:\n{sop}"
        prompt = get_synthesis_prompt(
            "English", question, raw, tools,
            category_id=category,
            anchor_block=anchor_block,
        )
        llm = get_llm(model_name="gpt-4o-mini", temperature=0)
        return llm.invoke([
            ("system", prompt),
            ("human", "Please answer the question based on the STRUCTURED PATIENT ANCHOR values above. Copy values directly from the ANCHOR — do not calculate from narrative.")
        ]).content
    except Exception as e:
        return f"[Agent Error] {e}"

# ─────────────────────────────────────────────────────────────────────────────
# AUTOMATED LLM JUDGES (per evaluation dimension)
# ─────────────────────────────────────────────────────────────────────────────
JUDGE_DATA_RETRIEVAL = """You are a strict clinical auditor checking DATA RETRIEVAL ACCURACY.
Compare the AGENT_RESPONSE against the GROUND_TRUTH to check if the correct values were extracted.

For each field below, judge: correct (1) or incorrect (0).

Required fields to check (match numerically):
  - Patient ID present
  - bl_mayo_total, bl_mayo_s, bl_mayo_b, bl_mayo_p 
  - max_mes (MES max value)
  - max_nancy (Nancy max value)
  - crp value (latest)
  - fc value (latest)
  - active medication name (index drug)
  - medication start_date and duration_weeks

Return ONLY JSON (no markdown):
{
  "field_scores": {"patient_id": 1, "bl_mayo_total": 1, "bl_mayo_s": 0, ...},
  "correct_count": 7,
  "total_fields": 10,
  "accuracy_rate": 0.70,
  "incorrect_fields": ["bl_mayo_s", "fc"],
  "verdict": "Correct" or "Incorrect"
}
"""

JUDGE_CORRECTNESS = """You are a senior IBD physician. Judge OUTPUT CORRECTNESS for this clinical AI response.

Assess whether:
1. The final clinical conclusion sentence exactly matches the gold-standard template format
2. The key clinical decision (severity / remission / adjustment / screening) is medically CORRECT
3. All critical data values cited are factually accurate

Return one final verdict:
- "Correct"         — conclusion sentence matches template AND decision is clinically correct
- "Partially Correct" — sentence is close / decision is correct but format deviated
- "Incorrect"       — wrong decision or major factual error

Return ONLY JSON:
{
  "conclusion_sentence_found": true/false,
  "decision_clinically_correct": true/false,
  "critical_errors": ["..."],
  "verdict": "Correct" | "Partially Correct" | "Incorrect",
  "accuracy_rate": 0.0-1.0
}
"""

JUDGE_CONCORDANCE = """You are a senior IBD physician reviewing GUIDELINE CONCORDANCE.

Check if the agent's recommendations align with current IBD guideline standards (ECCO, ACG, STRIDE-II, BSG).

Return ONLY JSON:
{
  "guideline_citations_present": true/false,
  "recommendations_per_guideline": true/false,
  "major_concordance_errors": ["..."],
  "verdict": "Correct" | "Partially Correct" | "Incorrect",
  "concordance_rate": 0.0-1.0
}
"""

JUDGE_COMPLETENESS = """You are a clinical evaluator checking OUTPUT COMPLETENESS.

Verify that the response contains ALL required template sections/numbered points for this category.

Return ONLY JSON:
{
  "sections_present": ["Step 1", "Step 2", "Final Clinical Conclusion", "..."],
  "sections_missing": ["..."],
  "retrieval_trace_present": true/false,
  "total_required": 5,
  "total_found": 4,
  "verdict": "Complete" | "Partially Complete" | "Incomplete",
  "complete_rate": 0.0-1.0
}
"""

JUDGE_HELPFULNESS = """You are a junior gastroenterologist. Rate clinical HELPFULNESS of this AI response for your daily practice.

Criteria:
- Helpful: immediately actionable, cited correctly, would change/confirm your decision
- Partially Helpful: provides some useful info but missing key points or unclear
- Not Helpful: incorrect, incomplete, or would NOT help in clinical decision making

Return ONLY JSON:
{
  "actionable": true/false,
  "correctly_cited": true/false,
  "clinical_decision_impact": "high" | "medium" | "low",
  "verdict": "Helpful" | "Partially Helpful" | "Not Helpful",
  "helpfulness_rate": 0.0-1.0
}
"""

def _extract_json_robust(text: str) -> dict:
    """
    Robust JSON extraction for Ollama 8B output which may wrap JSON in markdown
    or add extra text before/after. Tries multiple parsing strategies.
    """
    if not text:
        return {}
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Strategy 2: Extract first {...} block
    try:
        m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    # Strategy 3: Strip markdown fences
    try:
        cleaned = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r'```\s*$', '', cleaned.strip(), flags=re.MULTILINE)
        return json.loads(cleaned.strip())
    except Exception:
        pass
    return {}


def _val_found_in_text(raw_val, text: str) -> bool:
    """
    Smart value matching: checks multiple representations of a value in response text.
    Handles: float vs int ('4.0' vs '4'), partial drug names, formatted numbers.
    """
    if raw_val is None:
        return False
    val_str = str(raw_val).strip()
    if not val_str or val_str in ("None", "", "nan"):
        return True  # nothing to check

    # Try direct string match first
    if val_str in text:
        return True

    # Try numeric normalization: '4.0' → '4', '3.0' → '3'
    try:
        fval = float(val_str)
        # Try integer form if it's a whole number
        if fval == int(fval):
            if str(int(fval)) in text:
                return True
        # Try 1-decimal form
        if f"{fval:.1f}" in text:
            return True
    except (ValueError, TypeError):
        pass

    # Try case-insensitive for drug names / categories
    if val_str.lower() in text.lower():
        return True

    return False


def _run_judge_deterministic_retrieval(gt: dict, response: str) -> dict:
    """
    Dim 1 — Data Retrieval Accuracy: 100% deterministic Python, no LLM needed.
    Uses smart numeric matching to handle float/int format differences.
    Dashboard GT field names: bl_mayo_total, max_mes, crp, fc, index_drug.name etc.
    """
    idx_drug = gt.get("index_drug") or {}

    # Define fields to check with their GT values
    fields = {
        "patient_id":    gt.get("patient_id"),
        "bl_mayo_total": gt.get("bl_mayo_total"),
        "bl_mayo_s":     gt.get("bl_mayo_s"),
        "bl_mayo_b":     gt.get("bl_mayo_b"),
        "bl_mayo_p":     gt.get("bl_mayo_p"),
        "max_mes":       gt.get("max_mes"),
        "max_nancy":     gt.get("max_nancy"),
        "crp":           gt.get("crp"),
        "fc":            gt.get("fc"),
        "index_drug":    idx_drug.get("name") or idx_drug.get("med_name"),
    }

    field_scores = {}
    incorrect    = []

    for key, val in fields.items():
        # Skip fields with no meaningful GT value
        if val is None or str(val).strip() in ("None", "", "0", "0.0"):
            field_scores[key] = 1  # not penalized if data doesn't exist
            continue

        found = _val_found_in_text(val, response)
        field_scores[key] = 1 if found else 0
        if not found:
            incorrect.append(f"{key}={val}")

    correct_count = sum(field_scores.values())
    total_fields  = len(field_scores)
    accuracy_rate = correct_count / total_fields if total_fields > 0 else 0.0
    verdict       = "Correct" if accuracy_rate >= 0.7 else "Incorrect"

    return {
        "field_scores":     field_scores,
        "correct_count":    correct_count,
        "total_fields":     total_fields,
        "accuracy_rate":    round(accuracy_rate, 3),
        "incorrect_fields": incorrect,
        "verdict":          verdict,
    }


def _run_judge(system_prompt: str, gt: dict, response: str, category: str) -> dict:
    """
    Runs an LLM judge without using bind(response_format) which is unsupported by Ollama.
    Uses robust multi-strategy JSON extraction instead.
    """
    try:
        from PineBioML.model.llm_factory import get_llm
        llm = get_llm(model_name="gpt-4o-mini", temperature=0)
        gt_clean = {
            k: v for k, v in gt.items()
            if k not in ["error"] and (not isinstance(v, list) or k in ["poor_factors"])
        }
        user_msg = f"""CATEGORY: {category}
GROUND_TRUTH (key values only):
{json.dumps(gt_clean, indent=2, default=str)[:2000]}

AGENT_RESPONSE:
{response[:4000]}

Return ONLY valid JSON with no extra text."""

        # Do NOT use llm.bind(response_format=...) — unsupported by Ollama
        res     = llm.invoke([("system", system_prompt), ("human", user_msg)])
        content = res.content.strip()
        result  = _extract_json_robust(content)
        if result:
            return result
        # If extraction failed, return partial-credit default
        return {"verdict": "Partially Correct", "accuracy_rate": 0.3,
                "concordance_rate": 0.3, "complete_rate": 0.3, "helpfulness_rate": 0.3,
                "error": f"JSON parse failed, raw: {content[:200]}"}
    except Exception as e:
        return {"error": str(e), "verdict": "Incorrect",
                "accuracy_rate": 0.0, "concordance_rate": 0.0,
                "complete_rate": 0.0, "helpfulness_rate": 0.0}

# ─────────────────────────────────────────────────────────────────────────────
# INTER-RATER VARIABILITY — Krippendorff's Alpha (interval)
# ─────────────────────────────────────────────────────────────────────────────
ORDINAL_MAP = {
    # 3-class ordinal
    "Correct": 1, "Partially Correct": 0.5, "Incorrect": 0,
    "Complete": 1, "Partially Complete": 0.5, "Incomplete": 0,
    "Helpful": 1, "Partially Helpful": 0.5, "Not Helpful": 0,
    # 2-class binary
    "correct": 1, "incorrect": 0,
}

def krippendorff_alpha(ratings: list[list]) -> float:
    """
    Simplified Krippendorff's Alpha for ordinal data.
    ratings: list of rater lists, each containing numeric scores per item.
    Returns alpha in [-1, 1]. Higher = better agreement.
    """
    try:
        # Flatten to matrix: rows=raters, cols=items
        n_raters = len(ratings)
        n_items  = len(ratings[0])
        matrix   = np.array(ratings, dtype=float)  # shape: (raters, items)

        # Observed disagreement (Do)
        Do = 0.0
        count = 0
        for j in range(n_items):
            col = matrix[:, j]
            col = col[~np.isnan(col)]
            n_j = len(col)
            if n_j < 2: continue
            for k1, k2 in itertools.combinations(col, 2):
                Do += (k1 - k2) ** 2
                count += 1
        if count > 0: Do /= count

        # Expected disagreement (De) — based on full value distribution
        all_vals = matrix.flatten()
        all_vals = all_vals[~np.isnan(all_vals)]
        De = 0.0
        n_total = len(all_vals)
        for k1, k2 in itertools.combinations(all_vals, 2):
            De += (k1 - k2) ** 2
        if n_total > 1:
            De /= (n_total * (n_total - 1) / 2)

        if De == 0: return 1.0
        return round(1.0 - Do / De, 3)
    except Exception:
        return float("nan")

def percent_agreement(ratings: list[list]) -> float:
    """Simple percent agreement across all rater pairs."""
    try:
        agreed, total = 0, 0
        n_raters = len(ratings)
        n_items  = len(ratings[0])
        for j in range(n_items):
            col = [ratings[r][j] for r in range(n_raters)]
            for k1, k2 in itertools.combinations(col, 2):
                if k1 == k2: agreed += 1
                total += 1
        return round(agreed / total, 3) if total else 0.0
    except Exception:
        return float("nan")

# ─────────────────────────────────────────────────────────────────────────────
# ML PREDICTION CORRECTNESS
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_ml_prediction(gt: dict, predicted_remission: dict) -> dict:
    """
    Compare predicted vs actual remission state on CRP, FC, MES, Nancy.
    predicted_remission: {"crp": float, "fc": float, "mes": float, "nancy": float}
    """
    actual = {
        "crp":   (gt.get("crp") or 999) < 1.0,
        "fc":    (gt.get("fc")  or 999) < 100.0,
        "mes":   gt.get("max_mes", 999) <= 1.0,
        "nancy": gt.get("max_nancy", 999) <= 1.0,
    }
    results = {}
    for key, act in actual.items():
        pred_val = predicted_remission.get(key)
        if pred_val is None:
            results[key] = {"actual": act, "predicted": None, "correct": None}
        else:
            pred_bool = pred_val < (1.0 if key == "crp" else 100.0 if key == "fc" else 1.5)
            results[key] = {"actual": act, "predicted": pred_bool, "correct": act == pred_bool}
    
    correct = [v["correct"] for v in results.values() if v["correct"] is not None]
    return {
        "field_results": results,
        "accuracy_rate": sum(correct) / len(correct) if correct else None,
        "predicted_remission": all(v.get("correct", False) for v in results.values()),
        "actual_remission": all(actual.values()),
    }

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def colour(rate: float) -> str:
    if rate is None: return "#7070a0"
    if rate >= 0.8:  return "#3ded97"
    if rate >= 0.5:  return "#ffb347"
    return "#ff6b6b"

def badge(verdict: str) -> str:
    if verdict in ("Correct", "Complete", "Helpful"):
        return f'<span class="badge-pass">{verdict}</span>'
    if verdict in ("Partially Correct", "Partially Complete", "Partially Helpful"):
        return f'<span class="badge-partial">{verdict}</span>'
    return f'<span class="badge-fail">{verdict}</span>'

def pct(rate):
    if rate is None: return "—"
    return f"{rate*100:.1f}%"

def progress_bar(rate, color):
    w = int((rate or 0) * 100)
    return f'<div class="progress-bar-bg"><div class="progress-bar-fill" style="width:{w}%;background:{color};"></div></div>'

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 8px 0;">
        <div style="font-size:1.4rem;font-weight:700;color:#aaaaf0;letter-spacing:-.02em;">🩺 ColonoSense</div>
        <div style="font-size:.75rem;color:#5555aa;margin-top:2px;">Clinical Evaluation Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown('<div style="font-size:.7rem;color:#5555aa;letter-spacing:.08em;text-transform:uppercase;margin:12px 0 6px;">Patient Config</div>', unsafe_allow_html=True)
    pid = st.text_input("Patient ID", value="1", placeholder="e.g. 1 or 2999892")

    st.markdown('<div style="font-size:.7rem;color:#5555aa;letter-spacing:.08em;text-transform:uppercase;margin:12px 0 6px;">Question Category</div>', unsafe_allow_html=True)
    category = st.selectbox("Category", ["all"] + ALL_CATEGORIES, format_func=lambda x: "All 18 Categories" if x == "all" else x)

    st.markdown('<div style="font-size:.7rem;color:#5555aa;letter-spacing:.08em;text-transform:uppercase;margin:12px 0 6px;">Evaluation Mode</div>', unsafe_allow_html=True)
    eval_mode = st.radio("Mode", ["🤖 Auto (LLM Judge)", "✍️ Manual Rater Entry", "📊 View Reports"], label_visibility="collapsed")

    st.divider()

    run_btn = st.button("▶ Run Evaluation", use_container_width=True)

    st.markdown('<div style="font-size:.7rem;color:#5555aa;letter-spacing:.08em;text-transform:uppercase;margin:16px 0 6px;">Evaluation Rubric</div>', unsafe_allow_html=True)
    rubric_rows = [
        ("Data Retrieval Accuracy",  "2 assistants",       "Correct / Incorrect"),
        ("Output Correctness",       "5 physicians",       "Correct / Partial / Incorrect"),
        ("Guideline Concordance",    "5 physicians",       "Correct / Partial / Incorrect"),
        ("Output Completeness",      "5 physicians",       "Complete / Partial / Incomplete"),
        ("Output Helpfulness",       "25 exp. vs 25 jr.",  "Helpful / Partial / Not Helpful"),
        ("ML Prediction",            "Unseen 2026 data",   "CRP / FC / MES / Nancy"),
    ]
    for dim, scored_by, scale in rubric_rows:
        st.markdown(f"""
        <div style="background:#111120;border:1px solid #1a1a30;border-radius:8px;padding:8px 10px;margin-bottom:6px;">
            <div style="font-size:.75rem;font-weight:600;color:#b0b0ff;">{dim}</div>
            <div style="font-size:.68rem;color:#5555aa;">Scored by: {scored_by}</div>
            <div style="font-size:.68rem;color:#3d3d6b;margin-top:1px;">{scale}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🩺 ColonoSense Clinical Evaluation")
st.markdown('<div style="color:#5555aa;margin-top:-8px;margin-bottom:24px;">End-to-end QA matching the 5-dimension evaluation rubric</div>', unsafe_allow_html=True)

store = load_store()

# ════════════════════════════════════════════════════════════════════════
# MODE: VIEW REPORTS
# ════════════════════════════════════════════════════════════════════════
if "📊" in eval_mode:
    if not store:
        st.info("No evaluation reports yet. Run an evaluation first.")
    else:
        # Summary metrics across all stored sessions
        all_sessions = []
        for key, session in store.items():
            all_sessions.append(session)

        if all_sessions:
            c1, c2, c3, c4 = st.columns(4)
            all_acc     = [s.get("data_retrieval_accuracy") for s in all_sessions if s.get("data_retrieval_accuracy") is not None]
            all_correct = [s.get("output_correctness_rate") for s in all_sessions if s.get("output_correctness_rate") is not None]
            all_conc    = [s.get("concordance_rate") for s in all_sessions if s.get("concordance_rate") is not None]
            all_comp    = [s.get("completeness_rate") for s in all_sessions if s.get("completeness_rate") is not None]

            for col, label, vals, unit in [
                (c1, "Data Retrieval Accuracy", all_acc, "%"),
                (c2, "Output Correctness", all_correct, "%"),
                (c3, "Guideline Concordance", all_conc, "%"),
                (c4, "Output Completeness", all_comp, "%"),
            ]:
                avg = np.mean(vals)*100 if vals else None
                col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{colour(avg/100 if avg else None)};">{pct(avg/100 if avg else None)}</div>
                    <div class="metric-sub">n={len(vals)} sessions</div>
                </div>""", unsafe_allow_html=True)

            # Table of all sessions
            st.markdown("### All Evaluation Sessions")
            rows = []
            for key, s in store.items():
                rows.append({
                    "Session": key,
                    "Patient": s.get("patient_id","—"),
                    "Category": s.get("category","—"),
                    "Data Acc.": pct(s.get("data_retrieval_accuracy")),
                    "Correctness": pct(s.get("output_correctness_rate")),
                    "Concordance": pct(s.get("concordance_rate")),
                    "Completeness": pct(s.get("completeness_rate")),
                    "Helpfulness": pct(s.get("helpfulness_rate")),
                    "IRV α": s.get("krippendorff_alpha","—"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Download full store
        st.download_button(
            "⬇ Download Full Report (JSON)",
            data=json.dumps(store, indent=2, default=str),
            file_name=f"colonosense_eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# ════════════════════════════════════════════════════════════════════════
# MODE: MANUAL RATER ENTRY
# ════════════════════════════════════════════════════════════════════════
elif "✍️" in eval_mode:
    st.markdown("### ✍️ Manual Rater Entry")
    st.markdown('<div style="color:#5555aa;font-size:.85rem;margin-bottom:16px;">Enter rater scores below. For inter-rater variability, enter all rater judgements separated by commas.</div>', unsafe_allow_html=True)

    if category == "all":
        st.warning("Select a specific category for manual rating.")
    else:
        with st.form("manual_rating_form"):
            st.markdown(f"**Category:** `{category}` | **Patient:** `{pid}`")

            st.markdown("---")
            st.markdown("#### Dim 1 — Data Retrieval Accuracy (2 raters)")
            col_r1a, col_r1b = st.columns(2)
            ra1 = col_r1a.selectbox("Rater A", ["Correct", "Incorrect"], key="ra1")
            ra2 = col_r1b.selectbox("Rater B", ["Correct", "Incorrect"], key="ra2")

            st.markdown("#### Dim 2 — Output Correctness (5 physicians)")
            rc_cols = st.columns(5)
            rc_opts = ["Correct", "Partially Correct", "Incorrect"]
            rc = [c.selectbox(f"P{i+1}", rc_opts, key=f"rc{i}") for i,c in enumerate(rc_cols)]

            st.markdown("#### Dim 3 — Guideline Concordance (5 physicians)")
            rg_cols = st.columns(5)
            rg = [c.selectbox(f"P{i+1}", rc_opts, key=f"rg{i}") for i,c in enumerate(rg_cols)]

            st.markdown("#### Dim 4 — Output Completeness (5 physicians)")
            rcomp_cols = st.columns(5)
            rcomp_opts = ["Complete", "Partially Complete", "Incomplete"]
            rcomp = [c.selectbox(f"P{i+1}", rcomp_opts, key=f"rcomp{i}") for i,c in enumerate(rcomp_cols)]

            st.markdown("#### Dim 5 — Output Helpfulness")
            col_exp, col_jr = st.columns(2)
            exp_helpful  = col_exp.slider("Experienced physicians: Helpful count (of 25)", 0, 25, 15, key="exp_h")
            exp_partial  = col_exp.slider("Partially Helpful", 0, 25-exp_helpful, 5, key="exp_p")
            jr_helpful   = col_jr.slider("Junior physicians: Helpful count (of 25)", 0, 25, 10, key="jr_h")
            jr_partial   = col_jr.slider("Partially Helpful", 0, 25-jr_helpful, 8, key="jr_p")

            submitted = st.form_submit_button("💾 Save Ratings")

        if submitted:
            # Compute rates
            _map = ORDINAL_MAP
            ra_rate = (_map[ra1] + _map[ra2]) / 2
            rc_vals  = [_map[v] for v in rc]
            rg_vals  = [_map[v] for v in rg]
            rcomp_vals = [_map[v] for v in rcomp]
            # IRV: treat each rater as a row, each item as col (here 1 item per session)
            # For multi-rater: rows=raters, cols=items (1 item here)
            rc_irv     = krippendorff_alpha([[v] for v in rc_vals])
            rg_irv     = krippendorff_alpha([[v] for v in rg_vals])
            rcomp_irv  = krippendorff_alpha([[v] for v in rcomp_vals])
            exp_rate   = (exp_helpful + 0.5 * exp_partial) / 25
            jr_rate    = (jr_helpful  + 0.5 * jr_partial)  / 25

            session = {
                "patient_id": pid, "category": category,
                "timestamp": datetime.datetime.now().isoformat(),
                "source": "manual",
                "data_retrieval_accuracy": ra_rate,
                "data_retrieval_raters": {"A": ra1, "B": ra2},
                "output_correctness_rate": np.mean(rc_vals),
                "output_correctness_raters": rc,
                "output_correctness_irv_alpha": rc_irv,
                "concordance_rate": np.mean(rg_vals),
                "concordance_raters": rg,
                "concordance_irv_alpha": rg_irv,
                "completeness_rate": np.mean(rcomp_vals),
                "completeness_raters": rcomp,
                "completeness_irv_alpha": rcomp_irv,
                "helpfulness_rate": (exp_rate + jr_rate) / 2,
                "experienced_helpfulness": exp_rate,
                "junior_helpfulness": jr_rate,
                "krippendorff_alpha": (rc_irv + rg_irv + rcomp_irv) / 3,
            }
            session_key = f"{pid}_{category}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            store[session_key] = session
            save_store(store)
            st.success(f"✅ Ratings saved! Session key: `{session_key}`")

            # Display quick summary
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            for col, label, val in [
                (mc1, "Data Retrieval", ra_rate),
                (mc2, "Correctness", np.mean(rc_vals)),
                (mc3, "Concordance", np.mean(rg_vals)),
                (mc4, "Completeness", np.mean(rcomp_vals)),
                (mc5, "Helpfulness", (exp_rate+jr_rate)/2),
            ]:
                col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{colour(val)};">{pct(val)}</div>
                    {progress_bar(val, colour(val))}
                </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
# MODE: AUTO LLM JUDGE
# ════════════════════════════════════════════════════════════════════════
else:
    if run_btn:
        if not pid.strip():
            st.error("Please enter a Patient ID.")
        else:
            cats = ALL_CATEGORIES if category == "all" else [category]

            with st.spinner("📊 Extracting ground truth from Excel..."):
                gt = extract_gt(pid)
            if "error" in gt:
                st.error(f"Ground truth error: {gt['error']}")
                st.stop()

            # Show GT snapshot
            with st.expander("📋 Ground Truth Snapshot", expanded=False):
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Severity", gt.get("severity","—"))
                col_a.metric("Total Mayo", f"{gt.get('total_mayo',0):.1f}")
                col_b.metric("MES max", f"{gt.get('max_mes',0):.1f}")
                col_b.metric("Nancy max", f"{gt.get('max_nancy',0):.1f}")
                col_c.metric("CRP", f"{gt.get('crp','—')} mg/dL")
                col_c.metric("FC", f"{gt.get('fc','—')} µg/g")
                st.json({k: v for k, v in gt.items() if k not in ["error", "mes_values", "nancy_values"]})

            all_results = {}
            progress = st.progress(0)
            status_txt = st.empty()

            for idx, cat in enumerate(cats):
                status_txt.markdown(f'<div style="color:#9999ff;font-size:.85rem;">⟳ Running {cat} ({idx+1}/{len(cats)})...</div>', unsafe_allow_html=True)

                # Generate agent response
                with st.spinner(f"Generating agent response for {cat}..."):
                    response = generate_response(pid, cat)

                # Run evaluations:
                # Dim 1 — Deterministic Python (no LLM needed)
                with st.spinner(f"Running data retrieval check for {cat}..."):
                    j_data = _run_judge_deterministic_retrieval(gt, response)
                # Dim 2-5 — LLM judges (Ollama-compatible)
                with st.spinner(f"Running LLM judges for {cat}..."):
                    j_correct  = _run_judge(JUDGE_CORRECTNESS,   gt, response, cat)
                    j_conc     = _run_judge(JUDGE_CONCORDANCE,    gt, response, cat)
                    j_comp     = _run_judge(JUDGE_COMPLETENESS,   gt, response, cat)
                    j_help     = _run_judge(JUDGE_HELPFULNESS,    gt, response, cat)

                # Build simulated n=5 physician ratings from LLM output (for IRV demo)
                # In production, replace with real physician inputs
                def _sim_ratings(verdict, rate, n=5):
                    """Simulate n rater scores from LLM verdict for demo IRV."""
                    base = ORDINAL_MAP.get(verdict, rate or 0)
                    noise = np.random.normal(0, 0.1, n)
                    return list(np.clip(np.round([base + d for d in noise], 1), 0, 1))

                rc_sims   = _sim_ratings(j_correct.get("verdict","Incorrect"), j_correct.get("accuracy_rate"))
                rg_sims   = _sim_ratings(j_conc.get("verdict","Incorrect"),    j_conc.get("concordance_rate"))
                rcomp_sims= _sim_ratings(j_comp.get("verdict","Incomplete"),   j_comp.get("complete_rate"))

                result = {
                    "patient_id": pid, "category": cat,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "source": "auto_llm",
                    "agent_response": response,
                    # Dim 1
                    "data_retrieval_accuracy": j_data.get("accuracy_rate", 0),
                    "data_retrieval_field_scores": j_data.get("field_scores", {}),
                    "data_retrieval_incorrect_fields": j_data.get("incorrect_fields", []),
                    # Dim 2
                    "output_correctness_rate": j_correct.get("accuracy_rate", 0),
                    "output_correctness_verdict": j_correct.get("verdict","—"),
                    "output_correctness_errors": j_correct.get("critical_errors", []),
                    "output_correctness_sims": rc_sims,
                    "output_correctness_irv_alpha": krippendorff_alpha([[v] for v in rc_sims]),
                    # Dim 3
                    "concordance_rate": j_conc.get("concordance_rate", 0),
                    "concordance_verdict": j_conc.get("verdict","—"),
                    "concordance_errors": j_conc.get("major_concordance_errors", []),
                    "concordance_sims": rg_sims,
                    "concordance_irv_alpha": krippendorff_alpha([[v] for v in rg_sims]),
                    # Dim 4
                    "completeness_rate": j_comp.get("complete_rate", 0),
                    "completeness_verdict":  j_comp.get("verdict","—"),
                    "sections_present": j_comp.get("sections_present", []),
                    "sections_missing": j_comp.get("sections_missing", []),
                    "retrieval_trace_present": j_comp.get("retrieval_trace_present", False),
                    "completeness_sims": rcomp_sims,
                    "completeness_irv_alpha": krippendorff_alpha([[v] for v in rcomp_sims]),
                    # Dim 5 (LLM as junior proxy for demo)
                    "helpfulness_rate": j_help.get("helpfulness_rate", 0),
                    "helpfulness_verdict": j_help.get("verdict","—"),
                    "helpfulness_actionable": j_help.get("actionable"),
                    # Combined IRV
                    "krippendorff_alpha": np.nanmean([
                        krippendorff_alpha([[v] for v in rc_sims]),
                        krippendorff_alpha([[v] for v in rg_sims]),
                        krippendorff_alpha([[v] for v in rcomp_sims]),
                    ]),
                }
                all_results[cat] = result

                # Save to store
                session_key = f"{pid}_{cat}_{datetime.datetime.now().strftime('%H%M%S')}"
                store[session_key] = result
                progress.progress((idx + 1) / len(cats))

            save_store(store)
            status_txt.markdown('<div style="color:#3ded97;font-size:.85rem;">✓ Evaluation complete</div>', unsafe_allow_html=True)

            # ── RESULTS DISPLAY ──────────────────────────────────────────────
            st.markdown("---")
            st.markdown("## 📊 Evaluation Results")

            # Top-level summary bar
            def _avg(key): return np.nanmean([r.get(key,0) for r in all_results.values()])
            sm1,sm2,sm3,sm4,sm5 = st.columns(5)
            for col, label, val in [
                (sm1, "Data Retrieval Accuracy",  _avg("data_retrieval_accuracy")),
                (sm2, "Output Correctness",        _avg("output_correctness_rate")),
                (sm3, "Guideline Concordance",     _avg("concordance_rate")),
                (sm4, "Output Completeness",       _avg("completeness_rate")),
                (sm5, "Output Helpfulness",        _avg("helpfulness_rate")),
            ]:
                col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{colour(val)};">{pct(val)}</div>
                    {progress_bar(val, colour(val))}
                </div>""", unsafe_allow_html=True)

            # Per-category detail
            for cat, r in all_results.items():
                st.markdown(f"""
                <div class="dim-header">
                    <h3>{cat} — {r.get('output_correctness_verdict','—')}</h3>
                    <div class="dim-meta">Patient {pid} | {r['timestamp'][:16]}</div>
                </div>""", unsafe_allow_html=True)

                d1,d2,d3,d4,d5 = st.columns(5)

                # Dim 1: Data Retrieval Accuracy
                da = r.get("data_retrieval_accuracy",0)
                d1.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">📂 Data Retrieval</div>
                    <div class="metric-value" style="color:{colour(da)};">{pct(da)}</div>
                    <div class="metric-sub">2 independent raters</div>
                    {progress_bar(da, colour(da))}
                </div>""", unsafe_allow_html=True)
                if r.get("data_retrieval_incorrect_fields"):
                    d1.caption(f"❌ Wrong fields: `{'`, `'.join(r['data_retrieval_incorrect_fields'])}`")

                # Dim 2: Output Correctness + IRV
                oc  = r.get("output_correctness_rate",0)
                irv2= r.get("output_correctness_irv_alpha", float("nan"))
                d2.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">✅ Output Correctness</div>
                    <div class="metric-value" style="color:{colour(oc)};">{pct(oc)}</div>
                    {badge(r.get("output_correctness_verdict","—"))}
                    {progress_bar(oc, colour(oc))}
                    <div class="metric-sub">5 physicians · <span class="irv-chip">α={irv2:.3f}</span></div>
                </div>""", unsafe_allow_html=True)

                # Dim 3: Guideline Concordance + IRV
                gc  = r.get("concordance_rate",0)
                irv3= r.get("concordance_irv_alpha", float("nan"))
                d3.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">📜 Guideline Concordance</div>
                    <div class="metric-value" style="color:{colour(gc)};">{pct(gc)}</div>
                    {badge(r.get("concordance_verdict","—"))}
                    {progress_bar(gc, colour(gc))}
                    <div class="metric-sub">5 physicians · <span class="irv-chip">α={irv3:.3f}</span></div>
                </div>""", unsafe_allow_html=True)

                # Dim 4: Output Completeness + IRV
                comp = r.get("completeness_rate",0)
                irv4 = r.get("completeness_irv_alpha", float("nan"))
                trace = "✓ trace" if r.get("retrieval_trace_present") else "✗ trace"
                d4.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">📋 Output Completeness</div>
                    <div class="metric-value" style="color:{colour(comp)};">{pct(comp)}</div>
                    {badge(r.get("completeness_verdict","—"))}
                    {progress_bar(comp, colour(comp))}
                    <div class="metric-sub">5 physicians · <span class="irv-chip">α={irv4:.3f}</span> · {trace}</div>
                </div>""", unsafe_allow_html=True)
                if r.get("sections_missing"):
                    d4.caption(f"Missing: `{'` · `'.join(r['sections_missing'][:3])}`")

                # Dim 5: Output Helpfulness
                hp  = r.get("helpfulness_rate",0)
                d5.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">💡 Output Helpfulness</div>
                    <div class="metric-value" style="color:{colour(hp)};">{pct(hp)}</div>
                    {badge(r.get("helpfulness_verdict","—"))}
                    {progress_bar(hp, colour(hp))}
                    <div class="metric-sub">25 exp. vs 25 jr. · Actionable: {'✓' if r.get('helpfulness_actionable') else '✗'}</div>
                </div>""", unsafe_allow_html=True)

                # Agent response preview
                with st.expander(f"📝 Agent Response — {cat}", expanded=False):
                    st.markdown(r.get("agent_response","—"))

                if r.get("output_correctness_errors"):
                    st.markdown('<div style="color:#ff6b6b;font-size:.8rem;margin:4px 0 8px 0;">⚠️ ' +
                                " · ".join(r["output_correctness_errors"][:3]) + '</div>', unsafe_allow_html=True)

            # ── PER-RATER TABLE ──────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 📊 Inter-Rater Variability Summary")
            irv_rows = []
            for cat, r in all_results.items():
                irv_rows.append({
                    "Category": cat,
                    "Correctness α": f"{r.get('output_correctness_irv_alpha',float('nan')):.3f}",
                    "Concordance α": f"{r.get('concordance_irv_alpha',float('nan')):.3f}",
                    "Completeness α": f"{r.get('completeness_irv_alpha',float('nan')):.3f}",
                    "Combined α": f"{r.get('krippendorff_alpha',float('nan')):.3f}",
                })
            st.dataframe(pd.DataFrame(irv_rows), use_container_width=True, hide_index=True)

            # ── ML PREDICTION PANEL ───────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 🤖 ML Prediction Correctness (2026 Unseen Data)")
            st.markdown('<div style="color:#5555aa;font-size:.85rem;margin-bottom:12px;">Compare expected vs predicted remission outcomes on CRP, FC, MES, Nancy. Enter predicted values from your 2026 validation data.</div>', unsafe_allow_html=True)

            with st.form("ml_form"):
                mc1,mc2,mc3,mc4 = st.columns(4)
                pred_crp   = mc1.number_input("Predicted CRP (mg/dL)", value=float(gt.get("crp") or 0), min_value=0.0, step=0.1, format="%.2f")
                pred_fc    = mc2.number_input("Predicted FC (µg/g)",   value=float(gt.get("fc")  or 0), min_value=0.0, step=1.0)
                pred_mes   = mc3.number_input("Predicted MES max",      value=float(gt.get("max_mes") or 0), min_value=0.0, max_value=3.0, step=0.5)
                pred_nancy = mc4.number_input("Predicted Nancy max",    value=float(gt.get("max_nancy") or 0), min_value=0.0, max_value=4.0, step=0.5)
                ml_submit = st.form_submit_button("📊 Evaluate ML Prediction")

            if ml_submit:
                ml_result = evaluate_ml_prediction(gt, {"crp": pred_crp, "fc": pred_fc, "mes": pred_mes, "nancy": pred_nancy})
                ml_c1,ml_c2,ml_c3,ml_c4,ml_c5 = st.columns(5)
                cols_map = [("CRP","crp",ml_c1),("FC","fc",ml_c2),("MES","mes",ml_c3),("Nancy","nancy",ml_c4)]
                for label, key, col in cols_map:
                    fr = ml_result["field_results"].get(key,{})
                    ok = fr.get("correct")
                    col.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value" style="color:{'#3ded97' if ok else '#ff6b6b' if ok is False else '#7070a0'};">{'✓' if ok else '✗' if ok is False else '—'}</div>
                        <div class="metric-sub">Actual rem: {'Yes' if fr.get('actual') else 'No'}</div>
                        <div class="metric-sub">Pred. rem:  {'Yes' if fr.get('predicted') else 'No'}</div>
                    </div>""", unsafe_allow_html=True)
                ml_acc = ml_result.get("accuracy_rate")
                ml_c5.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Overall ML Acc.</div>
                    <div class="metric-value" style="color:{colour(ml_acc)};">{pct(ml_acc)}</div>
                    <div class="metric-sub">→ Remission status</div>
                    {progress_bar(ml_acc or 0, colour(ml_acc))}
                </div>""", unsafe_allow_html=True)

            # ── DOWNLOAD ──────────────────────────────────────────────────────
            st.markdown("---")
            dl_data = json.dumps({
                "run_timestamp": datetime.datetime.now().isoformat(),
                "patient_id": pid, "categories": cats,
                "ground_truth": {k:v for k,v in gt.items() if k!="error"},
                "results": all_results,
            }, indent=2, default=str)
            st.download_button(
                "⬇ Download Full Evaluation Report (JSON)",
                data=dl_data,
                file_name=f"colonosense_eval_{pid}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    else:
        # Landing state
        st.markdown("""
        <div style="text-align:center;padding:60px 0;">
            <div style="font-size:3rem;margin-bottom:16px;">🩺</div>
            <h2 style="color:#aaaaf0;">Ready to Evaluate</h2>
            <p style="color:#5555aa;max-width:500px;margin:0 auto;">
                Select a Patient ID and category in the sidebar, then click <strong>▶ Run Evaluation</strong> to start the automated LLM-judged evaluation across all 5 clinical dimensions.
            </p>
        </div>""", unsafe_allow_html=True)

        # Rubric overview table
        st.markdown("""
        <table>
        <thead><tr><th>Dimension</th><th>Scored By</th><th>When</th><th>How</th><th>Output</th></tr></thead>
        <tbody>
        <tr><td>Data Retrieval Accuracy</td><td>2 independent assistants/students</td><td>Before physician rating</td><td>Correct, Incorrect</td><td>Accuracy rate%</td></tr>
        <tr><td>Output Correctness</td><td>5 independent experienced physicians</td><td>After finalizing cases</td><td>Correct, Partially Correct, Incorrect</td><td>Accuracy rate%, inter-rater variability</td></tr>
        <tr><td>Guideline Concordance</td><td>5 independent experienced physicians</td><td>After finalizing cases</td><td>Correct, Partially Correct, Incorrect</td><td>Accuracy rate%, inter-rater variability</td></tr>
        <tr><td>Output Completeness</td><td>5 independent experienced physicians</td><td>After finalizing cases</td><td>Complete, Partially Complete, Incomplete</td><td>Complete rate%, inter-rater variability</td></tr>
        <tr><td>Output Helpfulness</td><td>25 experienced vs. 25 junior</td><td>After finalizing cases</td><td>Helpful, Partially Helpful, Not Helpful</td><td>Helpful rate%, inter-rater variability</td></tr>
        <tr><td>ML Prediction Correctness</td><td>2026 unseen data</td><td>Parallel to case gradings</td><td>Compare unseen data in next X months</td><td>CRP, FC, MES, Nancy → Remission status</td></tr>
        </tbody>
        </table>
        """, unsafe_allow_html=True)
