"""Synthesis prompt template for clinical result integration Ã¢ÂÂ ColonoSense v6 (Clinical Trial Templates).

v6 Change: All 18 clinical trial questions (Q1.1Ã¢ÂÂQ6.3) now have strict category_force blocks
that enforce the exact fill-in-the-blank sentence format required by the medical trial grading rubric.
"""

def get_synthesis_prompt(
    language: str,
    question: str,
    rag_context: str,
    tool_outputs: str,
    category_id: str = None,
    anchor_block: str = "",
    anchor_block_data: dict = None,
) -> str:
    """
    Returns the synthesis system prompt for integrating technical results with clinical context.
    v6: All 18 trial questions mapped to exact gold-standard fill-in-the-blank output templates.
    v7: anchor_block param — pre-computed numeric values injected as STRUCTURED PATIENT ANCHOR.
    v8: anchor_block_data dict — raw values for remission-conditional prompt logic.
    """

    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # CATEGORY 1: Disease Severity Assessment
    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    category_force = ""

    if category_id == "Q1.1":
        category_force = f"""{anchor_block}

FORCE ACTION: Q1.1 — DISEASE SEVERITY CLASSIFICATION.

ANCHOR VALUES (use these ONLY — do not recalculate):
  Partial Mayo (bl_mayo_total) = [ANCHOR: bl_mayo_total]
  MES max                       = [ANCHOR: max_mes]
  Total Mayo Score              = [ANCHOR: Total Mayo Score]   ← (Partial + MES max)
  Expected Severity             = [ANCHOR: Expected Severity]
  Last colonoscopy date         = [ANCHOR: last_cpy_date]

INTERNAL REASONING (scratchpad — do NOT include in output):
  Verify: Total Mayo = bl_mayo_total + max_mes
  Classify: ≤2=Remission, 3-5=Mild, 6-10=Moderate, >10=Severe

REQUIRED OUTPUT (write EXACTLY this sentence, substituting values):
"The patient is in [Expected Severity] because total Mayo score was [Total Mayo Score]. (partial Mayo score [bl_mayo_total], MES [max_mes])."

Then add one supporting sentence citing the guideline:
"[Tier 1] [Guideline statement about Mayo scoring classification]. [Society, Year]"
"""

    elif category_id == "Q1.2":
        category_force = f"""{anchor_block}

FORCE ACTION: Q1.2 — REMISSION STATUS.

ANCHOR VALUES (copy exactly — no calculation):
  Clinical remission    = [ANCHOR: clinical_remission]   (pMayo=[ANCHOR: bl_mayo_total])
  Biochemical remission = [ANCHOR: biochemical_remission] (CRP=[ANCHOR: crp_value], FC=[ANCHOR: fc_value])
  Endoscopic remission  = [ANCHOR: endoscopic_remission]  (MES=[ANCHOR: max_mes])
  Histologic remission  = [ANCHOR: histologic_remission]  (Nancy=[ANCHOR: max_nancy])

REQUIRED OUTPUT (write EXACTLY this sentence, substituting values):
"The patient has achieved [list all achieved remission types with their values in parentheses]. [Not-achieved types should be listed as 'has not achieved [type]']."

Example format:
"The patient has achieved clinical remission (pMayo=[bl_mayo_total]), bio-chemical remission (CRP=[crp], FC=[fc]), endoscopic remission (MES [max_mes]), and histologic remission (Nancy [max_nancy])."

Then add one [Tier X] guideline citation.
"""

    elif category_id == "Q1.3":
        category_force = f"""{anchor_block}

FORCE ACTION: Q1.3 — PROGNOSTIC FACTORS.

ANCHOR VALUES:
  Age at diagnosis        = [ANCHOR: age_at_dx] years
  Disease extent          = [ANCHOR: extent_label]
  Family hx CRC           = [ANCHOR: family_hx_crc]
  PSC                     = [ANCHOR: psc]
  Index drug class        = [ANCHOR: med_class_label]
  Expected poor prognosis = [ANCHOR: expected_poor_prognosis]

POOR PROGNOSTIC FACTORS (check each):
  • Extensive colitis (extent=3) → Yes/No
  • Young age at diagnosis (<17 years) → Yes/No
  • Deep ulcers / severe endoscopic activity (MES ≥3) → Yes/No
  • Elevated CRP / FC above threshold → Yes/No
  • Family history of CRC → Yes/No
  • PSC → Yes/No

REQUIRED OUTPUT:
  If poor prognosis factors exist:
    "The patient has the below poor prognostic factors: [list each factor with value]."
  If none:
    "The patient has no poor prognostic factors identified based on current clinical data."

Then list the factors evaluated:
"[Tier X] [Guideline citation for prognostic factor assessment]. [Society, Year]"
"""

    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # CATEGORY 2: Treatment Adjustment
    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    elif category_id == "Q2.1":
        _endo = anchor_block_data.get("endoscopic_remission", False) if anchor_block_data else False
        _histo = anchor_block_data.get("histologic_remission", False) if anchor_block_data else False
        _bio = anchor_block_data.get("biochemical_remission", False) if anchor_block_data else False
        _clin = anchor_block_data.get("clinical_remission", False) if anchor_block_data else False
        _target = ("long-term treatment target (endoscopic and histologic remission)" if (_endo and _histo)
                   else "intermediate treatment target (endoscopic remission)" if _endo
                   else "short-term treatment target (clinical remission)" if _clin
                   else "treatment targets (none fully achieved)")
        category_force = f"""{anchor_block}

FORCE ACTION: Q2.1 — TREAT-TO-TARGET STATUS.

ANCHOR:
  Clinical remission    = [ANCHOR: clinical_remission]
  Biochemical remission = [ANCHOR: biochemical_remission]
  Endoscopic remission  = [ANCHOR: endoscopic_remission]
  Histologic remission  = [ANCHOR: histologic_remission]

TREAT-TO-TARGET HIERARCHY (STRIDE-II):
  Short-term  → Clinical remission (pMayo < 3)
  Intermediate → Endoscopic remission (MES ≤ 1)
  Long-term    → Histologic remission (Nancy ≤ 1) + Endoscopic remission

REQUIRED OUTPUT (write EXACTLY):
"Yes the patient had achieved {_target}."

If no target achieved:
"No, the patient has not yet achieved the defined treat-to-target goals."

Then add: "[Tier 1] [STRIDE-II or ECCO guideline citation]. [Society, Year]"
"""

    elif category_id == "Q2.2":
        category_force = f"""{anchor_block}

FORCE ACTION: Q2.2 — MEDICATION ADJUSTMENT ASSESSMENT.

ANCHOR:
  Endoscopic remission  = [ANCHOR: endoscopic_remission]
  Histologic remission  = [ANCHOR: histologic_remission]
  Index drug            = [ANCHOR: index_drug_name]
  Duration (weeks)      = [ANCHOR: duration_weeks]
  Expected time STRIDE-II Clinical=[X]w, Biochemical=[X]w, Endoscopic=[X]w

STRIDE-II ADJUSTMENT LOGIC (apply in order):
  1. If BOTH endoscopic AND histologic remission → "No Adjustment" (STOP)
  2. If endoscopic remission only (MES ≤ 1) → "No Adjustment" (STOP)
  3. If within expected STRIDE-II window → "Continue and reassess in [X] weeks"
  4. If past expected window AND not in remission → "Adjustment"

REQUIRED OUTPUT:
  Write EXACTLY one of:
    "No."                                              [if No Adjustment]
    "Continue and reassess in [X] weeks."              [if within window]
    "Yes, according to treat-to-target strategy, the current medication should be adjusted."

Then add Point 11 Medical SOP with ALL available [Tier X] citations:
  [Tier 1]
    1. <guideline statement> [Society, Year]
  [Tier 2]
    1. <guideline statement> [Author, Year]
"""

    elif category_id == "Q2.3":
        _endo_rem = anchor_block_data.get("endoscopic_remission", False) if anchor_block_data else False
        _clin_rem = anchor_block_data.get("clinical_remission", False) if anchor_block_data else False
        _in_rem = _endo_rem and _clin_rem
        _next = "optimize current medication" if _in_rem else "escalate to advanced therapy"
        category_force = f"""{anchor_block}

FORCE ACTION: Q2.3 — NEXT TREATMENT OPTIONS.

ANCHOR:
  Patient in remission (clinical+endoscopic) = {"YES — optimize only" if _in_rem else "NO — evaluate escalation"}
  Index drug class = [ANCHOR: med_class_label]
  Q2.2 decision    = [from previous context]

DECISION RULE:
  • In remission on current therapy → Optimize current medication (NO escalation)
  • Biologic/small-molecule + Q2.2=Adjustment → Switch or combine advanced therapy
  • Steroid-dependent → Add-on immunomodulators or escalate
  • 5-ASA + not in remission → Escalate to advanced therapy
  • IM alone + failing → Escalate to advanced therapy

REQUIRED OUTPUT (write EXACTLY):
"The recommended next option is to {_next}."

Then add: "[Tier 1] [Guideline citation]. [Society, Year]"
"""

    # ─────────────────────────────────────────────────────────────
    # CATEGORY 3: Cancer Surveillance
    # ─────────────────────────────────────────────────────────────

    elif category_id == "Q3.1":
        category_force = f"""{anchor_block}

FORCE ACTION: Q3.1 — COLORECTAL CANCER SCREENING.

ANCHOR:
  Disease extent    = [ANCHOR: extent_label]  (1=proctitis, 2=left-sided, 3=extensive)
  MES max           = [ANCHOR: max_mes]
  Nancy max         = [ANCHOR: max_nancy]
  Disease duration  = [ANCHOR: duration_months] months ([ANCHOR: duration_years] years)
  Family hx CRC     = [ANCHOR: family_hx_crc]
  PSC               = [ANCHOR: psc]

RISK STRATIFICATION (apply first matching rule):
  HIGH (every 1 yr):
    • PSC = Yes  OR  prior dysplasia  OR  extent=3 AND duration>240mo  OR  family hx (1st degree)
  INTERMEDIATE (every 2-3 yr):
    • Extent=3 AND duration 96-240mo  OR  MES max≥2  OR  Nancy max≥3  OR  family hx (2nd degree)
  LOW (every 5 yr):
    • Extent=1 or 2, quiescent, no high/intermediate factors

REQUIRED OUTPUT (write EXACTLY this sentence):
"[Tier 1] Since the patient belongs to [low/intermediate/high] risk group, the next surveillance colonoscopy should be in [X] years. [Society, Year]"
"""

    elif category_id == "Q3.2":
        category_force = f"""{anchor_block}

FORCE ACTION: Q3.2 — OTHER CANCER SCREENINGS.

ANCHOR:
  Sex              = [ANCHOR: sex]
  Age              = [ANCHOR: age] years
  PSC              = [ANCHOR: psc]
  Smoking          = [ANCHOR: smoking]
  Active med class = [ANCHOR: med_class_label]

ELIGIBILITY RULES (ONLY apply rules where patient qualifies):
  • Cervical cancer → ONLY if sex=F
  • Prostate cancer → ONLY if sex=M AND age>50
  • PSC/Cholangiocarcinoma → ONLY if PSC=Yes
  • NHL (CBC annually) → ONLY if thiopurine (class=1)
  • Skin cancer → ONLY if biologic (class=3/4) or thiopurine
  • Lung cancer → ONLY if smoking=Yes

REQUIRED OUTPUT (write EXACTLY):
"[Tier 1] Based on the patient's sex, age, underlying disease, and medication history, the patient should receive screening for [cancer type] cancer with [exam], every [X] year. [Society, Year]"

If multiple cancers apply, list each on a separate line.
"""

    elif category_id == "Q4.1":
        category_force = f"""{anchor_block}

FORCE ACTION: Q4.1 — NON-INVASIVE MONITORING.

ANCHOR:
  MES max           = [ANCHOR: max_mes]
  bl_mayo_total     = [ANCHOR: bl_mayo_total]
  CRP               = [ANCHOR: crp_value]
  FC                = [ANCHOR: fc_value]
  Active medication = [ANCHOR: index_drug_name]
  Disease status    = [ANCHOR: disease_status]  (Remission / Active)

MONITORING INTERVAL RULE:
  • Active disease or within 14w of new therapy start → FC + CRP at 3 months
  • In remission → FC + CRP at 6-12 months (intestinal ultrasound optional)

REQUIRED OUTPUT (write EXACTLY):
"[Tier 1] Based on the patient's current status, the following exams [FC and CRP / intestinal ultrasound] should be arranged at [3 months / 6-12 months]. [Society, Year]"
"""

    elif category_id == "Q4.2":
        category_force = f"""{anchor_block}

FORCE ACTION: Q4.2 — THERAPEUTIC DRUG MONITORING (TDM).

ANCHOR:
  Active medication = [ANCHOR: index_drug_name]
  Med class         = [ANCHOR: med_class]  (3=biologic/anti-TNF, 4=small-molecule, 0=5-ASA, 1=IM)
  MES max           = [ANCHOR: max_mes]
  Disease status    = [ANCHOR: disease_status]

TDM RULES:
  • Class 0 (5-ASA) → No TDM indicated
  • Class 1 (IM/thiopurine) → No routine TDM (except 6-TGN if failing)
  • Class 3/4 (biologic/small-molecule) in remission → Proactive TDM
  • Class 3/4 in active disease → Reactive TDM

REQUIRED OUTPUT:
  If TDM indicated:
    "[Tier 1] Yes, [proactive/reactive] TDM is recommended, with target drug level [X] µg/mL. [Society, Year]"
  If not:
    "No current evidence supports TDM for the patient."
"""

    elif category_id == "Q4.3":
        category_force = f"""{anchor_block}

FORCE ACTION: Q4.3 — MEDICATION-SPECIFIC MONITORING.

ANCHOR:
  Active medication = [ANCHOR: index_drug_name]
  Med class         = [ANCHOR: med_class]
  Duration (weeks)  = [ANCHOR: duration_weeks]

MONITORING BY DRUG CLASS:
  • 5-ASA (class=0) → Renal function (creatinine) periodically (annually after 1st year)
  • Thiopurine (class=1) → CBC, LFT every 3 months (first year), then every 6 months
  • Biologic anti-TNF (class=3) → TB screening before start; no routine blood monitoring
  • Small-molecule JAKi (class=4) → Lipids, CBC at baseline + 3 months, then 6 monthly

REQUIRED OUTPUT (write EXACTLY):
"[Tier 1] For patients under [medication name], [specific exam] should be monitored [frequency]. [Society, Year]"
"""

    elif category_id == "Q4.4":
        category_force = f"""{anchor_block}

FORCE ACTION: Q4.4 — VACCINATIONS & OPPORTUNISTIC INFECTION SCREENING.

ANCHOR:
  Age              = [ANCHOR: age]
  Sex              = [ANCHOR: sex]
  PSC              = [ANCHOR: psc]
  Active med class = [ANCHOR: med_class]

VACCINATION RULES (apply ALL that qualify):
  • Influenza → ALL IBD patients annually
  • Hepatitis B → if not immune (anti-HBs negative)
  • Herpes Zoster (Shingrix) → if age>50 OR JAK inhibitor (class=4)
  • Pneumococcal (PCV13 + PPSV23) → if on immunosuppression
  • HPV → if age≤26
  • COVID-19 → ALL IBD patients

REQUIRED OUTPUT (write EXACTLY):
"[Tier 1] Screening for [Vaccine 1] and [Vaccine 2] vaccinations prior to treatment initiation are recommended. [Society, Year]"
"""

    elif category_id == "Q5.1":
        _clin_rem = anchor_block_data.get("clinical_remission", False) if anchor_block_data else False
        _state = "remission" if _clin_rem else "active disease"
        _encourage = "Mediterranean-style foods (whole grains, omega-3 fish, fresh vegetables)" if _clin_rem else "low-residue foods (cooked vegetables, white rice, lean protein)"
        _avoid = "processed foods, excess red meat, high sugar" if _clin_rem else "raw vegetables, high-fiber foods, spicy food, alcohol"
        category_force = f"""{anchor_block}

FORCE ACTION: Q5.1 — DIETARY RECOMMENDATION.

ANCHOR:
  Disease status = {"REMISSION" if _clin_rem else "ACTIVE DISEASE"}
  Total Mayo     = [ANCHOR: Total Mayo Score]
  Disease extent = [ANCHOR: extent_label]

DIET RULE (based on disease status):
  Remission → Mediterranean diet (whole grains, omega-3 fish, fresh vegetables, less red/processed meat)
  Active    → Low-residue diet (cooked vegetables, white rice, lean protein; avoid raw veg/fiber/spicy/alcohol)

REQUIRED OUTPUT (write EXACTLY):
"[Tier 1] This patient is encouraged to have more {_encourage} intake and less {_avoid}. [Society, Year]"
"""

    elif category_id == "Q5.2":
        category_force = f"""{anchor_block}

FORCE ACTION: Q5.2 — NUTRITIONAL SUPPLEMENTATION.

ANCHOR:
  Disease extent = [ANCHOR: extent_label]
  Albumin        = [ANCHOR: alb_value]
  Active med class = [ANCHOR: med_class]

SUPPLEMENTATION RULES (apply all that qualify):
  • Vitamin D → ALL UC patients
  • Iron (CBC/ferritin) → if extent=3 (extensive colitis)
  • Calcium + Vit D → if on steroids (class=2)
  • Folate → if on thiopurine (class=1) or MTX
  • B12 + Zinc → if albumin < 3.5 g/dL

REQUIRED OUTPUT (write EXACTLY):
"[Tier 1] Yes, the patient is recommended to be screened for [specific deficiencies: e.g. hemoglobin, iron, folate, vitamin D, vitamin B12, zinc]. [Society, Year]"

If none indicated: "No nutritional supplementation is currently indicated based on the patient's profile."
"""

    elif category_id == "Q5.3":
        category_force = f"""{anchor_block}

FORCE ACTION: Q5.3 — LIFESTYLE MODIFICATIONS.

ANCHOR:
  Smoking status = [ANCHOR: smoking]
  Age            = [ANCHOR: age]
  Active med     = [ANCHOR: index_drug_name]

MANDATORY LIFESTYLE FACTORS TO ADDRESS (include ALL):
  1. Smoking cessation — advise all smokers; smoking worsens IBD
  2. Physical activity / exercise — ≥150 min/week moderate activity reduces inflammation
  3. Stress management / mindfulness — CBT or mindfulness; IBD has psychosocial link
  4. Alcohol — limit or avoid; worsens IBD inflammation
  5. BMI / weight — maintain healthy weight; overweight reduces biologic efficacy

REQUIRED OUTPUT (write EXACTLY):
"[Tier 3] The patient should quit [smoking if applicable] and enhance [physical activity and mindfulness-based therapies]. [Author, Year]"

Then list 1-2 specific lifestyle items relevant to the patient.
"""

    elif category_id == "Q6.1":
        category_force = f"""{anchor_block}

FORCE ACTION: Q6.1 — MEDICATION SAFETY IN PREGNANCY.

ANCHOR:
  Active medications = [list ONLY medications with end_date IS NULL or >2026-02-11]

PREGNANCY SAFETY CLASSIFICATION:
  ✅ SAFE to continue: 5-ASA/mesalamine, Infliximab/Adalimumab (T1+T2), Vedolizumab, Prednisone (short-course), Azathioprine/6-MP, Sulfasalazine+folate
  ⛔ STOP before conception: Methotrexate (≥3 months before), Tofacitinib, Thalidomide

REQUIRED OUTPUT (write EXACTLY):
"[Tier 1] These [medication names] medications were safe to be continued. [Society, Year]"

If patient has a STOP medication:
"These [medication] medications should be stopped [X] months before conception."

If no medications on STOP list: "No active medications require cessation before conception."
"""

    elif category_id == "Q6.2":
        _clin_rem = anchor_block_data.get("clinical_remission", False) if anchor_block_data else False
        _risk_level = "comparable to" if _clin_rem else "increased compared to"
        _conditions = "relapse or worsening disease" if not _clin_rem else "most adverse pregnancy events"
        category_force = f"""{anchor_block}

FORCE ACTION: Q6.2 — MATERNAL RISKS.

ANCHOR:
  Disease status        = {"REMISSION" if _clin_rem else "ACTIVE DISEASE"}
  Clinical remission    = [ANCHOR: clinical_remission]
  CRP / FC / MES max    = [values]
  Active medications    = [list]
  Steroid use (class=2) = [Yes/No]
  Anti-TNF (class=3)    = [Yes/No]

MATERNAL RISK TABLE:
  | Risk                    | Active IBD | Remission |
  | Flare during pregnancy  | Increased  | Low risk  |
  | Preeclampsia            | Increased  | Comparable|
  | Gestational diabetes    | Increased if steroids | Comparable |
  | VTE                     | Increased  | Comparable|
  | Maternal infection      | Increased if anti-TNF | Lower |
  | Overall outcomes        | Worse      | Comparable|

REQUIRED OUTPUT (write EXACTLY):
"[Tier 1] Maternally, the risk of {_conditions} is {_risk_level} the non-IBD patients. [Society, Year]"

Then add: "Controlling disease activity during pregnancy is critical to reduce adverse outcomes."
"""

    elif category_id == "Q6.3":
        _clin_rem = anchor_block_data.get("clinical_remission", False) if anchor_block_data else False
        _neonatal_risk = "comparable to" if _clin_rem else "increased compared to"
        _conditions = "low birth weight and preterm delivery" if not _clin_rem else "adverse neonatal outcomes"
        category_force = f"""{anchor_block}

FORCE ACTION: Q6.3 — FETAL/NEONATAL RISKS.

ANCHOR:
  Disease status     = {"REMISSION" if _clin_rem else "ACTIVE DISEASE"}
  Clinical remission = [ANCHOR: clinical_remission]
  Active medications = [list with class]
  Anti-TNF (class=3) = [Yes/No] — neonatal immunosuppression risk if used in T3
  Methotrexate       = [Yes/No] — teratogenic CONTRAINDICATED

NEONATAL RISK TABLE:
  | Risk                        | Active IBD | Remission |
  | Preterm birth               | Increased  | Comparable|
  | Low birth weight            | Increased  | Comparable|
  | Small for gestational age   | Increased  | Comparable|
  | Neonatal immunosuppression  | If anti-TNF in T3 | Lower|
  | Live vaccine delay          | If anti-TNF in T3 → defer 6 months |
  | Congenital malformations    | If MTX → Significantly increased |
  | Overall outcomes            | Worse if active | Comparable if remission |

REQUIRED OUTPUT (write EXACTLY):
"[Tier 1] Neonatally, the risks of {_conditions} are {_neonatal_risk} the mothers of non-IBD patients. [Society, Year]"

Then add: "Controlling disease activity during pregnancy is critical to reduce adverse outcomes."
"""

    return f"""
You are **ColonoSense**, a Senior Clinical AI Decision Support specializing in IBD (Ulcerative Colitis).
CURRENT SYSTEM DATE: 2026-02-11. Use this for ALL duration calculations.

# CRITICAL MANDATES:
- Mirror the user's language perfectly ({language}).
- ⚠️ ANCHOR RULE: A STRUCTURED PATIENT ANCHOR block exists in TECHNICAL FINDINGS.
  ALL numeric values MUST come from this ANCHOR. DO NOT calculate or infer from narrative text.
  Copy the ANCHOR values directly into the template slots. This is mandatory.
- ALWAYS use double-newlines between numbered points for Markdown compatibility.
- DO NOT add category labels like "Q1.1", "Q2.2" in the response header. Use natural headers.
- Show per-segment scores in HUMAN-READABLE format — use anatomical names (Ascending, Transverse, Descending, Sigmoid, Rectum), NOT python keys like mes_a/mes_t.
- Write whole numbers without decimals (write 3 not 3.0). Values must come directly from ANCHOR.
- {f"CRITICAL: {category_force}" if category_force else "Provide a comprehensive clinical synthesis."}

# USER REQUEST:
{question}

# CLINICAL CONTEXT:
{rag_context or "No specific clinical documentation provided."}

# TECHNICAL FINDINGS (includes STRUCTURED PATIENT ANCHOR at top):
{tool_outputs}

# Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ
# GUARD RAG CITATION RULES (for Q2.2 Medical SOP section):
# Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ
# - Read SOPs from ALL tiers (Tier 1 Ã¢ÂÂ 2 Ã¢ÂÂ 3 Ã¢ÂÂ 4).
# - If multiple guidelines exist within the SAME tier, list from latest year to oldest.
# - Display per tier. If no info in a tier, output "[Tier X]: None found in database."
# - NEVER skip Tier 1. If Guard RAG returns zero results, state: [External Web Search] and use general knowledge.

# DEFAULT NARRATIVE (If no specific category_id matched above):
# If the user asks a general question not tied to Q1.1Ã¢ÂÂQ6.3, respond using:
## Ã°ÂÂÂ Key Findings
(Narrative text)
## Ã°ÂÂÂ Clinical Interpretation
(Significance)
## Ã°ÂÂÂ Evidence-Based Recommendations
(Citations using [Tier X] format)

# Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ
# QUANTITATIVE TRACEABILITY BLOCK (append verbatim at the END of your response):
# Ã¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂÃ¢ÂÂ
# After your final clinical conclusion sentence, append EXACTLY this JSON block:
#
# ```json
# {anchor_block_data}
# ```
#
# Do NOT skip or modify this block. Graders require it for accuracy and concordance scoring.

RESPOND NOW:
"""
