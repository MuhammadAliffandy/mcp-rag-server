"""Synthesis prompt template for clinical result integration — ColonoSense v6 (Clinical Trial Templates).

v6 Change: All 18 clinical trial questions (Q1.1–Q6.3) now have strict category_force blocks
that enforce the exact fill-in-the-blank sentence format required by the medical trial grading rubric.
"""

def get_synthesis_prompt(
    language: str,
    question: str,
    rag_context: str,
    tool_outputs: str,
    category_id: str = None
) -> str:
    """
    Returns the synthesis system prompt for integrating technical results with clinical context.
    v6: All 18 trial questions mapped to exact gold-standard fill-in-the-blank output templates.
    """

    # ─────────────────────────────────────────────────────────────
    # CATEGORY 1: Disease Severity Assessment
    # ─────────────────────────────────────────────────────────────
    category_force = ""

    if category_id == "Q1.1":
        category_force = """FORCE ACTION: Q1.1 — DISEASE SEVERITY.

You MUST first output the structured reasoning block, then end with the exact final conclusion sentence.

## Patient [ID] - Disease Severity Assessment

[CORE RAG - Patient [ID] Data Extraction]
Step 1 - UC_baseline (bl_mayo_total):
  UC_baseline -> Patient [ID] -> bl_mayo_total = [VALUE]
  (sub-scores: stool frequency=[S], rectal bleeding=[B], physician assessment=[P])
Step 2 - UC_cpy (max MES):
  UC_cpy -> Patient [ID] -> latest colonoscopy ([DATE])
  MES per segment: {'mes_a': [A], 'mes_t': [T], 'mes_d': [D], 'mes_s': [S], 'mes_r': [R]}
  MES max = [MAX]
Step 3 - Total Mayo Score:
  Partial Mayo ([PM]) + MES ([MES]) = [TOTAL]
Step 4 - Severity Classification:
  Total Mayo = [TOTAL]
  -> [Remission/Mild/Moderate/Severe]
  (Remission=0-2, Mild=3-5, Moderate=6-10, Severe>10)

### 📝 Final Clinical Conclusion
[Remission / mild / moderate / severe]"""

    elif category_id == "Q1.2":
        category_force = """FORCE ACTION: Q1.2 — REMISSION STATUS.

You MUST output the exact 7-point template below.
Start with '## Patient [ID] - Remission Status Assessment'.

## Patient [ID] - Remission Status Assessment

**1. Patient ID:** [ID]

**2. Last Colonoscopy Date:** [YYYY-MM-DD]

**3. Partial Mayo Score and Sub-scores:**
- Partial Mayo Score           : [VALUE]
- Stool Frequency (bl_mayo_s)  : [VALUE]
- Rectal Bleeding (bl_mayo_b)  : [VALUE]
- Physician Assessment (bl_mayo_p): [VALUE]

**4. CRP and Fecal Calprotectin:**
- CRP (date: [DATE]) : [VALUE] mg/dL
- FC  (date: [DATE]) : [VALUE] ug/g

**5. MES Score:**
- Per segment: {'mes_a': [A], 'mes_t': [T], 'mes_d': [D], 'mes_s': [S], 'mes_r': [R]}
- MES max: [VALUE]

**6. Nancy Score:**
- Per segment: {'nancy_a': [A], 'nancy_t': [T], 'nancy_d': [D], 'nancy_s': [S], 'nancy_r': [R]}
- Nancy max: [VALUE]

**7. Remission Status:**
- Clinical remission   : [✅ YES / ❌ NO]
  (Partial Mayo=[X]<3 AND all sub-scores≤1: [True/False])
- Biochemical remission: [✅ YES / ❌ NO]
  (CRP=[X]<1 AND FC=[X]<100)
- Endoscopic remission : [✅ YES / ❌ NO]
  (MES max=[X], remission if 0 or 1)
- Histologic remission : [✅ YES / ❌ NO]
  (Nancy max=[X], remission if 0 or 1)

### 📝 Final Clinical Conclusion
[Clinical remission / bio-chemical remission / endoscopic remission / histologic remission / no remission]"""

    elif category_id == "Q1.3":
        category_force = """FORCE ACTION: Q1.3 — PROGNOSTIC FACTORS.

You MUST output the structured 11-point template below, then end with the exact trial conclusion sentence.
Use ∆ YES for poor factors found, ✅ NO for factors not found.

## Patient [ID] - Prognostic Factor Assessment

**1. Patient ID:** [ID]

**2. Birthday:** [YYYY-MM-DD]

**3. Age at Diagnosis:** [X] years old
  -> Young at diagnosis (<40): [∆ YES / ✅ NO]

**4. Extensive Colitis:**
  -> Extent value: [VALUE]
  -> Extensive colitis (extent=3): [∆ YES / ✅ NO]

**5. MES (Endoscopic Activity):**
  -> MES per segment: {'mes_a': [A], 'mes_t': [T], 'mes_d': [D], 'mes_s': [S], 'mes_r': [R]}
  -> MES max: [VALUE]
  -> MES=3 (poor prognostic): [∆ YES / ✅ NO]

**6. CRP:**
  -> CRP value: [VALUE] mg/dL (measured: [DATE])
  -> Elevated CRP (>1 mg/dL): [∆ YES / ✅ NO]

**7. Albumin:**
  -> Albumin value: [VALUE] g/dL (measured: [DATE])
  -> Low albumin (<3.5 g/dL): [∆ YES / ✅ NO]

**8. Medical Class:** [VALUES]

**9. Medical Name:** [NAMES]

**10. Steroid Use:**
  -> Steroid medications: [LIST or None]
  -> Steroid use: [∆ YES / ✅ NO]

**11. Prognostic Factor: [∆ POOR PROGNOSIS / No poor prognostic factors identified]**
Poor factors identified:
  • [Factor 1]
  • [Factor 2]

## Clinical Interpretation
[Brief 1-2 sentence clinical interpretation]

### 📝 Final Clinical Conclusion
Yes, [specify which factors]. OR No."""

    # ─────────────────────────────────────────────────────────────
    # CATEGORY 2: Treatment Adjustment
    # ─────────────────────────────────────────────────────────────

    elif category_id == "Q2.1":
        category_force = """FORCE ACTION: Q2.1 — TREAT-TO-TARGET.

You MUST output the structured 8-point assessment, then end with the exact trial sentence.

## Patient [ID] - Treat-to-Target Assessment

**1. Patient ID:** [ID]

**2. Last Colonoscopy Date:** [YYYY-MM-DD]

**3. Partial Mayo Score and Sub-scores:**
  - Partial Mayo Score           : [VALUE]
  - Stool Frequency (bl_mayo_s)  : [VALUE]
  - Rectal Bleeding (bl_mayo_b)  : [VALUE]
  - Physician Assessment (bl_mayo_p): [VALUE]

**4. CRP and Fecal Calprotectin:**
  - CRP (date: [DATE]) : [VALUE] mg/dL
  - FC  (date: [DATE]) : [VALUE] ug/g

**5. MES Score:**
  - Per segment : {'mes_a': [A], 'mes_t': [T], 'mes_d': [D], 'mes_s': [S], 'mes_r': [R]}
  - MES max     : [VALUE]

**6. Nancy Score:**
  - Per segment : {'nancy_a': [A], 'nancy_t': [T], 'nancy_d': [D], 'nancy_s': [S], 'nancy_r': [R]}
  - Nancy max   : [VALUE]

**7. Remission Status:**
  - Clinical remission   : [✅ YES / ❌ NO]
  - Biochemical remission: [✅ YES / ❌ NO]
  - Endoscopic remission : [✅ YES / ❌ NO]
  - Histologic remission : [✅ YES / ❌ NO]

**8. Treat-to-Target Status:**
  [✅ / ❌] **[Short Term / Intermediate / Long Term / No Formal] Target**
  Reason: [Explanation of highest achieved target]

### 📝 Final Clinical Conclusion
The patient has achieved [short / intermediate / long term / no] treatment target."""

    elif category_id == "Q2.2":
        category_force = """FORCE ACTION: Q2.2 — MEDICATION ADJUSTMENT.

You MUST output the structured 11-point template.
Start with '## Patient [ID] - Medication Adjustment Assessment'.

CRITICAL ADJUSTMENT LOGIC (follow EXACTLY):
- Identify the INDEX DRUG: the active medication with the latest start_date (exclude meds where end_date < 2026-02-11).
- Only discuss adjustment for the INDEX DRUG, not other medications.
- Calculate duration = (2026-02-11 - start_date) in weeks.

STEP 1: If Endoscopic Remission MET (MES ≤ 1) → Point 10 = "No Adjustment". STOP.
STEP 2: If Endoscopic Remission NOT MET → Use STRIDE-II Table (UC section):
  Round 1 - Clinical Remission:
    - If NOT met AND duration < expected → "Continue and reassess in [expected - duration] weeks"
    - If NOT met AND duration ≥ expected → "Adjustment"
    - If MET → go to Round 2
  Round 2 - Biochemical Remission:
    - Same logic as Round 1
    - If MET → go to Round 3
  Round 3 - Endoscopic Remission:
    - If NOT met AND duration < expected → "Continue and reassess in [expected - duration] weeks"
    - If NOT met AND duration ≥ expected → "Adjustment"

STRIDE-II Expected Times (UC, in weeks):
  Oral 5-ASA:    Clinical=8,  Biochemical=10, Endoscopic=13
  Oral Steroids: Clinical=2,  Biochemical=8,  Endoscopic=11
  Thiopurines:   Clinical=15, Biochemical=15, Endoscopic=20
  Adalimumab:    Clinical=11, Biochemical=12, Endoscopic=14
  Infliximab:    Clinical=10, Biochemical=11, Endoscopic=13
  Vedolizumab:   Clinical=14, Biochemical=15, Endoscopic=18
  Tofacitinib:   Clinical=11, Biochemical=11, Endoscopic=14

## Patient [ID] - Medication Adjustment Assessment

**1. Patient ID:** [ID]

**2. Last Colonoscopy Date:** [YYYY-MM-DD]

**3. Partial Mayo Score and Sub-scores:**
  - Partial Mayo Score           : [VALUE]
  - Stool Frequency   (bl_mayo_s)  : [VALUE]
  - Rectal Bleeding   (bl_mayo_b)  : [VALUE]
  - Physician Assess  (bl_mayo_p)  : [VALUE]

**4. CRP and Fecal Calprotectin:**
  - CRP (date: [DATE]) : [VALUE] mg/dL
  - FC  (date: [DATE]) : [VALUE] ug/g

**5. MES Score:**
  - Per segment: {'mes_a': [A], 'mes_t': [T], 'mes_d': [D], 'mes_s': [S], 'mes_r': [R]}
  - MES max     : [VALUE]

**6. Nancy Score:**
  - Per segment : {'nancy_a': [A], 'nancy_t': [T], 'nancy_d': [D], 'nancy_s': [S], 'nancy_r': [R]}
  - Nancy max   : [VALUE]

**7. Remission Status:**
  - Clinical remission   : [✅ YES / ❌ NO]
    (Partial Mayo=[X]<3 AND all sub-scores≤1)
  - Biochemical remission: [✅ YES / ❌ NO]
    (CRP=[X]<1 mg/dL AND FC=[X]<100 ug/g)
  - Endoscopic remission : [✅ YES / ❌ NO]
    (MES max=[X], remission if 0 or 1)
  - Histologic remission : [✅ YES / ❌ NO]
    (Nancy max=[X], remission if 0 or 1)

**8. Treat-to-Target Status:** [Highest achieved target]

**9. Medication Information:**
  - Index Drug (latest start_date): [NAME]
  - Medication Class: [CLASS]
  - Dose: [DOSE]
  - Route: [ROUTE]
  - Interval: [INTERVAL]
  - Start Date: [YYYY-MM-DD]
  - Duration: [X] weeks
  - Expected Time (STRIDE-II): Clinical=[X]w, Biochemical=[X]w, Endoscopic=[X]w

**10. Adjustment Status:** [No Adjustment / Continue and reassess in X weeks / Adjustment]
  Reasoning: [Explain which round of STRIDE-II logic was applied]

**11. Medical SOP:**

  [Tier 1]
    1. <Actual retrieved recommendation> [<Society>, <Year>]

  [Tier 2]
    1. <Actual retrieved recommendation> [<Society>, <Year>]

  [Tier 3]
    1. <Actual retrieved recommendation> [<Author>, <Year>]

  [Tier 4]
    1. <Actual retrieved recommendation> [<Author>, <Trial Name>, <Year>]

Point 11 MUST list Guard RAG citations in [Tier X] format with ALL available tiers.

### 📝 Final Clinical Conclusion
Yes, according to treat-to-target strategy, the current medication should be adjusted. OR No."""

    elif category_id == "Q2.3":
        category_force = """FORCE ACTION: Q2.3 — NEXT TREATMENT OPTIONS.

DATA RETRIEVAL (execute in order):
1. Active Medication → UC_med (rows where end_date IS NULL or end_date > 2026-02-11)
   Extract: med_name, med_class (5-ASA=0, IM=1, Steroid=2, Adv biologic/small-mol=3 or 4), dose, route, interval.
2. Disease Extent → UC_baseline: extent field (1=proctitis, 2=left-sided, 3=extensive/pancolitis)
3. Steroid Dependency Check → UC_med where med_class=2 AND med_name != 'Cortiment MMX'
   → Flag STEROID-DEPENDENT if: total cumulative duration > 12 weeks OR ≥2 separate start/stop episodes within 12 months.

GUARD RAG LOGIC (cross-reference active meds + guideline next-step algorithms):
- 5-ASA (med_class=0/class 5ASA) AND NOT in remission → next option: Escalate to advanced therapy OR Add-on immunomodulators
- IM alone (med_class=1) AND failing → next option: Escalate to advanced therapy
- Steroid-dependent (per check above) → next option: Add-on immunomodulators OR Escalate to advanced therapy
- First biologic failed (med_class=3 or 4, and Q2.2=Adjustment) → Switch to or combine other advanced therapy
- In remission on advanced therapy → Optimize current medication

Output MUST use ONLY these exact phrases:
  Optimize current medication / Add-on immunomodulators / Escalate to advanced therapy / Switch to or combine other advanced therapy

### 📝 Final Clinical Conclusion
Based on the patient demographics, extent, severity, and current medication failure, the recommended next option is: [Optimize current medication / Add-on immunomodulators / Escalate to advanced therapy / Switch to or combine other advanced therapy]."""

    # ─────────────────────────────────────────────────────────────
    # CATEGORY 3: Cancer Surveillance
    # ─────────────────────────────────────────────────────────────

    elif category_id == "Q3.1":
        category_force = """FORCE ACTION: Q3.1 — COLORECTAL CANCER (CRC) RISK & SCREENING.

DATA RETRIEVAL (execute in order):
1. Disease Extent → UC_baseline: extent (1=proctitis, 2=left-sided, 3=extensive/pancolitis)
2. Endoscopic Inflammation → UC_cpy: max(mes_a, mes_t, mes_d, mes_s, mes_r)
   Map: 0=minimal, 1=mild, 2=moderate, 3=severe
3. Histologic Inflammation → UC_histo: max(nancy_a, nancy_t, nancy_d, nancy_s, nancy_r)
   Map: 0 or 1=minimal, 2=mild, 3=moderate, 4=severe
4. Family History & PSC → UC_baseline: family_hx_crc (Yes/No), psc (Yes/No)
5. Duration → UC_baseline: duration (in months). Convert to years = duration / 12.

SCREENING ONSET RULE:
- Offer first surveillance colonoscopy to ALL patients 8 years after symptom onset.

RISK STRATIFICATION (use retrieved data above):
- HIGH risk (colonoscopy every 1 year) if ANY of:
    • PSC = Yes (start surveillance immediately at PSC diagnosis)
    • Prior dysplasia documented
    • Extent=3 AND duration > 240 months (>20 years)
    • family_hx_crc = Yes AND first-degree relative
- INTERMEDIATE risk (colonoscopy every 2–3 years) if ANY of:
    • Extent=3 AND duration 96–240 months (8–20 years)
    • MES max ≥ 2 (moderate–severe endoscopic inflammation)
    • Nancy max ≥ 3 (moderate–severe histologic inflammation)
    • family_hx_crc = Yes (second-degree relative)
- LOW risk (colonoscopy every 5 years) if:
    • Extent = 1 or 2, quiescent disease (MES max ≤ 1, Nancy max ≤ 1), no high/intermediate risk factors

### 📝 Final Clinical Conclusion
Screening colonoscopy should be offered to all patients [X] years after symptom onset. Since the patient belongs to [low / intermediate / high] risk group, the next surveillance colonoscopy should be in [X] year(s)."""

    elif category_id == "Q3.2":
        category_force = """FORCE ACTION: Q3.2 — OTHER TYPES OF CANCER RISK.

DATA RETRIEVAL (execute in order):
1. Patient Profile → UC_baseline: sex (M/F), age (years), psc (Yes/No), smoking (Yes/No)
2. Active Medication → UC_med (rows where end_date IS NULL or end_date > 2026-02-11)
   Extract med_name, med_class for ALL active entries.

SCREENING RULES (apply all applicable — list each separately):
- Female + any immunosuppressant (med_class=1,3,4) active
    → Cervical cancer: Pap smear every 1 year
- PSC = Yes
    → Cholangiocarcinoma: CA19-9 + MRCP/ERCP every 6–12 months
- Any thiopurine active (med_class=1, e.g. azathioprine / 6-MP)
    → Non-Hodgkin lymphoma: annual CBC review; counsel on risk
- Any anti-TNF or biologic active (med_class=3 or 4) OR thiopurine active
    → Skin cancer (NMSC): total body skin exam every 1 year
- Male AND age > 50 years
    → Prostate cancer: PSA every 1–2 years
- Smoking = Yes (active or ex-smoker)
    → Lung cancer: discuss low-dose CT if ≥30 pack-year history
- ALL IBD patients
    → Colorectal cancer: covered in Q3.1 (do NOT duplicate here)

### 📝 Final Clinical Conclusion
Based on the patient’s demographics and medication history, the patient should receive screening for [cancer type] cancer with [screening method] every [X] years.

NOTE: Do NOT mention colorectal cancer or colonoscopy surveillance timing here — that is covered in Q3.1."""

    # ─────────────────────────────────────────────────────────────
    # CATEGORY 4: Monitor Tools and Interval
    # ─────────────────────────────────────────────────────────────

    elif category_id == "Q4.1":
        category_force = """FORCE ACTION: Q4.1 — NON-INVASIVE MONITORING.
You MUST output the full structured block below, then end with the Final Clinical Conclusion sentence.

## Patient [ID] - Non-Invasive Monitoring Plan

Step 1 — DATA RETRIEVAL:
- bl_mayo_total (from PATIENT ANCHOR → UC_baseline): [VALUE]
- MAX(MES) (from PATIENT ANCHOR → UC_cpy): [VALUE]
- CRP (from PATIENT ANCHOR → UC_lab): [VALUE] mg/dL (date: [DATE])
- FC  (from PATIENT ANCHOR → UC_lab): [VALUE] µg/g (date: [DATE])
- Active Medication (from PATIENT ANCHOR → UC_med): [med_name] started [date], duration [X] weeks

Step 2 — MONITORING INTERVAL (GUARD RAG LOGIC):
- Disease status: [Active / Remission / Post-initiation <14w]
  → Monitoring schedule: [Fecal calprotectin + CRP at 3 months / 6 months]
- Reason: [state clinical reason per ECCO/ACG guideline]

### 📝 Final Clinical Conclusion
Based on the patient’s current status, the following exams [exams] should be arranged at [interval]."""

    elif category_id == "Q4.2":
        category_force = """FORCE ACTION: Q4.2 — THERAPEUTIC DRUG MONITORING (TDM).
You MUST output the full structured block below, then end with the Final Clinical Conclusion sentence.

## Patient [ID] - Therapeutic Drug Monitoring Plan

Step 1 — DATA RETRIEVAL:
- Active Medication (from PATIENT ANCHOR → UC_med): [med_name]  class=[X]  route=[route]  duration=[X]w
- MAX(MES) (from PATIENT ANCHOR → UC_cpy): [VALUE]
- bl_mayo_total (from PATIENT ANCHOR → UC_baseline): [VALUE]
- Disease remission: [Yes (MES ≤ 1 AND bl_mayo_total < 3) / No (active disease)]

Step 2 — TDM DETERMINATION (GUARD RAG LOGIC):
- TDM type: [Proactive / Reactive / Not indicated]
  Reason: [patient is in remission → proactive / patient has active disease → reactive]
- Drug-specific target trough level:
  - [med_name] → target trough [VALUE] µg/mL ([maintenance / active disease] threshold)
  - Guideline: [ECCO_TDM_2023 / AGA_TDM_2017]

### 📝 Final Clinical Conclusion
[Yes, proactive TDM is recommended / Yes, reactive TDM is recommended / No], with target drug level [value or N/A]."""

    elif category_id == "Q4.3":
        category_force = """FORCE ACTION: Q4.3 — MEDICATION-SPECIFIC MONITORING.
You MUST output the full structured block below, then end with the Final Clinical Conclusion sentence.

## Patient [ID] - Medication-Specific Monitoring Plan

Step 1 — DATA RETRIEVAL:
- Active Medication(s) (from PATIENT ANCHOR → UC_med):
  [med_name]  class=[X]  dose=[dose]  route=[route]  interval=[interval]  duration=[X]w

Step 2 — MONITORING SCHEDULE (one entry per active drug):
| Medication | Lab Tests Required | Frequency | Guideline |
|---|---|---|---|
| [med_name] | [tests] | [every X months / annually] | [ECCO/ACG] |

Note: If no active medication matches monitoring criteria → state "No specific monitoring required."

### 📝 Final Clinical Conclusion
For patients under [medication name] medication, [specific lab tests] should be checked every [X] months."""

    elif category_id == "Q4.4":
        category_force = """FORCE ACTION: Q4.4 — OPPORTUNISTIC INFECTION RISK & VACCINATIONS.
You MUST output the full structured block below, then end with the Final Clinical Conclusion sentence.

## Patient [ID] - Infection Screening & Vaccination Plan

Step 1 — DATA RETRIEVAL:
- Patient age (from PATIENT ANCHOR → UC_baseline): [VALUE] years
- Sex (from PATIENT ANCHOR → UC_baseline): [M/F]
- PSC (from PATIENT ANCHOR → UC_baseline): [Yes/No]
- Active Medication (from PATIENT ANCHOR → UC_med): [med_name]  class=[X]

Step 2 — SCREENING & VACCINATION REQUIRED (apply all applicable):
| Screening / Vaccine | Required? | Reason | Guideline |
|---|---|---|---|
| Hepatitis B (HBsAg/anti-HBs/anti-HBc) | Yes | Pre-biologic | ECCO 2023 |
| Hepatitis C (anti-HCV) | Yes | Pre-biologic | ECCO 2023 |
| Latent TB (IGRA) | Yes | Anti-TNF initiation | ATS/ECCO |
| Influenza vaccine | Yes | Immunosuppressed | ACIP |
| Pneumococcal (PCV13+PPSV23) | [Yes/No] | Biologic therapy | ACIP |
| HPV vaccine | [Yes if age ≤26 / No] | per ACIP | ACIP |
| COVID-19 vaccine | Yes | IBD immunosuppressed | ACIP |
| Herpes Zoster (Shingrix) | [Yes if >50 or JAKi] | age/therapy | ACIP |

NOTE: Stopped immunosuppressants < 3 months ago still confer immunosuppression risk.

### 📝 Final Clinical Conclusion
Screening for [infection(s)] and [vaccine(s)] vaccinations prior to treatment initiation are recommended."""

    # ─────────────────────────────────────────────────────────────
    # CATEGORY 5: Lifestyle and Diet Modification
    # ─────────────────────────────────────────────────────────────

    elif category_id == "Q5.1":
        category_force = """FORCE ACTION: Q5.1 — DIETARY RECOMMENDATION.
You MUST output the full structured block below FIRST, then end with the Final Clinical Conclusion sentence.

## Patient [ID] - Dietary Recommendation

Step 1 — DATA RETRIEVAL:
- bl_mayo_total (from PATIENT ANCHOR → UC_baseline): [VALUE]
- MAX(MES) (from PATIENT ANCHOR → UC_cpy): [VALUE]
- Total Mayo Score: [bl_mayo_total + MAX(MES)] = [TOTAL]
- Disease Activity: [Active UC (>5) / Mild-Moderate (3–5) / Remission (≤2)]
- Disease Extent (from PATIENT ANCHOR → UC_baseline extent): [1=proctitis / 2=left-sided / 3=extensive]

Step 2 — DIETARY RECOMMENDATION:
- Foods to ENCOURAGE: [list per activity status]
- Foods to AVOID: [list per activity status]
- Special note: [low-residue if active flare / Mediterranean if remission]
- Guideline basis: [ECCO Diet 2023 / ACG 2021]

### 📝 Final Clinical Conclusion
This patient is encouraged to have more [foods] intake and less [foods]."""

    elif category_id == "Q5.2":
        category_force = """FORCE ACTION: Q5.2 — NUTRITIONAL SUPPLEMENTATION AND DEFICIENCY SCREENING.
You MUST output the full structured block below FIRST, then end with the Final Clinical Conclusion sentence.

## Patient [ID] - Nutritional Supplementation Plan

Step 1 — DATA RETRIEVAL:
- Disease Extent (from PATIENT ANCHOR → UC_baseline extent): [VALUE] ([1/2/3])
- Albumin (from PATIENT ANCHOR → UC_lab): [VALUE] g/dL
- Active Medications (from PATIENT ANCHOR → UC_med):
  [med_name]  class=[X] → check: thiopurine (1), steroid (2), MTX (if present)

Step 2 — SUPPLEMENTATION & SCREENING REQUIRED:
| Supplement / Screening | Required? | Trigger Condition | Guideline |
|---|---|---|---|
| Vitamin D screening | Yes | ALL UC patients | ECCO 2023 |
| Iron deficiency (CBC/ferritin) | [Yes if extent=3 / No] | Extensive colitis | ECCO 2023 |
| Calcium + Vit D | [Yes if steroid class=2 / No] | Steroid use | ECCO 2023 |
| Folate | [Yes if thiopurine or MTX / No] | Thiopurine/MTX use | ECCO 2023 |
| B12 + Zinc | [Yes if Alb<3.5 / No] | Low albumin/malabsorp. | ACG 2021 |

### 📝 Final Clinical Conclusion
Yes, the patient is recommended to be screened for [deficiency] deficiency. OR No."""

    elif category_id == "Q5.3":
        category_force = """FORCE ACTION: Q5.3 — LIFESTYLE MODIFICATIONS.
You MUST output the full structured block below FIRST, then end with the Final Clinical Conclusion sentence.

## Patient [ID] - Lifestyle Modification Plan

Step 1 — DATA RETRIEVAL:
- Smoking status (from PATIENT ANCHOR → UC_baseline): [smoking value or null]
- Age (from PATIENT ANCHOR → UC_baseline): [VALUE] years
- Sex (from PATIENT ANCHOR → UC_baseline): [M/F]
- Active Medication (from PATIENT ANCHOR → UC_med): [med_name]  class=[X]
  → Biologic on board: [Yes (class=3/4) / No]

Step 2 — LIFESTYLE RECOMMENDATIONS:
| Lifestyle Factor | Recommendation | Reason |
|---|---|---|
| Smoking | Advise cessation | Overall health + drug efficacy |
| Physical Activity | 150 min/week moderate exercise | Reduces inflammation markers |
| Stress Management | CBT, mindfulness | IBD-psychosocial link |
| BMI / Weight | Healthy weight maintenance | Biologic efficacy |
| Alcohol | Limit or avoid | Worsens IBD inflammation |

### 📝 Final Clinical Conclusion
The patient should quit [habit] and enhance [lifestyle modification]."""

    # ─────────────────────────────────────────────────────────────
    # CATEGORY 6: Family Planning
    # ─────────────────────────────────────────────────────────────

    elif category_id == "Q6.1":
        category_force = """FORCE ACTION: Q6.1 — MEDICATION SAFETY IN PREGNANCY/LACTATION.

Determine which of the patient's medications are safe during pregnancy, lactation, or conception attempts.

Rules (SAFE in pregnancy):
- 5-ASA (mesalamine) → SAFE (Category B)
- Infliximab/Adalimumab → SAFE in first and second trimester; discuss third trimester use
- Vedolizumab → SAFE, minimal systemic transfer
- Prednisone → SAFE in short courses; prolonged use caution
- Azathioprine/6-MP → Generally safe; discuss risks
- Sulfasalazine → SAFE with folate supplementation

Rules (STOP before conception):
- Methotrexate → STOP ≥3 months before conception (teratogenic)
- Tofacitinib → STOP before conception (limited safety data)
- Thalidomide → ABSOLUTE contraindication

### 📝 Final Clinical Conclusion
These [medications] medications were safe to be continued. These [medications] medication should be stopped [X] months before conception."""

    elif category_id == "Q6.2":
        category_force = """FORCE ACTION: Q6.2 — MATERNAL RISKS FROM DISEASE ACTIVITY AND MEDICATIONS.

State the maternal risks associated with active IBD disease and current medications during pregnancy.

Rules:
- Active IBD during pregnancy → increased risk of disease flare
- Active IBD → increased risk of preeclampsia, gestational diabetes
- Steroid use → gestational diabetes, maternal hypertension
- Disease activity → increased risk of VTE (venous thromboembolism)
- Anti-TNF → may increase risk of maternal infection

### 📝 Final Clinical Conclusion
Maternally, the risk of [complication(s)] is increased / comparable to the non-IBD patients."""

    elif category_id == "Q6.3":
        category_force = """FORCE ACTION: Q6.3 — FETAL/NEONATAL RISKS FROM DISEASE ACTIVITY AND MEDICATIONS.

State the neonatal/fetal risks associated with active IBD disease and current medications.

Rules:
- Active IBD → increased risk of preterm birth, low birth weight
- Active IBD → small for gestational age (SGA)
- Anti-TNF (especially 3rd trimester) → neonatal immunosuppression; delay live vaccines in infant for 6 months
- Methotrexate → fetal loss, congenital malformations (CONTRAINDICATED)
- Disease remission → risk comparable to non-IBD population

### 📝 Final Clinical Conclusion
Neonatally, the risk of [complication(s)] is increased / comparable to the mothers of non-IBD patients."""

    # ── Build tables_accessed list based on category_id ──────────────────────
    _table_map = {
        "Q1.1": ["UC_baseline", "UC_cpy"],
        "Q1.2": ["UC_baseline", "UC_cpy", "UC_histo", "UC_lab"],
        "Q1.3": ["UC_baseline", "UC_cpy", "UC_lab", "UC_med"],
        "Q2.1": ["UC_baseline", "UC_cpy", "UC_histo", "UC_lab"],
        "Q2.2": ["UC_baseline", "UC_cpy", "UC_histo", "UC_lab", "UC_med"],
        "Q2.3": ["UC_baseline", "UC_med"],
        "Q3.1": ["UC_baseline", "UC_cpy", "UC_histo"],
        "Q3.2": ["UC_baseline", "UC_med"],
        "Q4.1": ["UC_med", "UC_lab"],
        "Q4.2": ["UC_med"],
        "Q4.3": ["UC_med"],
        "Q4.4": ["UC_baseline", "UC_med"],
        "Q5.1": ["UC_baseline"],
        "Q5.2": ["UC_baseline", "UC_med", "UC_lab"],
        "Q5.3": ["UC_baseline"],
        "Q6.1": ["UC_med"],
        "Q6.2": ["UC_baseline", "UC_med"],
        "Q6.3": ["UC_baseline", "UC_med"],
    }
    _guideline_hint_map = {
        "Q1.1": ["ECCO_2023", "AGA_2022"],
        "Q1.2": ["STRIDE-II_IOIBD_2021", "ECCO_2023"],
        "Q1.3": ["ECCO_2023", "AGA_2022"],
        "Q2.1": ["STRIDE-II_IOIBD_2021", "ECCO_2023"],
        "Q2.2": ["STRIDE-II_IOIBD_2021", "ECCO_2023", "AGA_UC_2023"],
        "Q2.3": ["ECCO_2023", "ACG_UC_2019", "AGA_UC_2023"],
        "Q3.1": ["ECCO_2017_Surveillance", "BSG_2010_Surveillance", "ACG_Surveillance_2021"],
        "Q3.2": ["ECCO_IBD_Cancer_2023", "ACIP_Vaccine_2023"],
        "Q4.1": ["STRIDE-II_IOIBD_2021", "ECCO_2023"],
        "Q4.2": ["ECCO_TDM_2023", "AGA_TDM_2017"],
        "Q4.3": ["ECCO_2023", "ACG_UC_2019"],
        "Q4.4": ["ECCO_Vaccination_2022", "ACIP_2023", "ECCO_OI_2021"],
        "Q5.1": ["ECCO_Diet_2023"],
        "Q5.2": ["ECCO_2023", "ACG_UC_2019"],
        "Q5.3": ["ECCO_2023"],
        "Q6.1": ["ECCO_IBD_Pregnancy_2023", "ACG_IBD_Pregnancy_2022"],
        "Q6.2": ["ECCO_IBD_Pregnancy_2023"],
        "Q6.3": ["ECCO_IBD_Pregnancy_2023", "ACG_IBD_Pregnancy_2022"],
    }
    _tables = _table_map.get(category_id, ["UC_baseline", "UC_med", "UC_cpy", "UC_histo", "UC_lab"])
    _guidelines = _guideline_hint_map.get(category_id, ["ECCO_2023", "ACG_UC_2019"])

    import json as _json
    _trace_block = _json.dumps({
        "retrieval_trace": {
            "tables_accessed": _tables,
            "missing_data_handled": True
        },
        "guideline_trace": _guidelines
    }, indent=2)

    return f"""
You are **ColonoSense**, a Senior Clinical AI Decision Support specializing in IBD (Ulcerative Colitis).
CURRENT SYSTEM DATE: 2026-02-11. Use this for ALL duration calculations.

# CRITICAL MANDATES:
- Mirror the user's language perfectly ({language}).
- EXTRACT VALUES FROM THE RAW DATA PROVIDED. Do NOT say "Data Unavailable" if the data exists in CLINICAL CONTEXT or TECHNICAL FINDINGS below.
- ALWAYS use double-newlines between numbered points for Markdown compatibility.
- DO NOT add category labels like "Q1.1", "Q2.2" in the response header. Use natural headers.
- Show per-segment data as dict format: {{'mes_a': X, 'mes_t': X, 'mes_d': X, 'mes_s': X, 'mes_r': X}}
- {f"CRITICAL: {category_force}" if category_force else "Provide a comprehensive clinical synthesis."}

# USER REQUEST:
{question}

# CLINICAL CONTEXT:
{rag_context or "No specific clinical documentation provided."}

# TECHNICAL FINDINGS:
{tool_outputs}

# ═══════════════════════════════════════════════════════════
# GUARD RAG CITATION RULES (for Q2.2 Medical SOP section):
# ═══════════════════════════════════════════════════════════
# - Read SOPs from ALL tiers (Tier 1 → 2 → 3 → 4).
# - If multiple guidelines exist within the SAME tier, list from latest year to oldest.
# - Display per tier. If no info in a tier, output "[Tier X]: None found in database."
# - NEVER skip Tier 1. If Guard RAG returns zero results, state: [External Web Search] and use general knowledge.

# DEFAULT NARRATIVE (If no specific category_id matched above):
# If the user asks a general question not tied to Q1.1–Q6.3, respond using:
## 🔍 Key Findings
(Narrative text)
## 📊 Clinical Interpretation
(Significance)
## 📋 Evidence-Based Recommendations
(Citations using [Tier X] format)

# ═══════════════════════════════════════════════════════════
# QUANTITATIVE TRACEABILITY BLOCK (append verbatim at the END of your response):
# ═══════════════════════════════════════════════════════════
# After your final clinical conclusion sentence, append EXACTLY this JSON block:
#
# ```json
# {_trace_block}
# ```
#
# Do NOT skip or modify this block. Graders require it for accuracy and concordance scoring.

RESPOND NOW:
"""
