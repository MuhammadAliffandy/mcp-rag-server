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
Based on the retrieved data and guidelines provided, I would classify the disease severity of Patient [ID] as "[Severity]" using the validated scoring systems."""

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
  (Nancy max=[X], remission if 0 or 1)"""

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
FINAL ANSWER (TRIAL FORMAT): Yes, [list poor factors]. OR No."""

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
FINAL ANSWER (TRIAL FORMAT): The patient has achieved [short term / intermediate / long term / no formal] treatment target."""

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
FINAL ANSWER (TRIAL FORMAT): Yes, according to treat-to-target strategy, the current medication should be adjusted. OR No."""

    elif category_id == "Q2.3":
        category_force = """FORCE ACTION: Q2.3 — NEXT TREATMENT OPTIONS.

Based on the patient's demographics, disease extent, severity, and current medication, recommend the next options.

RULES:
- If patient is on 5-ASA and not in remission → escalate
- If patient is on first biologic and failed → switch or combine
- Output MUST use ONLY these options: Optimize current medication / Add on immunomodulators / Escalate to advanced therapy / Switch to or combine other advanced therapy

### 📝 Final Clinical Conclusion
FINAL ANSWER (TRIAL FORMAT): [Optimize current medication / Add on immunomodulators / Escalate to advanced therapy / Switch to or combine other advanced therapy]"""

    # ─────────────────────────────────────────────────────────────
    # CATEGORY 3: Cancer Surveillance
    # ─────────────────────────────────────────────────────────────

    elif category_id == "Q3.1":
        category_force = """FORCE ACTION: Q3.1 — COLORECTAL CANCER SURVEILLANCE.

Determine the CRC risk group and next colonoscopy interval based on:
- High risk: PSC, prior dysplasia, extent=3 with >20 years disease, or family history CRC
- Intermediate risk: Extent=3 with 8–20 years disease, and/or active inflammation
- Low risk: Extent=1–2, quiescent disease, no risk factors

Rules:
- High risk → every 1 year
- Intermediate risk → every 2–3 years
- Low risk → every 5 years

### 📝 Final Clinical Conclusion
FINAL ANSWER (TRIAL FORMAT): Since the patient belongs to [low / intermediate / high] risk group, the next surveillance colonoscopy should be in [X] years."""

    elif category_id == "Q3.2":
        category_force = """FORCE ACTION: Q3.2 — OTHER CANCER SCREENING.

Based on patient's sex, age, underlying disease, and medication history, determine applicable cancer screens.

Rules:
- Female on immunosuppressants → cervical cancer: Pap smear every 1 year
- Immunosuppressed patients → skin cancer: total body skin exam every 1 year
- Thiopurine use → lymphoma risk: annual CBC
- Male >50 years → prostate cancer: PSA every 1–2 years
- All patients → colorectal already covered in Q3.1

### 📝 Final Clinical Conclusion
FINAL ANSWER (TRIAL FORMAT): Based on the patient's sex, age, underlying disease, and medication history, the patient should receive screening for [cancer type] with [screening method], every [X] year(s)."""

    # ─────────────────────────────────────────────────────────────
    # CATEGORY 4: Monitor Tools and Interval
    # ─────────────────────────────────────────────────────────────

    elif category_id == "Q4.1":
        category_force = """FORCE ACTION: Q4.1 — NON-INVASIVE MONITORING SCHEDULE.

Determine appropriate non-invasive disease activity monitoring based on the patient's current status.

Rules:
- Active disease or post-adjustment → fecal calprotectin + CRP at 3 months
- Remission and stable → fecal calprotectin + CRP at 6 months
- If on biologic → also add TDM (handled in Q4.2 separately)

### 📝 Final Clinical Conclusion
FINAL ANSWER (TRIAL FORMAT): Based on the patient's current status, the following exams [fecal calprotectin / CRP / others] should be arranged at [3 months / 6 months / 3-6 months]."""

    elif category_id == "Q4.2":
        category_force = """FORCE ACTION: Q4.2 — THERAPEUTIC DRUG MONITORING (TDM).

Determine TDM necessity and type based on the patient's current medication:

Rules:
- Proactive TDM: Patient in remission, to optimise drug levels proactively
- Reactive TDM: Patient not in remission or flaring, to investigate failure
- If on Infliximab → target trough level > 5 μg/mL
- If on Adalimumab → target trough level > 7.5 μg/mL
- If on Vedolizumab → target trough level > 18–20 μg/mL
- If NOT on biologic or small molecule → "No TDM indicated."

### 📝 Final Clinical Conclusion
FINAL ANSWER (TRIAL FORMAT): Yes, [proactive / reactive] TDM is recommended, with target drug level [X]. OR No."""

    elif category_id == "Q4.3":
        category_force = """FORCE ACTION: Q4.3 — MEDICATION-SPECIFIC MONITORING.

Determine medication-specific safety monitoring requirements from the patient's medication history.

Rules:
- Thiopurines (azathioprine/6-MP) → CBC + liver enzymes every 3 months
- Methotrexate → CBC + liver enzymes every 1–3 months
- Biologics (infliximab/adalimumab) → no specific lab monitoring required
- JAK inhibitors (tofacitinib) → CBC + lipid panel every 3 months
- Steroids → blood glucose, bone density screening
- 5-ASA → renal function annually

### 📝 Final Clinical Conclusion
FINAL ANSWER (TRIAL FORMAT): For patients under [medication name] medication, [specific lab tests] should be checked every [X] months. OR No specific monitoring required."""

    elif category_id == "Q4.4":
        category_force = """FORCE ACTION: Q4.4 — OPPORTUNISTIC INFECTIONS AND VACCINATIONS.

Determine infection risk screening and required vaccinations based on immunosuppression status.

Rules:
- Before any immunosuppressive therapy → Hepatitis B screen + vaccination if negative
- Annual influenza vaccine for all immunosuppressed patients
- Pneumococcal vaccine (PCV13 + PPSV23) for patients on biologics/thiopurines
- Varicella IgG check before immunosuppression; VZV vaccine only if NOT immunosuppressed
- TB screening (IGRA/Mantoux) before anti-TNF therapy
- HPV vaccine for patients <26 years (or <45 per ACIP)
- COVID-19 vaccine recommended for all IBD patients

### 📝 Final Clinical Conclusion
FINAL ANSWER (TRIAL FORMAT): Screening for [infection(s)] and [vaccine(s)] vaccinations prior to treatment initiation are recommended."""

    # ─────────────────────────────────────────────────────────────
    # CATEGORY 5: Lifestyle and Diet Modification
    # ─────────────────────────────────────────────────────────────

    elif category_id == "Q5.1":
        category_force = """FORCE ACTION: Q5.1 — DIETARY RECOMMENDATION.

Provide a concise dietary recommendation for the patient based on their disease status.

Rules:
- Active UC → avoid high-fiber, spicy, caffeinated, and alcohol-containing foods
- Remission → mediterranean diet, high-fiber, plant-based foods encouraged
- ALL UC patients → avoid processed foods, red meat; encourage dietary fiber
- Flare → low-residue diet may be temporarily recommended

### 📝 Final Clinical Conclusion
FINAL ANSWER (TRIAL FORMAT — output ONLY this sentence): This patient is encouraged to have more [dietary recommendation] intake and less [foods to avoid]."""

    elif category_id == "Q5.2":
        category_force = """FORCE ACTION: Q5.2 — NUTRITIONAL SUPPLEMENTATION AND DEFICIENCY SCREENING.

Determine if the patient requires nutritional supplementation or deficiency screening.

Rules:
- ALL UC patients → screen for Vitamin D deficiency (IBD linked to low Vit D)
- If on thiopurines or methotrexate → folate supplementation recommended
- If extensive colitis → iron deficiency screening (CBC/ferritin)
- If steroid use → calcium + Vitamin D supplementation
- If malabsorption signs → B12, zinc screening
- If pregnant → folate supplementation mandatory

### 📝 Final Clinical Conclusion
FINAL ANSWER (TRIAL FORMAT): Yes, the patient is recommended to be screened for [deficiency] deficiency. OR No."""

    elif category_id == "Q5.3":
        category_force = """FORCE ACTION: Q5.3 — LIFESTYLE MODIFICATIONS.

Provide concise lifestyle modification advice for the patient.

Rules:
- Smoking → smoking worsens CD but may paradoxically reduce UC; however advise cessation for overall health
- Physical activity → moderate exercise (150 min/week) reduces inflammation markers
- Stress management → IBD linked to psychosocial stress; CBT, mindfulness recommended
- BMI → obesity management important for biologic efficacy
- Alcohol → limit or avoid; worsens IBD inflammation

### 📝 Final Clinical Conclusion
FINAL ANSWER (TRIAL FORMAT — output ONLY this sentence): The patient should quit [smoking/alcohol/other] and enhance [physical activity/stress management/other]."""

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
FINAL ANSWER (TRIAL FORMAT): These [medication name(s)] medications were safe to be continued. These [medication name(s)] medication should be stopped [X] months before conception."""

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
FINAL ANSWER (TRIAL FORMAT): Maternally, the risk of [complication(s)] is increased / comparable to the non-IBD patients."""

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
FINAL ANSWER (TRIAL FORMAT): Neonatally, the risk of [complication(s)] is increased / comparable to the mothers of non-IBD patients."""

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

RESPOND NOW:
"""
