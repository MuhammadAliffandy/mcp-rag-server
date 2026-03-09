"""
Embedded Clinical Knowledge Base for Guard RAG.

Contains structured clinical guidelines from major authorities (ACG, ECCO, WHO, etc.)
organized by condition and severity. This provides RELIABLE, INSTANT guideline answers
even without internet access, matching the expected PPT output format.

Architecture:
  - CLINICAL_GUIDELINES: Dict of condition → guideline entries
  - match_guideline(): Finds the best matching guideline for a clinical question
  - format_guideline_answer(): Formats a guideline match into the PPT-style citation
"""

from typing import List, Dict, Optional, Tuple
import re


# ============================================================================
# STRUCTURED CLINICAL GUIDELINES DATABASE
# ============================================================================
# Each entry maps clinical keywords → structured guideline recommendation.
# Source citations match real medical society publications.

CLINICAL_GUIDELINES: List[Dict] = [
    # ─── ULCERATIVE COLITIS (UC) ──────────────────────────────────────────────
    {
        "id": "UC_MES3_SEVERE",
        "keywords": ["mes 3", "severe colitis", "severe uc", "severe ulcerative colitis", "severe activity", "fulminant colitis"],
        "condition": "Severe Ulcerative Colitis (MES 3 / Mayo Endoscopic Score 3)",
        "source": "ACG Clinical Guidelines",
        "source_ref": "Rubin DT et al. ACG Clinical Guideline: Ulcerative Colitis in Adults. Am J Gastroenterol. 2019;114(3):384-413",
        "recommendation": (
            "Patients with Severe Colitis (MES 3) require **hospitalization** for **IV Corticosteroids** "
            "(e.g., methylprednisolone 60 mg/day or hydrocortisone 100 mg q8h). "
            "If no significant response by **Day 3**, consider **rescue therapy** with "
            "**Infliximab** (5 mg/kg) or **Cyclosporine** (2-4 mg/kg/day continuous IV). "
            "Surgical consultation should be obtained early. "
            "Failure to respond to rescue therapy within 5-7 days is an indication for **colectomy**."
        ),
        "severity": "severe",
        "specialty": "gi",
    },
    {
        "id": "UC_MES2_MODERATE",
        "keywords": ["mes 2", "moderate colitis", "moderate uc", "moderate ulcerative colitis", "moderate activity", "moderate-to-severe"],
        "condition": "Moderate Ulcerative Colitis (MES 2)",
        "source": "ACG Clinical Guidelines",
        "source_ref": "Rubin DT et al. ACG Clinical Guideline: Ulcerative Colitis in Adults. Am J Gastroenterol. 2019;114(3):384-413",
        "recommendation": (
            "For moderate UC (MES 2), first-line therapy is **oral 5-ASA** (mesalamine 2.4-4.8 g/day). "
            "If inadequate response, escalate to **oral corticosteroids** (prednisone 40-60 mg/day, tapering over 8-12 weeks). "
            "For steroid-dependent or steroid-refractory patients, consider **thiopurines** (azathioprine/6-MP) "
            "or **biologic therapy** (anti-TNF: Infliximab, Adalimumab; anti-integrin: Vedolizumab; "
            "or JAK inhibitor: Tofacitinib)."
        ),
        "severity": "moderate",
        "specialty": "gi",
    },
    {
        "id": "UC_MES1_MILD",
        "keywords": ["mes 1", "mild colitis", "mild uc", "mild ulcerative colitis", "mild activity"],
        "condition": "Mild Ulcerative Colitis (MES 1)",
        "source": "ACG Clinical Guidelines",
        "source_ref": "Rubin DT et al. ACG Clinical Guideline: Ulcerative Colitis in Adults. Am J Gastroenterol. 2019;114(3):384-413",
        "recommendation": (
            "Mild UC (MES 1) is primarily treated with **topical and/or oral 5-ASA** (mesalamine). "
            "For left-sided/distal disease: **mesalamine enemas** (4 g/day) or **suppositories** (1 g/day) "
            "combined with oral mesalamine (2.4-4.8 g/day) for optimal response. "
            "Combination topical + oral 5-ASA is superior to either alone."
        ),
        "severity": "mild",
        "specialty": "gi",
    },
    {
        "id": "UC_MES0_REMISSION",
        "keywords": ["mes 0", "remission", "maintenance", "mucosal healing", "inactive"],
        "condition": "Ulcerative Colitis in Remission (MES 0)",
        "source": "ACG Clinical Guidelines",
        "source_ref": "Rubin DT et al. ACG Clinical Guideline: Ulcerative Colitis in Adults. Am J Gastroenterol. 2019;114(3):384-413",
        "recommendation": (
            "Patients in remission (MES 0) should continue **maintenance therapy** with oral 5-ASA "
            "(mesalamine ≥2 g/day). For patients who achieved remission with biologics or immunomodulators, "
            "continue the same agent for maintenance. "
            "**Surveillance colonoscopy** should be performed 8 years after diagnosis onset, then every 1-3 years. "
            "Corticosteroids should NOT be used for maintenance."
        ),
        "severity": "remission",
        "specialty": "gi",
    },
    {
        "id": "UC_STEROID_REFRACTORY",
        "keywords": ["steroid refractory", "steroid dependent", "not responding to steroids", "steroid failure", "steroid resistant"],
        "condition": "Steroid-Refractory Ulcerative Colitis",
        "source": "ECCO Guidelines",
        "source_ref": "Harbord M et al. Third European Evidence-based Consensus on Diagnosis and Management of UC. J Crohns Colitis. 2017",
        "recommendation": (
            "According to **ECCO Guidelines**: Steroid-refractory UC requires escalation to **biologic therapy**. "
            "First-line biologics: **Infliximab** (5 mg/kg at weeks 0, 2, 6 then q8w) or **Vedolizumab** "
            "(300 mg at weeks 0, 2, 6 then q8w). "
            "Second-line options: **Adalimumab**, **Golimumab**, or **Tofacitinib** (JAK inhibitor). "
            "For acute severe steroid-refractory UC, IV **Infliximab** or **Cyclosporine** are rescue options. "
            "Early surgical consultation is recommended if rescue therapy fails within 5-7 days."
        ),
        "severity": "severe",
        "specialty": "gi",
    },

    # ─── CROHN'S DISEASE ──────────────────────────────────────────────────────
    {
        "id": "CD_MODERATE_SEVERE",
        "keywords": ["crohn", "crohn's disease", "moderate crohn", "severe crohn", "cd treatment"],
        "condition": "Moderate-to-Severe Crohn's Disease",
        "source": "ACG Clinical Guidelines",
        "source_ref": "Lichtenstein GR et al. ACG Clinical Guideline: Management of Crohn's Disease. Am J Gastroenterol. 2018;113(4):481-517",
        "recommendation": (
            "For moderate-to-severe Crohn's Disease, **biologic therapy** is recommended as first-line: "
            "**Anti-TNF agents** (Infliximab, Adalimumab, Certolizumab) or **Vedolizumab** or **Ustekinumab**. "
            "For induction: corticosteroids (budesonide 9 mg/day or prednisone 40-60 mg). "
            "Combination therapy (anti-TNF + immunomodulator) is superior to monotherapy. "
            "**Thiopurines** (azathioprine, 6-MP) or **methotrexate** for steroid-sparing maintenance."
        ),
        "severity": "moderate-severe",
        "specialty": "gi",
    },

    # ─── IBD GENERAL ──────────────────────────────────────────────────────────
    {
        "id": "IBD_BIOLOGICS_ESCALATION",
        "keywords": ["biologic escalation", "biologic therapy", "anti-tnf", "vedolizumab", "ustekinumab", "tofacitinib", "ibd escalation"],
        "condition": "IBD Biologic Therapy Escalation",
        "source": "ECCO Guidelines & ACG Guidelines",
        "source_ref": "ECCO Guidelines 2023; ACG Guidelines 2019",
        "recommendation": (
            "**Biologic Escalation Pathway for IBD:**\n"
            "1. **First-line biologics**: Anti-TNF (Infliximab/Adalimumab) ± immunomodulator\n"
            "2. **Second-line (anti-TNF failure)**: Vedolizumab (gut-selective) or Ustekinumab (anti-IL-12/23)\n"
            "3. **Third-line**: JAK inhibitors (Tofacitinib for UC; Upadacitinib for UC/CD)\n"
            "4. **Consider combination therapy**: biologic + thiopurine/methotrexate for higher efficacy\n"
            "5. **Therapeutic drug monitoring (TDM)**: Measure trough levels and anti-drug antibodies "
            "before switching within class vs. out of class."
        ),
        "severity": "all",
        "specialty": "gi",
    },
    {
        "id": "IBD_COLONOSCOPY_SURVEILLANCE",
        "keywords": ["colonoscopy surveillance", "screening colonoscopy", "ibd surveillance", "cancer screening ibd"],
        "condition": "IBD Colonoscopy Surveillance",
        "source": "ACG / ECCO Guidelines",
        "source_ref": "ACG 2019; ECCO 2017",
        "recommendation": (
            "IBD patients should begin **surveillance colonoscopy 8 years** after symptom onset. "
            "Frequency: every **1-3 years** depending on risk factors. "
            "**High-risk factors** requiring annual surveillance: primary sclerosing cholangitis (PSC), "
            "extensive colitis, family history of CRC, history of dysplasia, or stricture. "
            "**Chromoendoscopy** with targeted biopsies is preferred over random biopsies."
        ),
        "severity": "all",
        "specialty": "gi",
    },

    # ─── MAYO SCORE ───────────────────────────────────────────────────────────
    {
        "id": "MAYO_SCORE_INTERPRETATION",
        "keywords": ["mayo score", "pmayo", "partial mayo", "mayo scoring", "mayo interpretation", "mes score"],
        "condition": "Mayo Endoscopic Score (MES) Interpretation",
        "source": "ACG Clinical Guidelines",
        "source_ref": "Schroeder KW et al. N Engl J Med. 1987;317:1625-1629; ACG 2019",
        "recommendation": (
            "**Mayo Endoscopic Score (MES) Classification:**\n"
            "- **MES 0 (Normal/Remission)**: Normal mucosa or inactive disease\n"
            "- **MES 1 (Mild)**: Erythema, decreased vascular pattern, mild friability\n"
            "- **MES 2 (Moderate)**: Marked erythema, absent vascular pattern, friability, erosions\n"
            "- **MES 3 (Severe)**: Spontaneous bleeding, ulceration\n\n"
            "**Partial Mayo Score (pMayo)**: 0-1 (remission), 2-4 (mild), 5-7 (moderate), 8-9 (severe).\n"
            "**Target for treatment**: MES 0 (mucosal healing) is the gold standard therapeutic target."
        ),
        "severity": "all",
        "specialty": "gi",
    },

    # ─── GENERAL MEDICAL ──────────────────────────────────────────────────────
    {
        "id": "DIABETES_HBA1C_HIGH",
        "keywords": ["hba1c", "diabetes", "glucose management", "hyperglycemia", "diabetes management", "dm type 2"],
        "condition": "Type 2 Diabetes — Glycemic Management",
        "source": "ADA Standards of Care",
        "source_ref": "American Diabetes Association. Standards of Care in Diabetes—2024. Diabetes Care. 2024;47(Suppl 1)",
        "recommendation": (
            "Per **ADA Standards of Care**: Target HbA1c <7.0% for most adults. "
            "**First-line**: Metformin + lifestyle modifications. "
            "**If HbA1c >9%**: Consider dual therapy (Metformin + GLP-1 RA or SGLT2i) or initiate insulin. "
            "**If HbA1c >10% with symptoms**: Start **basal insulin** immediately. "
            "**CV risk**: Prefer GLP-1 RA (semaglutide/liraglutide) or SGLT2i (empagliflozin/dapagliflozin) "
            "for patients with established ASCVD, HF, or CKD."
        ),
        "severity": "all",
        "specialty": "diabetes",
    },
    {
        "id": "SEPSIS_MANAGEMENT",
        "keywords": ["sepsis", "septic shock", "sepsis management", "sepsis protocol", "surviving sepsis"],
        "condition": "Sepsis / Septic Shock Management",
        "source": "Surviving Sepsis Campaign (SCCM/ESICM)",
        "source_ref": "Evans L et al. Surviving Sepsis Campaign: International Guidelines. Crit Care Med. 2021;49(11):e1063-e1143",
        "recommendation": (
            "Per **Surviving Sepsis Campaign Guidelines**:\n"
            "1. **Hour-1 Bundle**: Measure lactate, obtain blood cultures, administer broad-spectrum antibiotics, "
            "begin rapid IV crystalloid (30 mL/kg for hypotension or lactate ≥4 mmol/L), apply vasopressors if MAP <65 mmHg\n"
            "2. **Vasopressor of choice**: **Norepinephrine** (first-line), add **vasopressin** if needed\n"
            "3. **Source control**: Within 6-12 hours (drain abscesses, remove infected devices)\n"
            "4. **De-escalate antibiotics** based on culture results (aim for narrowest spectrum)\n"
            "5. **Corticosteroids**: IV hydrocortisone 200 mg/day ONLY for refractory septic shock"
        ),
        "severity": "severe",
        "specialty": "icu",
    },
    {
        "id": "HEART_FAILURE_MANAGEMENT",
        "keywords": ["heart failure", "hf management", "ejection fraction", "ef reduced", "hfref", "lvef"],
        "condition": "Heart Failure with Reduced Ejection Fraction (HFrEF)",
        "source": "ESC Guidelines",
        "source_ref": "McDonagh TA et al. 2021 ESC Guidelines for the diagnosis and treatment of acute and chronic heart failure. Eur Heart J. 2021",
        "recommendation": (
            "Per **ESC Guidelines** for HFrEF (LVEF ≤40%):\n"
            "**Foundational therapy (all 4 pillars):**\n"
            "1. **ACEi/ARB or ARNI** (sacubitril/valsartan preferred)\n"
            "2. **Beta-blocker** (bisoprolol, carvedilol, or metoprolol succinate)\n"
            "3. **MRA** (spironolactone or eplerenone)\n"
            "4. **SGLT2i** (dapagliflozin or empagliflozin)\n\n"
            "**Additional**: Loop diuretics for congestion, ICD if LVEF ≤35% despite 3 months OMT, "
            "CRT if LBBB + QRS ≥150ms. **Iron replacement** if ferritin <100 or TSAT <20%."
        ),
        "severity": "all",
        "specialty": "cardiology",
    },
    {
        "id": "COPD_GOLD_MANAGEMENT",
        "keywords": ["copd", "gold", "chronic obstructive", "fev1", "copd exacerbation", "copd management"],
        "condition": "COPD Management",
        "source": "GOLD Guidelines",
        "source_ref": "Global Initiative for Chronic Obstructive Lung Disease (GOLD). 2024 Report.",
        "recommendation": (
            "Per **GOLD 2024 Guidelines**:\n"
            "**Group A** (low symptoms, low risk): SABA or SAMA as needed\n"
            "**Group B** (more symptoms): LABA or LAMA monotherapy\n"
            "**Group E** (exacerbation history): LABA + LAMA; if eos ≥300: LABA + LAMA + ICS\n\n"
            "**Acute exacerbation**: Short-course systemic corticosteroids (prednisone 40 mg × 5 days), "
            "antibiotics if purulent sputum, supplemental O2 to target SpO2 88-92%. "
            "**NIV** for acute hypercapnic respiratory failure."
        ),
        "severity": "all",
        "specialty": "respiratory",
    },
]


# ============================================================================
# GUIDELINE MATCHING ENGINE
# ============================================================================

def match_guideline(question: str, patient_context: str = "") -> List[Dict]:
    """
    Finds the best matching clinical guidelines for a question.
    Returns a list of matching guidelines, sorted by relevance score.
    """
    q_combined = f"{question} {patient_context}".lower()
    matches = []

    for guideline in CLINICAL_GUIDELINES:
        score = 0
        for keyword in guideline["keywords"]:
            if keyword in q_combined:
                # Longer keyword matches = higher confidence
                score += len(keyword.split())

        if score > 0:
            matches.append({**guideline, "_score": score})

    # Sort by score descending
    matches.sort(key=lambda x: -x["_score"])
    return matches


def format_guideline_answer(matches: List[Dict], question: str) -> str:
    """
    Formats matched guidelines into the PPT-style clinical answer.
    
    PPT expected format:
    "According to ACG Clinical Guidelines: Patients with Severe Colitis (MES 3)
     require hospitalization for IV Corticosteroids..."
    
    Always includes a 📚 Sources Consulted section at the bottom.
    """
    if not matches:
        return ""

    parts = []
    sources_list = []

    for i, m in enumerate(matches[:3]):  # Max 3 guidelines
        source = m["source"]
        condition = m["condition"]
        recommendation = m["recommendation"]
        source_ref = m.get("source_ref", "")

        if i == 0:
            # Primary match — PPT citation style
            block = (
                f"According to **{source}**: {recommendation}\n\n"
                f"*Reference: {source_ref}*"
            )
        else:
            # Secondary matches — additional context
            block = (
                f"\n\n---\n\n"
                f"**Additional guidance from {source}** ({condition}):\n"
                f"{recommendation}\n\n"
                f"*Reference: {source_ref}*"
            )
        parts.append(block)
        sources_list.append(f"- **{source}**: {source_ref}")

    # Always add Sources Consulted section
    answer = "".join(parts)
    sources_section = "\n".join(sources_list)
    answer += f"\n\n📚 **Sources Consulted:**\n{sources_section}"

    return answer
