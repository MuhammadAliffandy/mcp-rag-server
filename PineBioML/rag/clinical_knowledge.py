"""
Embedded Clinical Knowledge Base for Guard RAG — ColonoSense v2.

Contains structured clinical guidelines from major authorities (ACG, ECCO, WHO, etc.)
organized by condition and severity. This provides RELIABLE, INSTANT guideline answers
even without internet access, matching the ColonoSense tiered citation format.

Tier Hierarchy (STRICT):
  - Tier 1: Global Guidelines (ACG, ECCO, AGA, WGO)
  - Tier 2: Local Guidelines (country/hospital-specific)
  - Tier 3: Meta-analyses (systematic reviews, Cochrane)
  - Tier 4: Pivotal Trials (landmark RCTs)

Architecture:
  - CLINICAL_GUIDELINES: Dict of condition → guideline entries with tier + year
  - match_guideline(): Finds the best matching guideline for a clinical question
  - format_guideline_answer(): Formats with [Tier X] prefix and year-descending sort
"""

from typing import List, Dict, Optional, Tuple
import re


# ============================================================================
# STRUCTURED CLINICAL GUIDELINES DATABASE
# ============================================================================
# Each entry includes tier (1-4), year, and structured recommendation.

CLINICAL_GUIDELINES: List[Dict] = [
    # ─── Q1.1: DISEASE SEVERITY ASSESSMENT ──────────────────────────────────
    {
        "id": "UC_DISEASE_SEVERITY_ASSESSMENT",
        "keywords": ["disease severity", "disease severity status", "severity classification", "severity for patient"],
        "condition": "Ulcerative Colitis Disease Severity Classification",
        "source": "ACG / ECCO Clinical Guidelines",
        "source_ref": "Rubin DT et al. ACG Clinical Guideline: Ulcerative Colitis in Adults. Am J Gastroenterol. 2019;114(3):384-413",
        "tier": 1,
        "year": 2020,
        "recommendation": (
            "Ulcerative colitis disease severity is categorized by Total Mayo Score (partial Mayo score + endoscopic score MES): "
            "Total Mayo <= 2 indicates Clinical Remission; Total Mayo 3-5 indicates Mild disease severity; "
            "Total Mayo 6-10 indicates Moderate disease severity; Total Mayo > 10 indicates Severe disease severity. "
            "[Tier 1] Clinical disease severity guides choice of induction and maintenance therapies. [ACG, 2020]"
        ),
        "severity": "all",
        "specialty": "gi",
    },

    # ─── Q1.3: PROGNOSTIC FACTOR ASSESSMENT ──────────────────────────────────
    {
        "id": "UC_PROGNOSTIC_FACTOR_ASSESSMENT",
        "keywords": ["prognostic factors", "prognostic factor", "poor prognostic factors", "poor prognosis"],
        "condition": "Prognostic Factor Assessment in Ulcerative Colitis",
        "source": "ECCO Guidelines on Therapeutics in UC",
        "source_ref": "Raine T et al. ECCO Guidelines on Therapeutics in UC. J Crohns Colitis. 2022;16(1):2-17",
        "tier": 1,
        "year": 2022,
        "recommendation": (
            "Poor prognostic factors in ulcerative colitis include: "
            "age at diagnosis < 40 years, extensive colitis (Montreal E3), severe endoscopic disease (MES >= 3 or deep ulcerations), "
            "elevated inflammatory markers (CRP > 1.0 mg/dL, fecal calprotectin > 100 ug/g), low serum albumin (< 3.5 g/dL), and early requirement for systemic corticosteroids. "
            "[Tier 1] Identification of poor prognostic factors warrants early optimization or escalation to advanced biologic therapies. [ECCO, 2022]"
        ),
        "severity": "all",
        "specialty": "gi",
    },

    # ─── Q2.2: MEDICATION ADJUSTMENT ──────────────────────────────────────────
    {
        "id": "UC_MEDICATION_ADJUSTMENT_STRATEGY",
        "keywords": ["medication be adjusted", "medication adjusted", "adjust medication", "adjust the medication"],
        "condition": "Treat-to-Target Medication Adjustment Strategy",
        "source": "STRIDE-II Consensus",
        "source_ref": "Turner D et al. STRIDE-II: An Update on Selecting Therapeutic Targets in IBD. Gastroenterology. 2021;160(5):1570-1583",
        "tier": 1,
        "year": 2021,
        "recommendation": (
            "According to treat-to-target strategy (STRIDE-II): "
            "If the patient has achieved treatment targets (clinical and biochemical remission), no medication adjustment is required, and current maintenance therapy should be maintained. "
            "If targets are not achieved or flare occurs, adjust therapy via dose optimization, adding topical agents, or escalating to biologic or targeted small molecules. "
            "[Tier 1] Treat-to-target strategy dictates adjusting therapy only when objective inflammatory targets are unmet. [STRIDE-II, 2021]"
        ),
        "severity": "all",
        "specialty": "gi",
    },

    # ─── Q2.3: NEXT TREATMENT OPTION ──────────────────────────────────────────
    {
        "id": "UC_NEXT_TREATMENT_OPTION_ALGORITHM",
        "keywords": ["recommended next treatment option", "next treatment option", "next treatment", "recommended next treatment"],
        "condition": "Next Treatment Option in Ulcerative Colitis",
        "source": "ACG / ECCO Guidelines",
        "source_ref": "Rubin DT et al. ACG Clinical Guideline: Ulcerative Colitis in Adults. Am J Gastroenterol. 2019;114(3):384-413",
        "tier": 1,
        "year": 2020,
        "recommendation": (
            "The recommended next treatment option depends on clinical response to current therapy: "
            "For patients in clinical and endoscopic remission, continue current maintenance therapy. "
            "For patients failing 5-ASA, optimize oral 5-ASA to 4.8 g/day or add topical rectal mesalamine. "
            "For steroid-dependent or refractory disease, escalate to biologic therapy (anti-TNF: Infliximab/Adalimumab, anti-integrin: Vedolizumab, or anti-IL23: Ustekinumab) or JAK inhibitors. "
            "[Tier 1] Escalation of therapy should follow failure of optimized first-line agents. [ACG, 2020]"
        ),
        "severity": "all",
        "specialty": "gi",
    },

    # ─── Q3.2: OTHER CANCER SCREENINGS ────────────────────────────────────────
    {
        "id": "IBD_OTHER_CANCER_SCREENINGS",
        "keywords": ["other cancer screenings", "other cancer screening", "other cancer", "other cancers"],
        "condition": "Non-CRC Cancer Screening in IBD",
        "source": "ECCO Guidelines",
        "source_ref": "Annese V et al. European Evidence-based Consensus on Opportunistic Infections and Malignancies in IBD. J Crohns Colitis. 2017",
        "tier": 1,
        "year": 2021,
        "recommendation": (
            "Patients with IBD should receive non-colorectal cancer screening including: "
            "1. Cervical cancer: Annual Pap smear screening is recommended for women with IBD receiving immunosuppressive therapy. "
            "2. Skin cancer: Annual dermatological screening for melanoma and non-melanoma skin cancer, especially for patients on thiopurines or anti-TNF agents. "
            "3. Cholangiocarcinoma: Surveillance with annual ultrasound / MRI and CA 19-9 for patients with Primary Sclerosing Cholangitis (PSC). "
            "[Tier 1] Immunosuppressed IBD patients require targeted screening for cervical and cutaneous malignancies. [ECCO, 2021]"
        ),
        "severity": "all",
        "specialty": "gi",
    },

    # ─── ULCERATIVE COLITIS (UC) ──────────────────────────────────────────────
    {
        "id": "UC_MES3_SEVERE",
        "keywords": ["mes 3", "severe colitis", "severe uc", "severe ulcerative colitis", "severe activity", "fulminant colitis"],
        "condition": "Severe Ulcerative Colitis (MES 3 / Mayo Endoscopic Score 3)",
        "source": "ACG Clinical Guidelines",
        "source_ref": "Rubin DT et al. ACG Clinical Guideline: Ulcerative Colitis in Adults. Am J Gastroenterol. 2019;114(3):384-413",
        "tier": 1,
        "year": 2019,
        "recommendation": (
            "Patients with Severe Colitis (MES 3, Total Mayo >10) require **hospitalization** for **IV Corticosteroids** "
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
        "condition": "Moderate Ulcerative Colitis (MES 2, Total Mayo 6-10)",
        "source": "ACG Clinical Guidelines",
        "source_ref": "Rubin DT et al. ACG Clinical Guideline: Ulcerative Colitis in Adults. Am J Gastroenterol. 2019;114(3):384-413",
        "tier": 1,
        "year": 2019,
        "recommendation": (
            "For moderate UC (MES 2, Total Mayo 6-10), first-line therapy is **oral 5-ASA** (mesalamine 2.4-4.8 g/day). "
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
        "condition": "Mild Ulcerative Colitis (MES 1, Total Mayo 3-5)",
        "source": "ACG Clinical Guidelines",
        "source_ref": "Rubin DT et al. ACG Clinical Guideline: Ulcerative Colitis in Adults. Am J Gastroenterol. 2019;114(3):384-413",
        "tier": 1,
        "year": 2019,
        "recommendation": (
            "Mild UC (MES 1, Total Mayo 3-5) is primarily treated with **topical and/or oral 5-ASA** (mesalamine). "
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
        "condition": "Ulcerative Colitis in Remission (MES 0, Total Mayo 0-2)",
        "source": "ACG Clinical Guidelines",
        "source_ref": "Rubin DT et al. ACG Clinical Guideline: Ulcerative Colitis in Adults. Am J Gastroenterol. 2019;114(3):384-413",
        "tier": 1,
        "year": 2019,
        "recommendation": (
            "Patients in remission (MES 0, Total Mayo 0-2) should continue **maintenance therapy** with oral 5-ASA "
            "(mesalamine ≥2 g/day). For patients who achieved remission with biologics or immunomodulators, "
            "continue the same agent for maintenance. "
            "**Remission Checklist:** Clinical (pMayo <3, no sub-score >1), Biochemical (CRP <1 & FC <100), "
            "Endoscopic (MES 0 or 1), Histologic (Nancy 0 or 1). "
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
        "source_ref": "Raine T et al. ECCO Guidelines on Therapeutics in UC. J Crohns Colitis. 2022;16(1):2-17",
        "tier": 1,
        "year": 2022,
        "recommendation": (
            "According to **ECCO Guidelines**: Steroid-refractory UC (>12 weeks steroid use) requires escalation to **biologic therapy**. "
            "First-line biologics: **Infliximab** (5 mg/kg at weeks 0, 2, 6 then q8w) or **Vedolizumab** "
            "(300 mg at weeks 0, 2, 6 then q8w). "
            "Second-line options: **Adalimumab**, **Golimumab**, or **Tofacitinib** (JAK inhibitor). "
            "For acute severe steroid-refractory UC, IV **Infliximab** or **Cyclosporine** are rescue options. "
            "Early surgical consultation is recommended if rescue therapy fails within 5-7 days."
        ),
        "severity": "severe",
        "specialty": "gi",
    },

    # ─── 5-ASA OPTIMIZATION ──────────────────────────────────────────────────
    {
        "id": "UC_5ASA_OPTIMIZATION",
        "keywords": ["5-asa", "mesalamine", "5asa optimization", "aminosalicylate", "optimasi 5-asa", "mesalazine"],
        "condition": "5-ASA Optimization in Ulcerative Colitis",
        "source": "ACG Clinical Guidelines",
        "source_ref": "Rubin DT et al. ACG Clinical Guideline: Ulcerative Colitis in Adults. Am J Gastroenterol. 2019;114(3):384-413",
        "tier": 1,
        "year": 2019,
        "recommendation": (
            "Before escalating to advanced therapy, ensure **5-ASA is optimized to 4.8 g/d**. "
            "For left-sided colitis or proctitis, add **rectal 5-ASA** (mesalamine enemas 4 g/day or "
            "suppositories 1 g/day). Combination oral + topical 5-ASA is superior to either alone. "
            "For extensive mild-to-moderate UC, use **standard-dose mesalamine (2-3 g/d)** rather than low-dose. "
            "Escalation to advanced therapy should only occur after optimized 5-ASA ± immunomodulator failure."
        ),
        "severity": "mild",
        "specialty": "gi",
    },

    # ─── RESPONSE TIMELINE ──────────────────────────────────────────────────
    {
        "id": "UC_RESPONSE_TIMELINE",
        "keywords": ["response timeline", "treatment response", "when to expect response", "time to remission", "medication adequacy", "waktu respons"],
        "condition": "UC Treatment Response Timeline",
        "source": "ECCO / ACG Consensus",
        "source_ref": "ECCO Guidelines 2022; ACG Guidelines 2019",
        "tier": 1,
        "year": 2022,
        "recommendation": (
            "**Treatment Response Timeline for UC (expected clinical remission):**\n"
            "| Medication | Expected Remission | Assessment Point |\n"
            "|---|---|---|\n"
            "| Infliximab | 10 weeks | Week 14 |\n"
            "| Adalimumab | 11 weeks | Week 12 |\n"
            "| Vedolizumab | 14 weeks | Week 14 |\n"
            "| Tofacitinib | 8 weeks | Week 8 |\n"
            "| Upadacitinib | 8 weeks | Week 8 |\n"
            "| Ustekinumab | 12 weeks | Week 16 |\n\n"
            "If no response by the assessment point, consider optimizing dose, checking therapeutic drug levels, "
            "or switching mechanism of action."
        ),
        "severity": "all",
        "specialty": "gi",
    },

    # ─── CRC SURVEILLANCE ──────────────────────────────────────────────────
    {
        "id": "UC_CRC_SURVEILLANCE",
        "keywords": ["colorectal cancer screening plan", "colorectal cancer screening", "colorectal cancer", "crc screening", "colon cancer", "cancer surveillance", "colonoscopy surveillance", "screening colonoscopy", "cancer prevention"],
        "condition": "CRC Surveillance in UC",
        "source": "ACG / ECCO Guidelines",
        "source_ref": "ACG 2019; ECCO 2017",
        "tier": 1,
        "year": 2019,
        "recommendation": (
            "**CRC Screening:** Start 8 years after symptom onset.\n\n"
            "**Surveillance Intervals:**\n"
            "| Risk Category | Interval | Criteria |\n"
            "|---|---|---|\n"
            "| High Risk | 1 year | Severe inflammation, PSC (start immediately), CRC family history |\n"
            "| Intermediate | 2-3 years | Mild-moderate inflammation or CRC family history |\n"
            "| Low Risk | 5 years | Left-sided colitis or minimal inflammation |\n\n"
            "**Malignancy Awareness:**\n"
            "- Skin cancer: yearly dermatological exam (especially if on thiopurines)\n"
            "- Cervical cancer: Pap smear per protocol\n"
            "- Cholangiocarcinoma: CA19-9 monitoring for PSC patients\n\n"
            "**Chromoendoscopy** with targeted biopsies is preferred over random biopsies."
        ),
        "severity": "all",
        "specialty": "gi",
    },

    # ─── TREAT-TO-TARGET ──────────────────────────────────────────────────
    {
        "id": "UC_TREAT_TO_TARGET",
        "keywords": ["treat to target", "treatment target", "target terapi", "target pengobatan", "treat-to-target"],
        "condition": "Treat-to-Target Strategy in UC",
        "source": "STRIDE-II Consensus",
        "source_ref": "Turner D et al. STRIDE-II: An Update on the Selecting Therapeutic Targets in IBD. Gastroenterology. 2021;160(5):1570-1583",
        "tier": 1,
        "year": 2021,
        "recommendation": (
            "**STRIDE-II Treat-to-Target for UC:**\n"
            "1. **Short-term target:** Clinical remission (Partial Mayo <3, no sub-score >1)\n"
            "2. **Intermediate target:** Biochemical remission (CRP normalization + FC <100-250 µg/g)\n"
            "3. **Long-term target:** Endoscopic remission (MES 0 or 1)\n"
            "4. **Aspirational target:** Histologic remission (Nancy 0 or 1)\n\n"
            "Reassess at regular intervals and adjust therapy if targets not met. "
            "Treatment escalation should follow this sequential target approach."
        ),
        "severity": "all",
        "specialty": "gi",
    },

    # ─── CROHN'S DISEASE ──────────────────────────────────────────────────────
    {
        "id": "CD_MODERATE_SEVERE",
        "keywords": ["crohn", "crohn's disease", "moderate crohn", "severe crohn", "cd treatment"],
        "condition": "Moderate-to-Severe Crohn's Disease",
        "source": "ACG Clinical Guidelines",
        "source_ref": "Lichtenstein GR et al. ACG Clinical Guideline: Management of Crohn's Disease. Am J Gastroenterol. 2018;113(4):481-517",
        "tier": 1,
        "year": 2018,
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
        "source_ref": "ECCO Guidelines 2022; ACG Guidelines 2019",
        "tier": 1,
        "year": 2022,
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
        "keywords": ["colonoscopy surveillance", "ibd surveillance", "cancer screening ibd"],
        "condition": "IBD Colonoscopy Surveillance",
        "source": "ACG / ECCO Guidelines",
        "source_ref": "ACG 2019; ECCO 2017",
        "tier": 1,
        "year": 2019,
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
        "tier": 1,
        "year": 2019,
        "recommendation": (
            "**Mayo Endoscopic Score (MES) Classification:**\n"
            "- **MES 0 (Normal/Remission)**: Normal mucosa or inactive disease\n"
            "- **MES 1 (Mild)**: Erythema, decreased vascular pattern, mild friability\n"
            "- **MES 2 (Moderate)**: Marked erythema, absent vascular pattern, friability, erosions\n"
            "- **MES 3 (Severe)**: Spontaneous bleeding, ulceration\n\n"
            "**Total Mayo Score Classification (Partial Mayo + MES, range 0-12):**\n"
            "- **Remission: 0-2** | **Mild: 3-5** | **Moderate: 6-10** | **Severe: >10**\n\n"
            "**Partial Mayo Score (pMayo)**: 0-1 (remission), 2-4 (mild), 5-7 (moderate), 8-9 (severe).\n"
            "**Target for treatment**: MES 0 (mucosal healing) is the gold standard therapeutic target."
        ),
        "severity": "all",
        "specialty": "gi",
    },

    # ─── FAMILY PLANNING ──────────────────────────────────────────────────────
    {
        "id": "IBD_FAMILY_PLANNING",
        "keywords": ["pregnancy", "family planning", "kehamilan", "ibu hamil", "conception", "fertility", "pregnant"],
        "condition": "Family Planning in IBD",
        "source": "ECCO Guidelines",
        "source_ref": "van der Woude CJ et al. ECCO Guidelines on Reproduction in IBD. J Crohns Colitis. 2015;9(2):107-124",
        "tier": 1,
        "year": 2015,
        "recommendation": (
            "**IBD Family Planning Recommendations:**\n"
            "- **Maintain remission** before conception — active disease decreases fertility\n"
            "These 5-ASA, thiopurines, anti-TNF (stop in 3rd trimester), vedolizumab medications were safe to be continued.\n"
            "- **CONTRAINDICATED**: Methotrexate (MUST discontinue ≥3 months before conception — both males & females), "
            "Tofacitinib (stop ≥4 weeks before conception)\n"
            "- **Corticosteroids**: Use lowest effective dose; budesonide preferred\n"
            "- Active disease during pregnancy carries higher risk than most IBD medications\n"
            "- **Males on sulfasalazine**: Temporary oligospermia; switch to mesalamine if planning conception"
        ),
        "severity": "all",
        "specialty": "gi",
    },

    
    # ─── DIET AND LIFESTYLE (Q5) ──────────────────────────────────────────────
    {
        "id": "IBD_DIET_NUTRITION",
        "keywords": ["diet", "dietary", "nutrition", "food", "what to eat", "mediterranean", "low-residue"],
        "condition": "Dietary Recommendations in IBD",
        "source": "ECCO Guidelines / IOIBD",
        "source_ref": "Levine A et al. Dietary Guidance from the IOIBD. Gut. 2020;69(6):947-953",
        "tier": 1,
        "year": 2020,
        "recommendation": (
            "**Dietary Recommendations for IBD:**\n"
            "- **For Patients in Remission**: A **Mediterranean** or **balanced diet** rich in **fresh vegetable**, **whole grain**, and **omega-3** fatty acids is recommended to maintain mucosal health and microbiome diversity. A generic **fiber-rich** diet is well-tolerated.\n"
            "- **For Active Disease / Strictures**: A **low-residue**, **low-fiber** diet with **cooked vegetable** and **white rice** is recommended temporarily to reduce bowel movements and abdominal pain."
        ),
        "severity": "all",
        "specialty": "gi",
    },
    {
        "id": "IBD_NUTRITION_SCREENING",
        "keywords": ["supplement", "deficiency", "screening", "vitamin", "mineral", "iron", "calcium", "b12"],
        "condition": "Nutritional Supplementation in IBD",
        "source": "ECCO Guidelines",
        "source_ref": "Forbes A et al. ESPEN guideline: Clinical nutrition in IBD. Clin Nutr. 2017;36(2):321-347",
        "tier": 1,
        "year": 2017,
        "recommendation": (
            "**Nutritional Screening and Supplementation:**\n"
            "- Screen all IBD patients for **iron deficiency anemia**, **Vitamin B12**, **Folate**, and **Vitamin D** deficiencies.\n"
            "- Recommend **Calcium and Vitamin D** supplementation, especially for patients on corticosteroids, to prevent osteopenia/osteoporosis."
        ),
        "severity": "all",
        "specialty": "gi",
    },
    {
        "id": "IBD_LIFESTYLE",
        "keywords": ["lifestyle", "habit", "stress", "smoking", "alcohol", "exercise", "weight"],
        "condition": "Lifestyle Modifications in IBD",
        "source": "ACG Clinical Guidelines",
        "source_ref": "Rubin DT et al. ACG Clinical Guideline. Am J Gastroenterol. 2019",
        "tier": 1,
        "year": 2019,
        "recommendation": (
            "**Lifestyle Modifications:**\n"
            "- **Smoking cessation** is critical. Smoking worsens Crohn's disease but its effect on UC is complex; regardless, cessation is recommended for general health.\n"
            "- Moderate **physical activity** and **exercise** are encouraged to improve fatigue and bone density.\n"
            "- Psychological **stress** and **mindfulness** management is highly recommended, as stress can precipitate flares.\n"
            "- Limit **alcohol** consumption.\n"
            "- Monitor **weight** and **BMI** to avoid malnutrition or obesity-related complications."
        ),
        "severity": "all",
        "specialty": "gi",
    },

    # ─── PREGNANCY AND MATERNAL RISKS (Q6) ────────────────────────────────────
    {
        "id": "IBD_MATERNAL_RISK",
        "keywords": ["maternal risk", "preeclampsia", "flare during pregnancy", "vte", "gestational"],
        "condition": "Maternal Risks in IBD Pregnancy",
        "source": "ECCO Guidelines",
        "source_ref": "van der Woude CJ et al. ECCO Guidelines on Reproduction in IBD. J Crohns Colitis. 2015",
        "tier": 1,
        "year": 2015,
        "recommendation": (
            "If active disease: [Tier 1] Maternally, the risk of relapse or worsening disease is increased compared to the non-IBD patients. [ECCO, 2015]\n"
            "If in remission: [Tier 1] Maternally, the risk of most adverse pregnancy events is comparable to the non-IBD patients. [ECCO, 2015]\n"
            "Controlling disease activity during pregnancy is critical to reduce adverse outcomes."
        ),
        "severity": "all",
        "specialty": "gi",
    },
    {
        "id": "IBD_NEONATAL_RISK",
        "keywords": ["fetal risk", "neonatal risk", "preterm", "birth weight", "sga", "congenital"],
        "condition": "Fetal/Neonatal Risks in IBD Pregnancy",
        "source": "ECCO Guidelines",
        "source_ref": "van der Woude CJ et al. ECCO Guidelines on Reproduction in IBD. J Crohns Colitis. 2015",
        "tier": 1,
        "year": 2015,
        "recommendation": (
            "If active disease: [Tier 1] Fetal and neonatal, the risk of preterm birth and low birth weight is increased compared to the non-IBD patients. [ECCO, 2015]\n"
            "If in remission: [Tier 1] Fetal and neonatal, the risk of preterm birth and low birth weight is comparable to the non-IBD patients. [ECCO, 2015]\n"
            "If anti-TNF used in T3: Infants may have neonatal immunosuppression; defer live vaccines for 6 months."
        ),
        "severity": "all",
        "specialty": "gi",
    },


    # ─── MONITORING AND SCREENING (Q4) ─────────────────────────────────────────
    {
        "id": "IBD_MONITORING_NONINVASIVE",
        "keywords": ["non-invasive monitoring", "non invasive", "biomarker monitoring", "fecal calprotectin monitoring", "crp monitoring"],
        "condition": "Non-Invasive Monitoring in IBD",
        "source": "STRIDE-II Guidelines",
        "source_ref": "Turner D et al. STRIDE-II. Gastroenterology 2021",
        "tier": 1,
        "year": 2021,
        "recommendation": (
            "[Tier 1] Based on the patient's current status, the following exams (FC and CRP / intestinal ultrasound) should be arranged at [3 months for active disease / 6-12 months for remission]. [STRIDE-II, 2021]"
        ),
        "severity": "all",
        "specialty": "gi",
    },
    {
        "id": "IBD_TDM",
        "keywords": ["therapeutic drug monitoring", "tdm", "proactive tdm", "reactive tdm", "drug level"],
        "condition": "Therapeutic Drug Monitoring (TDM) in IBD",
        "source": "AGA Guidelines",
        "source_ref": "Feuerstein JD et al. AGA Clinical Practice Guidelines on TDM. Gastroenterology 2017",
        "tier": 1,
        "year": 2017,
        "recommendation": (
            "For 5-ASA or Thiopurine monotherapy: No current evidence supports routine TDM for the patient.\n"
            "[Tier 1] Yes, proactive TDM is recommended for patients in remission on biologics/small-molecules, with target drug level depending on the agent. [AGA, 2017]\n"
            "[Tier 1] Yes, reactive TDM is recommended for patients with active disease on biologics/small-molecules. [AGA, 2017]"
        ),
        "severity": "all",
        "specialty": "gi",
    },
    {
        "id": "IBD_MED_SPECIFIC_MONITORING",
        "keywords": ["medication-specific monitoring", "drug monitoring requirement", "safety monitoring", "blood monitoring", "renal function monitoring"],
        "condition": "Medication-Specific Safety Monitoring in IBD",
        "source": "ECCO Guidelines",
        "source_ref": "Kucharzik T et al. ECCO Guidelines on Prevention of Opportunistic Infections. J Crohns Colitis 2021",
        "tier": 1,
        "year": 2021,
        "recommendation": (
            "[Tier 1] For patients under 5-ASA, renal function should be monitored annually. [ECCO, 2021]\n"
            "[Tier 1] For patients under thiopurine, CBC and LFTs should be monitored every 3-6 months. [ECCO, 2021]\n"
            "[Tier 1] For patients under biologics, TB screening before start is required; routine blood monitoring is not strictly mandated unless symptomatic. [ECCO, 2021]\n"
            "[Tier 1] For patients under JAK inhibitors, Lipids and CBC should be monitored at baseline, 3 months, then 6 monthly. [ECCO, 2021]"
        ),
        "severity": "all",
        "specialty": "gi",
    },
    {
        "id": "IBD_OPPORTUNISTIC_INFECTIONS",
        "keywords": ["opportunistic infection screening", "vaccination", "vaccine", "shingrix", "pneumococcal", "hepatitis", "influenza", "hpv"],
        "condition": "Vaccinations & Opportunistic Infection Screening in IBD",
        "source": "ECCO Guidelines",
        "source_ref": "Kucharzik T et al. ECCO Guidelines on Prevention of Opportunistic Infections. J Crohns Colitis 2021",
        "tier": 1,
        "year": 2021,
        "recommendation": (
            "[Tier 1] Screening for Influenza, Pneumococcal, Hepatitis B, and COVID-19 vaccinations prior to treatment initiation are recommended. [ECCO, 2021]\n"
            "If age > 50 or on JAK inhibitors: [Tier 1] Screening for Herpes Zoster (Shingrix) vaccination is recommended. [ECCO, 2021]\n"
            "If age <= 26: [Tier 1] Screening for HPV vaccination is recommended. [ECCO, 2021]"
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
        "tier": 1,
        "year": 2024,
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
        "tier": 1,
        "year": 2021,
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
        "tier": 1,
        "year": 2021,
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
        "tier": 1,
        "year": 2024,
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
    Returns a list of matching guidelines, sorted by relevance score,
    then by tier (ascending — Tier 1 first), then by year (descending — latest first).
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

    # Sort by: score descending, then tier ascending, then year descending
    matches.sort(key=lambda x: (-x["_score"], x.get("tier", 1), -x.get("year", 2020)))
    return matches


def format_guideline_answer(matches: List[Dict], question: str) -> str:
    """
    Formats matched guidelines into the ColonoSense tiered citation format.
    
    Output format:
    [Tier X] 1. Recommendation [Society/Author, Year]
    
    Groups by tier and orders by year descending within each tier.
    Always includes a 📚 Sources Consulted section at the bottom.
    """
    if not matches:
        return ""

    # Group matches by tier
    tier_groups: Dict[int, List[Dict]] = {}
    for m in matches[:5]:  # Max 5 guidelines
        tier = m.get("tier", 1)
        if tier not in tier_groups:
            tier_groups[tier] = []
        tier_groups[tier].append(m)

    # Sort each tier group by year descending
    for tier in tier_groups:
        tier_groups[tier].sort(key=lambda x: -x.get("year", 2020))

    TIER_LABELS = {
        1: "Global Guidelines",
        2: "Local Guidelines",
        3: "Meta-analyses",
        4: "Pivotal Trials",
    }

    parts = []
    sources_list = []
    recommendation_counter = 0

    # Output tiers in order 1 → 2 → 3 → 4
    TIER_ORDER = [1, 2, 3, 4]
    for tier_num in TIER_ORDER:
        if tier_num not in tier_groups:
            continue
        tier_label = TIER_LABELS.get(tier_num, f"Tier {tier_num}")
        tier_matches = tier_groups[tier_num]

        for m in tier_matches:
            recommendation_counter += 1
            source = m["source"]
            year = m.get("year", "")
            condition = m["condition"]
            recommendation = m["recommendation"]
            source_ref = m.get("source_ref", "")

            block = (
                f"**[Tier {tier_num}]** {recommendation_counter}. "
                f"{recommendation} "
                f"[{source}, {year}]\n\n"
                f"*Reference: {source_ref}*"
            )
            parts.append(block)
            sources_list.append(f"- **[Tier {tier_num}] {source}** ({year}): {source_ref}")

    # Assemble answer
    answer = "\n\n---\n\n".join(parts)

    # Always add Sources Consulted section
    sources_section = "\n".join(sources_list)
    answer += f"\n\n📚 **Sources Consulted:**\n{sources_section}"

    return answer
