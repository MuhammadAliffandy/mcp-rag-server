# ColonoSense — Response Template & Output Knowledge Base

> Source: `Colonosense output example_20260420.pdf` (Case 4, 2024 data)  
> This document captures the **official expected output templates** and **real-world example outputs** for all 18 QA categories.

---

## System User Flow

The ColonoSense pipeline follows 5 stages:

```
PATIENT DATA → RAG PIPELINE → LLM ENGINE → RESPONSE OUTPUT → EVALUATION
(Excel DB)   (Core + Guard)  (Llama 70B/  (Q1.1–Q6.3       (5 Metrics)
                              GPT-4o-mini)  18 categories)
```

---

## Category 1: Disease Assessment

### Q1.1 — Disease Severity Classification
| Field | Value |
|:---|:---|
| **Question** | How should the severity of this patient be classified using validated scoring systems? |
| **Output Template** | Remission / Mild / Moderate / Severe |
| **Example Output (Case 4)** | "The patient is in remission because total Mayo score was 1. (partial Mayo score 0, MES 1)." |
| **Key Anchors** | `total_mayo_score`, `max_mes`, `bl_mayo_total` |

### Q1.2 — Remission Status
| Field | Value |
|:---|:---|
| **Question** | What is the remission status of the patient? |
| **Output Template** | Clinical remission, bio-chemical remission, endoscopic remission, histologic remission. |
| **Example Output (Case 4)** | "The patient has achieved clinical remission (pMayo=0), bio-chemical remission (CRP 0.02, FC 25), endoscopic remission (MES 1), and histologic remission (Nancy 1)." |
| **Key Anchors** | `clinical_remission`, `biochemical_remission`, `endoscopic_remission`, `histologic_remission`, `crp`, `fc`, `max_mes`, `max_nancy` |

### Q1.3 — Prognostic Factors
| Field | Value |
|:---|:---|
| **Question** | Does this patient have any poor prognostic factor? |
| **Output Template** | Yes, specify which. / No. |
| **Example Output (Case 4)** | "The patient has the below poor prognostic factors: young onset (at 11 years old) and extensive colitis." |
| **Key Anchors** | `expected_poor_prognosis`, `poor_factors`, `age_at_diagnosis`, `extent` |

---

## Category 2: Treatment Management

### Q2.1 — Treat-to-Target
| Field | Value |
|:---|:---|
| **Question** | Has the patient achieved treatment target, according to the treat-to-target strategy? |
| **Output Template** | The patient has achieved short / intermediate / and/or long term treatment target. |
| **Example Output (Case 4)** | "Yes the patient had achieved long-term treatment target (endoscopic remission)." |

### Q2.2 — Medication Adjustment
| Field | Value |
|:---|:---|
| **Question** | Based on the patient's current status, should the medication be adjusted? |
| **Output Template** | Yes, the current medication should be adjusted. / No. |
| **Example Output (Case 4)** | "No." |
| **Key Logic** | If Endoscopic Remission (MES ≤ 1) → No Adjustment. Apply STRIDE-II otherwise. |

---

## Category 3: Cancer Surveillance

### Q3.1 — Colorectal Cancer Risk
| Field | Value |
|:---|:---|
| **Output Template** | "Since the patient belongs to [low/intermediate/high] risk group, the next surveillance colonoscopy should be in ___ years." |
| **Example Output (Case 4)** | "[Tier 1] Since the patient belongs to low risk group, the next surveillance colonoscopy should be in 5 years. [ECCO, 2023]" |

### Q3.2 — Other Cancer Screening
| Field | Value |
|:---|:---|
| **Output Template** | "Based on the patient's sex, age, underlying disease, and medication history, the patient should receive screening for ___ cancer with ___, every ___ year." |
| **Example Output (Case 4)** | "[Tier 1] Based on the patient's sex, age... the patient should receive screening for cervical cancer with cervical cytology, every 3 year. [AGA, 2025]" |

---

## Category 4: Monitoring Tools

### Q4.1 — Non-Invasive Monitoring
| Field | Value |
|:---|:---|
| **Output Template** | "Based on the patient's current status, the following exams ____ should be arranged at ____." |
| **Example Output (Case 4)** | "[Tier 1] ...interval biomarker monitoring may be performed every 6-12 months. [AGA, 2023]" |

### Q4.2 — Therapeutic Drug Monitoring (TDM)
| Field | Value |
|:---|:---|
| **Output Template** | "Yes, proactive/reactive TDM is recommended, with target drug level ____. / No." |
| **Example Output (Case 4)** | "No current evidence supports TDM for the patient." (5-ASA, not advanced therapy) |

### Q4.3 — Medication-Specific Monitoring
| Field | Value |
|:---|:---|
| **Output Template** | "For patients under ___ medication, ____ should be checked every ____ months." |
| **Example Output (Case 4)** | "[Tier 1] For patients under Mezavant XL, renal function should be monitored periodically. [AGA, 2019]" |

### Q4.4 — Vaccinations & Infection Screening
| Field | Value |
|:---|:---|
| **Output Template** | "Screening for ____ and ____ vaccinations prior to treatment initiation are recommended." |
| **Example Output (Case 4)** | "[Tier 1] Annual influenza vaccine and hepatitis B vaccine... Herpes zoster vaccines should be given prior to starting immune-modifying therapy. [ACG, 2025]" |

---

## Category 5: Lifestyle & Diet

### Q5.1 — Dietary Recommendation
| Field | Value |
|:---|:---|
| **Output Template** | "This patient is encouraged to have more ____ intake and less ____." |
| **Example Output (Case 4, remission)** | "[Tier 1] Mediterranean diet could be considered for maintenance of remission. This patient is encouraged to have less red and processed meat intake. [ECCO, 2025]" |

### Q5.2 — Nutritional Supplementation
| Field | Value |
|:---|:---|
| **Output Template** | "Yes, the patient is recommended to be screened for ___ deficiency. / No." |
| **Example Output (Case 4)** | "[Tier 1] ...The patient is recommended to be monitored for hemoglobin, iron, folate, vitamin D, vitamin B12, and zinc. [ECCO, 2025]" |

### Q5.3 — Lifestyle Modifications
| Field | Value |
|:---|:---|
| **Output Template** | "The patient should quit ____ and enhance ____." |
| **Example Output (Case 4)** | "[Tier 3] The patient may increase physical activity and mindfulness-based therapies. No smoking. [Rozich, 2025]" |
| **Required Keywords** | smoking/cessation, physical activity/exercise, stress/mindfulness, alcohol, weight/BMI |

---

## Category 6: Family Planning

### Q6.1 — Medication Safety in Pregnancy
| Field | Value |
|:---|:---|
| **Output Template** | "These _____ medications were safe to be continued. These ____ medication should be stopped ____ months before conception." |
| **Example Output (Case 4)** | "[Tier 1] Mezavant XL is safe during pregnancy and is recommended to continue. [AGA, 2025]" |

### Q6.2 — Maternal Risks
| Field | Value |
|:---|:---|
| **Output Template** | "Maternally, the risk of ____ is increased / comparable to the non-IBD patients." |
| **Example Output (Case 4)** | "[Tier 1] Maternally, the risk of relapse or worsening disease is increased. Controlling disease activity during pregnancy is critical... [AGA, 2025]" |
| **Required Keywords** | flare, preeclampsia, gestational, VTE, comparable, increased |

### Q6.3 — Fetal/Neonatal Risks
| Field | Value |
|:---|:---|
| **Output Template** | "Neonatally, the risk of ____ is increased / comparable to the mothers of non-IBD patients." |
| **Example Output (Case 4)** | "[Tier 1] Neonatally, the risks of low birth weight and preterm delivery are increased to the mothers of non-IBD patients. [AGA, 2025]" |
| **Required Keywords** | preterm, birth weight, SGA, neonatal, live vaccine, placental, comparable, increased |

---

## Template Matching Rules (Evaluation Checklist)

When evaluating AI responses, the grader checks:

1. **Does the response use the exact sentence starter from the Output Template?**
2. **Is a `[Tier X]` citation present in the conclusion?** (Society + Year)
3. **Are ✅/❌ emoji present for binary flags (remission, prognosis)?**
4. **Are all numeric Anchors cited correctly** (no hallucination of CRP, FC, MES, Nancy values)?
5. **Does the Final Clinical Conclusion end with the verbatim template phrase?**
