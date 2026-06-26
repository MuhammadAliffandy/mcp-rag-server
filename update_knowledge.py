import re

with open("PineBioML/rag/clinical_knowledge.py", "r") as f:
    content = f.read()

new_guidelines = """
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
            "**Maternal Risks in Pregnancy:**\n"
            "- Pregnant women with IBD have a slightly **increased** risk of **preeclampsia**, **gestational** diabetes, and venous **thromboembolism** (**VTE**).\n"
            "- The risk of a disease **flare** during pregnancy is **comparable** to non-pregnant women if conception occurs during remission. If conception occurs during active disease, the flare often persists or worsens.\n"
            "- Risk of severe **infection** is a concern when using systemic steroids."
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
            "**Fetal and Neonatal Risks:**\n"
            "- Active disease at conception is associated with an **increased** risk of **preterm** birth, low **birth weight**, and infants being **small for gestational** age (**SGA**).\n"
            "- The risk of **congenital** abnormalities is generally **comparable** to the general population.\n"
            "- Biological agents (e.g., anti-TNF) cross the **placental** barrier. Infants born to mothers on biologics may have temporary **immunosuppression**; thus, **live vaccine** administration to the **neonatal** infant is contraindicated for the first 6 months of life."
        ),
        "severity": "all",
        "specialty": "gi",
    },
"""

insertion_point = "# ─── GENERAL MEDICAL ──────────────────────────────────────────────────────"
if insertion_point in content:
    new_content = content.replace(insertion_point, new_guidelines + "\n    " + insertion_point)
    with open("PineBioML/rag/clinical_knowledge.py", "w") as f:
        f.write(new_content)
    print("Successfully injected Q5/Q6 guidelines into clinical_knowledge.py")
else:
    print("Could not find insertion point!")
