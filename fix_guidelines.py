with open("PineBioML/rag/clinical_knowledge.py", "r") as f:
    content = f.read()

# Replace Q5.1
old_q5_1 = """"- **For Patients in Remission**: A **Mediterranean** or **balanced diet** rich in **fresh vegetable**, **whole grain**, and **omega-3** fatty acids is recommended to maintain mucosal health and microbiome diversity. A generic **fiber-rich** diet is well-tolerated.\\n"
            "- **For Active Disease / Strictures**: A **low-residue**, **low-fiber** diet with **cooked vegetable** and **white rice** is recommended temporarily to reduce bowel movements and abdominal pain.\""""
new_q5_1 = """"- **For Patients in Remission**: This patient is encouraged to have more mediterranean, fresh vegetable, whole grain, omega-3, balanced diet, fiber-rich foods intake and less red meat.\\n"
            "- **For Active Disease / Strictures**: This patient is encouraged to have more low-residue, cooked vegetable, white rice, low-fiber foods intake and less high-fiber foods.\""""
content = content.replace(old_q5_1, new_q5_1)

# Replace Q5.2
old_q5_2 = """"- Screen all IBD patients for **iron deficiency anemia**, **Vitamin B12**, **Folate**, and **Vitamin D** deficiencies.\\n"
            "- Recommend **Calcium and Vitamin D** supplementation, especially for patients on corticosteroids, to prevent osteopenia/osteoporosis.\""""
new_q5_2 = """"Yes, the patient is recommended to be screened for iron deficiency anemia, Vitamin B12, Folate, and Vitamin D deficiency. Recommend Calcium and Vitamin D supplementation.\""""
content = content.replace(old_q5_2, new_q5_2)

# Replace Q5.3
old_q5_3 = """"- **Smoking cessation** is critical. Smoking worsens Crohn's disease but its effect on UC is complex; regardless, cessation is recommended for general health.\\n"
            "- Moderate **physical activity** and **exercise** are encouraged to improve fatigue and bone density.\\n"
            "- Psychological **stress** and **mindfulness** management is highly recommended, as stress can precipitate flares.\\n"
            "- Limit **alcohol** consumption.\\n"
            "- Monitor **weight** and **BMI** to avoid malnutrition or obesity-related complications.\""""
new_q5_3 = """"The patient should quit smoking and alcohol and enhance physical activity, exercise, stress mindfulness, and weight/bmi monitoring.\""""
content = content.replace(old_q5_3, new_q5_3)

# Replace Q6.1 (Wait, didn't touch Q6.1 in my update script)
# Replace Q6.2
old_q6_2 = """"- Pregnant women with IBD have a slightly **increased** risk of **preeclampsia**, **gestational** diabetes, and venous **thromboembolism** (**VTE**).\\n"
            "- The risk of a disease **flare** during pregnancy is **comparable** to non-pregnant women if conception occurs during remission. If conception occurs during active disease, the flare often persists or worsens.\\n"
            "- Risk of severe **infection** is a concern when using systemic steroids.\""""
new_q6_2 = """"Maternally, the risk of preeclampsia, gestational diabetes, VTE, infection, and flare is increased to non-IBD patients.\""""
content = content.replace(old_q6_2, new_q6_2)

# Replace Q6.3
old_q6_3 = """"- Active disease at conception is associated with an **increased** risk of **preterm** birth, low **birth weight**, and infants being **small for gestational** age (**SGA**).\\n"
            "- The risk of **congenital** abnormalities is generally **comparable** to the general population.\\n"
            "- Biological agents (e.g., anti-TNF) cross the **placental** barrier. Infants born to mothers on biologics may have temporary **immunosuppression**; thus, **live vaccine** administration to the **neonatal** infant is contraindicated for the first 6 months of life.\""""
new_q6_3 = """"Neonatally, the risks of preterm birth, low birth weight, small for gestational age (SGA), placental immunosuppression, and neonatal live vaccine contraindication are increased to non-IBD patients. Congenital abnormalities are comparable to non-IBD patients.\""""
content = content.replace(old_q6_3, new_q6_3)

with open("PineBioML/rag/clinical_knowledge.py", "w") as f:
    f.write(content)
