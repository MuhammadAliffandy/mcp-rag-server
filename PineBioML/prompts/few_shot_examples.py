"""Few-shot examples for orchestration to teach LLM proper tool selection — ColonoSense v2."""

def get_few_shot_examples() -> str:
    """Returns comprehensive few-shot examples for agentic routing with reasoning, aligned with ColonoSense 7 categories."""
    return (
        "# FEW-SHOT EXAMPLES (Mental Models for ColonoSense Agent):\n\n"

        "## Example 1: Disease Severity Assessment (Category 1)\n"
        'User: "What is the severity of patient 123 based on recent labs?"\n'
        "Output: {\n"
        '  "answer": "**Category Recognized: Disease Severity & Remission (Category 1).** I will retrieve the recent clinical data, Mayo scores, Nancy scores, and lab results for patient 123 to classify severity and check remission criteria.",\n'
        '  "thoughts": "Category 1: Must retrieve Total Mayo (Partial Mayo + MES) to classify Remission (0-2), Mild (3-5), Moderate (6-10), Severe (>10). Also check remission checklist: pMayo <3, CRP <1, FC <100, MES 0-1, Nancy 0-1.",\n'
        '  "tasks": [\n'
        '    { "tool": "query_core_rag", "args": { "patient_id": "123", "query_intent": "retrieve Mayo score (stool, bleeding, physician), MES, Nancy score, CRP, FC, albumin, hemoglobin for severity classification and remission checklist" } }\n'
        "  ]\n"
        "}\n\n"

        "## Example 2: Treatment Adjustment (Category 2)\n"
        'User: "Patient 45 isn\'t responding to steroids. What is the recommended escalation?"\n'
        "Output: {\n"
        '  "answer": "**Category Recognized: Treatment Adjustment (Category 2).** I will check the patient\'s current medication history and recent clinical events, then consult the official medical guidelines for escalation SOPs.",\n'
        '  "thoughts": "Category 2: Steroid-dependent/refractory. Need current meds from Core RAG and escalation SOP from Guard RAG. Check if 5-ASA optimized to 4.8g/d before escalating. Steroid >12 weeks = escalation trigger.",\n'
        '  "tasks": [\n'
        '    { "tool": "query_core_rag", "args": { "patient_id": "45", "query_intent": "current medication list, steroid duration, dose, recent clinical events, disease severity" } },\n'
        '    { "tool": "query_guard_rag", "args": { "query_intent": "SOP for steroid-refractory or steroid-dependent UC escalation therapy including biologic options" } }\n'
        "  ]\n"
        "}\n\n"

        "## Example 3: Colon Cancer Surveillance (Category 3)\n"
        'User: "When should we schedule the next colonoscopy for patient 9?"\n'
        "Output: {\n"
        '  "answer": "**Category Recognized: Colon Cancer Surveillance (Category 3).** I will check the patient\'s disease duration, extent, PSC status, and family history, then retrieve the surveillance interval protocol.",\n'
        '  "thoughts": "Category 3: CRC screening starts 8 years after symptom onset. Need extent, PSC, family_hx_crc from UC_baseline. Intervals: High Risk 1yr (PSC, severe inflammation, CRC FH), Intermediate 2-3yr, Low Risk 5yr.",\n'
        '  "tasks": [\n'
        '    { "tool": "query_core_rag", "args": { "patient_id": "9", "query_intent": "disease duration from date_onset, extent, PSC status, family_hx_crc, current inflammation severity" } },\n'
        '    { "tool": "query_guard_rag", "args": { "query_intent": "CRC surveillance interval protocol based on disease duration, extent, PSC, and risk stratification" } }\n'
        "  ]\n"
        "}\n\n"

        "## Example 4: Monitor Tools and Interval (Category 4)\n"
        'User: "What monitoring should patient 22 receive and how often?"\n'
        "Output: {\n"
        '  "answer": "**Category Recognized: Monitor Tools and Interval (Category 4).** I will retrieve the patient\'s baseline status and current disease activity, then consult guidelines for recommended monitoring tools and intervals.",\n'
        '  "thoughts": "Category 4: Need current status from Core RAG (stable vs unstable). Guard RAG for monitoring interval guidelines. Stable = 3-6 months, unstable = can be daily.",\n'
        '  "tasks": [\n'
        '    { "tool": "query_core_rag", "args": { "patient_id": "22", "query_intent": "current disease activity, recent labs (CRP, FC, Hb), latest MES, medication status" } },\n'
        '    { "tool": "query_guard_rag", "args": { "query_intent": "recommended monitoring tools, lab intervals, and endoscopy timing for UC patients based on disease activity" } }\n'
        "  ]\n"
        "}\n\n"

        "## Example 5: Risk of Complications (Category 5)\n"
        'User: "Can you calculate the statistical risk score for complication in patient 88?"\n'
        "Output: {\n"
        '  "answer": "**Category Recognized: Risk of Complications (Category 5).** I will gather the patient\'s historical events and compute the statistical risk score using the ML engine.",\n'
        '  "thoughts": "Category 5: Risk computation requires historical events from Core RAG (ED, hospitalizations, surgery) then PineBio ML for statistical risk score.",\n'
        '  "tasks": [\n'
        '    { "tool": "query_core_rag", "args": { "patient_id": "88", "query_intent": "historical clinical events including ED visits, hospitalizations, surgeries, CRC, clinical trials" } },\n'
        '    { "tool": "execute_pinebio_ml", "args": { "data_payload": "patient_88_history", "task_type": "compute statistical risk score for complications" } }\n'
        "  ]\n"
        "}\n\n"

        "## Example 6: Lifestyle and Diet Modification (Category 6)\n"
        'User: "What should patient 12 eat to avoid a flare up?"\n'
        "Output: {\n"
        '  "answer": "**Category Recognized: Lifestyle and Diet Modification (Category 6).** I will consult the official dietary and lifestyle guidelines for UC patients.",\n'
        '  "thoughts": "Category 6: Diet/lifestyle = Guard RAG primary. Core RAG only if patient has documented allergies or restrictions.",\n'
        '  "tasks": [\n'
        '    { "tool": "query_guard_rag", "args": { "query_intent": "official dietary and lifestyle guidelines tailored to UC patients to prevent flare-ups" } }\n'
        "  ]\n"
        "}\n\n"

        "## Example 7: Family Planning (Category 7)\n"
        'User: "Is it safe for a 30yo female patient on Infliximab to pursue pregnancy?"\n'
        "Output: {\n"
        '  "answer": "**Category Recognized: Family Planning (Category 7).** I will verify the patient\'s current medication list and fetch the family planning protocols.",\n'
        '  "thoughts": "Category 7: Need sex, age, current meds from Core RAG. Pregnancy protocols from Guard RAG. Critical: check methotrexate (must discontinue before conception).",\n'
        '  "tasks": [\n'
        '    { "tool": "query_guard_rag", "args": { "query_intent": "family planning and pregnancy safety protocols related to Infliximab and other IBD biologics in reproductive-age females" } }\n'
        "  ]\n"
        "}\n\n"

        "## Example 8: Trend Analysis (Category 1 + ML)\n"
        'User: "Analyze the trend of Mayo scores for this patient over the last 3 visits."\n'
        "Output: {\n"
        '  "answer": "**Category Recognized: Disease Severity Assessment (Category 1) + Trend Analysis.** I will retrieve the historical Mayo scores and perform a trend analysis.",\n'
        '  "thoughts": "Category 1 + ML: Need longitudinal Mayo data from Core RAG, then PineBio ML for trend analysis. Check if trajectory is improving or deteriorating.",\n'
        '  "tasks": [\n'
        '    { "tool": "query_core_rag", "args": { "patient_id": "current", "query_intent": "historical Mayo Endoscopic Subscore, Partial Mayo, and Nancy Score over last 3 visits with dates" } },\n'
        '    { "tool": "execute_pinebio_ml", "args": { "data_payload": "historical_scores", "task_type": "trend analysis of severity scores across visits" } }\n'
        "  ]\n"
        "}\n\n"
    )
