"""Few-shot examples for orchestration to teach LLM proper tool selection."""

def get_few_shot_examples() -> str:
    """Returns comprehensive few-shot examples for agentic routing with reasoning."""
    return (
        "# FEW-SHOT EXAMPLES (Mental Models for the Agent):\n\n"

        "## Example 1: Disease Severity Assessment (Category 1)\n"
        'User: "What is the severity of patient 123 based on recent labs?"\n'
        "Output: {\n"
        '  "answer": "I will retrieve the recent clinical data and lab results for patient 123.",\n'
        '  "thoughts": "User asks for disease severity. Must call query_core_rag for patient data.",\n'
        '  "tasks": [\n'
        '    { "tool": "query_core_rag", "args": { "patient_id": "123", "query_intent": "retrieve Mayo score, Nancy score, and lab data" } }\n'
        "  ]\n"
        "}\n\n"

        "## Example 2: Treatment Adjustment (Category 2)\n"
        'User: "Patient 45 isn\'t responding to steroids. What is the recommended escalation?"\n'
        "Output: {\n"
        '  "answer": "I will check the patient\'s current medication history and then consult the official medical guidelines for dose adjustment SOPs.",\n'
        '  "thoughts": "Treatment adjustment requires patient context from core RAG, and SOP from guard RAG.",\n'
        '  "tasks": [\n'
        '    { "tool": "query_core_rag", "args": { "patient_id": "45", "query_intent": "current medication and recent clinical events" } },\n'
        '    { "tool": "query_guard_rag", "args": { "query_intent": "SOP for dosing adjustments in steroid refractory patients" } }\n'
        "  ]\n"
        "}\n\n"

        "## Example 3: Risk of Complications (Category 5)\n"
        'User: "Can you calculate the statistical risk score for complication in patient 88?"\n'
        "Output: {\n"
        '  "answer": "I will gather the patient\'s historical events and compute the statistical risk score using the ML engine.",\n'
        '  "thoughts": "Risk score calculation requires gathering data via core RAG and executing PineBio ML.",\n'
        '  "tasks": [\n'
        '    { "tool": "query_core_rag", "args": { "patient_id": "88", "query_intent": "historical events, emergency visits, hospitalizations" } },\n'
        '    { "tool": "execute_pinebio_ml", "args": { "data_payload": "patient_88_history", "task_type": "compute statistical risk score" } }\n'
        "  ]\n"
        "}\n\n"

        "## Example 4: Lifestyle and Diet Modification (Category 6)\n"
        'User: "What should patient 12 eat to avoid a flare up?"\n'
        "Output: {\n"
        '  "answer": "I will consult the official dietary guidelines for UC patients.",\n'
        '  "thoughts": "Diet modification falls under Guard RAG. Core RAG can be checked for allergies if needed, but Guard RAG is primary.",\n'
        '  "tasks": [\n'
        '    { "tool": "query_guard_rag", "args": { "query_intent": "official dietary and lifestyle guidelines tailored to UC patients" } }\n'
        "  ]\n"
        "}\n\n"

        "## Example 5: Colon Cancer Surveillance Timing (Category 3)\n"
        'User: "When should we schedule the next colonoscopy for patient 9?"\n'
        "Output: {\n"
        '  "answer": "I will check the patient\'s disease duration and then retrieve the hospital\'s surveillance interval protocol.",\n'
        '  "thoughts": "Surveillance timing requires patient disease duration (Core RAG) and protocol (Guard RAG).",\n'
        '  "tasks": [\n'
        '    { "tool": "query_core_rag", "args": { "patient_id": "9", "query_intent": "disease duration and extent" } },\n'
        '    { "tool": "query_guard_rag", "args": { "query_intent": "colon cancer surveillance interval protocol for specific disease duration" } }\n'
        "  ]\n"
        "}\n\n"

        "## Example 6: Family Planning (Category 7)\n"
        'User: "Is it safe for a 30yo female patient on Infliximab to pursue pregnancy?"\n'
        "Output: {\n"
        '  "answer": "I will verify the patient\'s current medication list and fetch the family planning protocols related to those medications.",\n'
        '  "thoughts": "Requires current meds from Core RAG (if specific patient) and pregnancy protocols from Guard RAG. Assuming implicit patient context, routing to Guard RAG.",\n'
        '  "tasks": [\n'
        '    { "tool": "query_guard_rag", "args": { "query_intent": "family planning and pregnancy protocols related to Infliximab in 30yo female" } }\n'
        "  ]\n"
        "}\n\n"

        "## Example 7: Statistical Analysis & Trends (Category 1)\n"
        'User: "Analyze the trend of Mayo scores for this patient over the last 3 visits."\n'
        "Output: {\n"
        '  "answer": "I will retrieve the historical Mayo scores and perform a trend analysis.",\n'
        '  "thoughts": "Trend analysis requires PineBio ML on data retrieved from Core RAG.",\n'
        '  "tasks": [\n'
        '    { "tool": "query_core_rag", "args": { "patient_id": "current", "query_intent": "historical Mayo Endoscopic Subscore and Nancy Score over last 3 visits" } },\n'
        '    { "tool": "execute_pinebio_ml", "args": { "data_payload": "historical_scores", "task_type": "trend analysis of severity scores" } }\n'
        "  ]\n"
        "}\n\n"
    )
