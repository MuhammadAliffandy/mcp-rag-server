#!/bin/bash

# Default values (can be overridden via arguments)
START_PATIENT=${1:-1}
TOTAL_PATIENTS=${2:-63}

# Create a dedicated directory for the final DOCX reports
mkdir -p final_docx_reports

echo "Starting Batch Generation from Patient $START_PATIENT to $TOTAL_PATIENTS..."

for i in $(seq $START_PATIENT $TOTAL_PATIENTS)
do
    echo ""
    echo "========================================"
    echo "Processing Patient ID: $i"
    echo "========================================"
    
    # 1. Run the RAG pipeline for all categories (Q1.1 - Q6.3)
    # This automatically saves a JSON file in the 'eval_results' folder.
    python run_eval.py --patient_id $i --category all
    
    # 2. Find the newly generated JSON file
    # We use 'ls -t' to get the most recently created JSON for this specific patient
    LATEST_JSON=$(ls -t eval_report_${i}_*.json 2>/dev/null | head -n 1)
    
    if [ -n "$LATEST_JSON" ] && [ -f "$LATEST_JSON" ]; then
        DOCX_FILE="final_docx_reports/ColonoSense_Report_Patient_${i}.docx"
        
        # 3. Export to Word DOCX
        python ppt_and_reports/export_patient_report.py "$LATEST_JSON" "$DOCX_FILE"
        echo "✅ Successfully exported Patient $i to $DOCX_FILE"
    else
        echo "❌ Failed to find JSON output for Patient $i. Skipping to next."
    fi
done

echo ""
echo "🎉 All done! You can find all 63 reports in the 'final_docx_reports' folder."
