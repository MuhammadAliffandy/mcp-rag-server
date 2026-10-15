#!/bin/bash

# Default values (can be overridden via arguments)
START_PATIENT=${1:-1}
TOTAL_PATIENTS=${2:-63}

# Create a dedicated directory for the final DOCX reports
mkdir -p final_docx_reports

# Check if python has the required packages
if ! python -c "import dotenv; import docx" 2>/dev/null; then
    echo "❌ Error: Current Python is missing required packages ('dotenv' or 'docx')."
    echo "Current python: $(which python)"
    echo "Please activate your conda environment: 'conda activate colono_env'"
    echo "and ensure packages are installed: 'pip install python-dotenv python-docx'"
    exit 1
fi

echo "Starting Batch Generation from Patient $START_PATIENT to $TOTAL_PATIENTS..."
echo "Using Python: $(which python)"

export PYTHONUNBUFFERED=1

for i in $(seq $START_PATIENT $TOTAL_PATIENTS)
do
    echo ""
    echo "========================================"
    echo "Processing Patient ID: $i"
    echo "========================================"
    
    # 1. Run the RAG pipeline for all categories (Q1.1 - Q6.3) with unbuffered output
    python -u run_eval.py --patient_id $i --category all
    
    # 2. Find the newly generated JSON file
    LATEST_JSON=$(ls -t eval_report_${i}_*.json 2>/dev/null | head -n 1)
    
    if [ -n "$LATEST_JSON" ] && [ -f "$LATEST_JSON" ]; then
        DOCX_FILE="final_docx_reports/ColonoSense_Report_Patient_${i}.docx"
        
        # 3. Export to Word DOCX
        if python ppt_and_reports/export_patient_report.py "$LATEST_JSON" "$DOCX_FILE"; then
            echo "✅ Successfully exported Patient $i to $DOCX_FILE"
        else
            echo "❌ Failed to export Patient $i to DOCX"
        fi
    else
        echo "❌ Failed to find JSON output for Patient $i. Skipping to next."
    fi
done

echo ""
echo "🎉 All done! You can find all 63 reports in the 'final_docx_reports' folder."
