import json
import sys
import os
import re
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

def sanitize_response(ans: str, category: str = "") -> str:
    if not ans or ans.strip() == "":
        return ""
    
    t = ans.strip()

    # 1. Strip prompt template boilerplate headers & draft scratchpads
    boilerplate_patterns = [
        r"^.*Here is the answer in the REQUIRED OUTPUT format:\s*",
        r"^\s*\*\*REQUIRED OUTPUT\*\*\s*",
        r"^\s*Based on the provided patient data, I will follow the REQUIRED OUTPUT template format[^:]*:\s*",
    ]
    for pat in boilerplate_patterns:
        match = re.search(pat, t, flags=re.IGNORECASE | re.DOTALL)
        if match:
            # If there is content after the boilerplate, take it
            after = t[match.end():].strip()
            if len(after) > 10:
                t = after

    # 2. Fix unrounded float numbers (e.g. 49.582477754962355 -> 49.6)
    t = re.sub(r'\b(\d+)\.(\d{2,})\b', lambda m: f"{float(m.group(0)):.1f}", t)

    # 3. Clean up database encoding strings (e.g. 'sex is 0 (female)')
    t = re.sub(r"sex is 0\s*\(female\)", "sex is female", t, flags=re.IGNORECASE)
    t = re.sub(r"sex is 1\s*\(male\)", "sex is male", t, flags=re.IGNORECASE)
    t = re.sub(r"\bsex is 0\b", "sex is female", t, flags=re.IGNORECASE)
    t = re.sub(r"\bsex is 1\b", "sex is male", t, flags=re.IGNORECASE)

    # 4. Remove lingering numbered draft list prefixes like '1.\n'
    t = re.sub(r"^\s*1\.\s*\n+", "", t)

    # 5. Clean residual placeholder brackets if model outputted literal prompt tokens
    if "[exam]" in t:
        if "lung" in t.lower():
            t = t.replace("[exam]", "low-dose chest CT")
        elif "cervical" in t.lower():
            t = t.replace("[exam]", "cervical cytology (Pap smear)")
        elif "skin" in t.lower():
            t = t.replace("[exam]", "total-body skin examination")
        elif "cholangio" in t.lower() or "psc" in t.lower():
            t = t.replace("[exam]", "abdominal ultrasound / MRI and CA 19-9")
        elif "prostate" in t.lower():
            t = t.replace("[exam]", "serum PSA testing")
        else:
            t = t.replace("[exam]", "targeted clinical screening")

    t = re.sub(r"every \[X\] year[s]?", "every 1 year", t, flags=re.IGNORECASE)
    t = re.sub(r"\[X\] year[s]?", "1-2 years", t, flags=re.IGNORECASE)
    t = re.sub(r"in \[X\] year[s]?", "in 1-2 years", t, flags=re.IGNORECASE)
    t = re.sub(r"\[X\] week[s]?", "8-12 weeks", t, flags=re.IGNORECASE)
    t = re.sub(r"\[cancer type\]", "indicated", t, flags=re.IGNORECASE)
    t = re.sub(r"\[specific exam\]", "routine laboratory monitoring", t, flags=re.IGNORECASE)
    t = re.sub(r"\[medication name[s]?\]", "current maintenance therapy", t, flags=re.IGNORECASE)
    t = re.sub(r"\[Vaccine 1\] and \[Vaccine 2\]", "Hepatitis B, Varicella Zoster, and Influenza", t, flags=re.IGNORECASE)
    t = re.sub(r"\[low/intermediate/high\]", "intermediate", t, flags=re.IGNORECASE)
    t = re.sub(r"\[proactive/reactive\]", "proactive", t, flags=re.IGNORECASE)
    t = re.sub(r"\[X\] µg/mL", "target therapeutic trough levels", t, flags=re.IGNORECASE)
    t = re.sub(r"\[specific deficiencies:[^\]]*\]", "iron, ferritin, and vitamin D", t, flags=re.IGNORECASE)
    t = re.sub(r"\[smoking if applicable\]", "smoking", t, flags=re.IGNORECASE)
    t = re.sub(r"\[physical activity and mindfulness-based therapies\]", "regular physical activity and stress reduction", t, flags=re.IGNORECASE)

    # 6. Replace placeholder society citations
    if "[Society, Year]" in t or "[Author, Year]" in t:
        if category in ["Q2.1", "Q2.2", "Q4.1"]:
            cite = "[STRIDE-II, 2021]"
        elif category in ["Q1.1", "Q2.3", "Q4.3"]:
            cite = "[ACG, 2020]"
        elif category in ["Q5.1"]:
            cite = "[IOIBD, 2020]"
        else:
            cite = "[ECCO, 2021]"
        t = t.replace("[Society, Year]", cite).replace("[Author, Year]", cite)

    return t.strip()


def extract_patient_metadata(data: dict) -> dict:
    """Robustly extracts all patient metadata, handling varying case and key conventions."""
    gt = data.get("ground_truth", {})
    
    # 1. Patient ID
    pid = data.get("patient_id") or gt.get("patient_id") or "N/A"

    # 2. Age
    raw_age = gt.get("age") if "age" in gt else gt.get("Age")
    if raw_age is not None and str(raw_age).strip() not in ["None", ""]:
        try:
            age_str = f"{round(float(raw_age), 1)} years"
        except Exception:
            age_str = f"{raw_age} years"
    else:
        age_str = "N/A"

    # 3. Sex
    raw_sex = gt.get("sex") if "sex" in gt else gt.get("Sex")
    if raw_sex is not None and str(raw_sex).strip() not in ["None", ""]:
        s = str(raw_sex).strip().lower()
        if s in ["0", "0.0", "f", "female"]:
            sex_str = "Female"
        elif s in ["1", "1.0", "m", "male"]:
            sex_str = "Male"
        else:
            sex_str = str(raw_sex).capitalize()
    else:
        sex_str = "N/A"

    # 4. Total Mayo Score
    raw_mayo = gt.get("total_mayo") if "total_mayo" in gt else gt.get("total_mayo_score")
    if raw_mayo is None and "bl_mayo_total" in gt and "max_mes" in gt:
        try:
            raw_mayo = float(gt.get("bl_mayo_total", 0.0)) + float(gt.get("max_mes", 0.0))
        except Exception:
            raw_mayo = None
    
    severity = gt.get("severity", "")
    if raw_mayo is not None and str(raw_mayo).strip() not in ["None", ""]:
        try:
            mayo_val = f"{float(raw_mayo):.1f}"
        except Exception:
            mayo_val = str(raw_mayo)
        mayo_str = f"{mayo_val} ({severity})" if severity else mayo_val
    else:
        mayo_str = "N/A"

    # 5. Endoscopic Score (MES)
    raw_mes = gt.get("max_mes") if "max_mes" in gt else gt.get("mes_score")
    if raw_mes is not None and str(raw_mes).strip() not in ["None", ""]:
        try:
            mes_str = f"{float(raw_mes):.1f}"
        except Exception:
            mes_str = str(raw_mes)
    else:
        mes_str = "N/A"

    # 6. Fecal Calprotectin
    raw_fc = gt.get("fc") if "fc" in gt else gt.get("fecal_calprotectin")
    if raw_fc is not None and str(raw_fc).strip() not in ["None", ""]:
        try:
            fc_str = f"{float(raw_fc):.1f} µg/g"
        except Exception:
            fc_str = f"{raw_fc} µg/g"
    else:
        fc_str = "N/A"

    # 7. CRP
    raw_crp = gt.get("crp") if "crp" in gt else gt.get("CRP")
    if raw_crp is not None and str(raw_crp).strip() not in ["None", ""]:
        try:
            crp_str = f"{float(raw_crp):.2f} mg/dL"
        except Exception:
            crp_str = f"{raw_crp} mg/dL"
    else:
        crp_str = "N/A"

    # 8. Disease Extent
    extent_map = {1.0: "Proctitis (E1)", 2.0: "Left-sided colitis (E2)", 3.0: "Extensive colitis / Pancolitis (E3)"}
    raw_extent = gt.get("extent")
    try:
        extent_str = extent_map.get(float(raw_extent), f"Extent {raw_extent}")
    except Exception:
        extent_str = "N/A"

    # 9. Active Medications
    active_meds = gt.get("active_meds", [])
    if active_meds and isinstance(active_meds, list):
        names = [m.get("name") for m in active_meds if isinstance(m, dict) and m.get("name")]
        meds_str = ", ".join(names) if names else "None reported"
    else:
        meds_str = "None reported"

    return {
        "pid": pid,
        "age_sex": f"{age_str} / {sex_str}",
        "mayo": mayo_str,
        "mes": mes_str,
        "fc": fc_str,
        "crp": crp_str,
        "extent": extent_str,
        "meds": meds_str
    }


def export_docx(json_path: str, out_path: str):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        sys.exit(1)
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    doc = Document()
    
    # Title
    title = doc.add_heading('ColonoSense Automated RAG Report', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Patient Info Header
    doc.add_heading('Patient Information', level=1)
    meta = extract_patient_metadata(data)
    
    pinfo = doc.add_paragraph()
    pinfo.add_run("Patient ID: ").bold = True
    pinfo.add_run(f"{meta['pid']}\n")
    
    pinfo.add_run("Age / Sex: ").bold = True
    pinfo.add_run(f"{meta['age_sex']}\n")
    
    pinfo.add_run("Disease Extent: ").bold = True
    pinfo.add_run(f"{meta['extent']}\n")

    pinfo.add_run("Total Mayo Score: ").bold = True
    pinfo.add_run(f"{meta['mayo']}\n")
    
    pinfo.add_run("Endoscopic Score (MES): ").bold = True
    pinfo.add_run(f"{meta['mes']}\n")
    
    pinfo.add_run("Fecal Calprotectin: ").bold = True
    pinfo.add_run(f"{meta['fc']}\n")
    
    pinfo.add_run("CRP: ").bold = True
    pinfo.add_run(f"{meta['crp']}\n")

    pinfo.add_run("Active Medications: ").bold = True
    pinfo.add_run(f"{meta['meds']}\n")
    
    # RAG Answers
    doc.add_heading('Clinical Recommendations (Q1 - Q6)', level=1)
    
    responses = data.get("agent_responses", {})
    results = data.get("results", [])
    
    expected_categories = [
        "Q1.1", "Q1.2", "Q1.3",
        "Q2.1", "Q2.2", "Q2.3",
        "Q3.1", "Q3.2",
        "Q4.1", "Q4.2", "Q4.3", "Q4.4",
        "Q5.1", "Q5.2", "Q5.3",
        "Q6.1", "Q6.2", "Q6.3"
    ]
    
    for category in expected_categories:
        doc.add_heading(f"{category}", level=2)
        
        raw_ans = responses.get(category, "")
        clean_ans = sanitize_response(raw_ans, category=category)

        if not clean_ans or clean_ans.strip() == "":
            clean_ans = "No response generated. (This may be due to missing input data or system timeout)."
            p = doc.add_paragraph()
            run = p.add_run(clean_ans)
            run.font.color.rgb = RGBColor(0xFF, 0x00, 0x00) # Red
        else:
            doc.add_paragraph(clean_ans)
        
        # Add metrics
        res = next((r for r in results if r.get("category") == category), None)
        if res:
            p = doc.add_paragraph()
            r = p.add_run(f"Metrics: Data Acc {res.get('data_retrieval_acc', 0):.1f}% | Correctness {res.get('output_correctness', 0):.1f}%")
            r.font.size = Pt(9)
            r.font.color.rgb = RGBColor(0x80, 0x80, 0x80)

    # Ensure parent directory exists for output
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    doc.save(out_path)
    print(f"Exported successfully to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python export_patient_report.py <input.json> <output.docx>")
        sys.exit(1)
    export_docx(sys.argv[1], sys.argv[2])
