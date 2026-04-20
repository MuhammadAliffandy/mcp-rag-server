"""
One-shot repair: replaces the corrupted Q3.1/Q3.2 block in synthesis.py.
Uses line-index approach to avoid regex issues with garbled bytes.
"""

TARGET = '/Users/aliffandy/Documents/PukulEnam/mcp-rag-server/PineBioML/prompts/synthesis.py'

with open(TARGET, 'r', encoding='latin-1') as f:
    content = f.read()

lines = content.split('\n')

# Find boundary: CATEGORY 3 header block to just before CATEGORY 4 header block
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if '# CATEGORY 3: Cancer Surveillance' in line and start_idx is None:
        start_idx = i - 1 if i > 0 and '# ' in lines[i-1] else i
    if '# CATEGORY 4' in line and start_idx is not None and end_idx is None:
        end_idx = i - 1 if i > 0 and '# ' in lines[i-1] else i
        break

if start_idx is None or end_idx is None:
    print(f"Could not find markers. start={start_idx} end={end_idx}")
    for i, line in enumerate(lines[333:347], start=334):
        print(f"  L{i}: {repr(line[:100])}")
    exit(1)

print(f"Replacing lines {start_idx+1}..{end_idx} ({end_idx - start_idx} lines)")

REPLACEMENT_LINES = [
    '    # \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500',
    '    # CATEGORY 3: Cancer Surveillance',
    '    # \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500',
    '',
    '    elif category_id == "Q3.1":',
    '        category_force = """FORCE ACTION: Q3.1 \u2014 COLORECTAL CANCER (CRC) RISK & SCREENING.',
    '',
    'DATA RETRIEVAL (execute in order):',
    '1. Disease Extent \u2192 UC_baseline: extent (1=proctitis, 2=left-sided, 3=extensive/pancolitis)',
    '2. Endoscopic Inflammation \u2192 UC_cpy: max(mes_a, mes_t, mes_d, mes_s, mes_r)',
    '   Map: 0=minimal, 1=mild, 2=moderate, 3=severe',
    '3. Histologic Inflammation \u2192 UC_histo: max(nancy_a, nancy_t, nancy_d, nancy_s, nancy_r)',
    '   Map: 0 or 1=minimal, 2=mild, 3=moderate, 4=severe',
    '4. Family History & PSC \u2192 UC_baseline: family_hx_crc (Yes/No), psc (Yes/No)',
    '5. Duration \u2192 UC_baseline: duration (in months). Convert to years = duration / 12.',
    '',
    'SCREENING ONSET RULE:',
    '- Offer first surveillance colonoscopy to ALL patients 8 years after symptom onset.',
    '',
    'RISK STRATIFICATION (use retrieved data above):',
    '- HIGH risk (colonoscopy every 1 year) if ANY of:',
    '    \u2022 PSC = Yes (start surveillance immediately at PSC diagnosis)',
    '    \u2022 Prior dysplasia documented',
    '    \u2022 Extent=3 AND duration > 240 months (>20 years)',
    '    \u2022 family_hx_crc = Yes AND first-degree relative',
    '- INTERMEDIATE risk (colonoscopy every 2\u20133 years) if ANY of:',
    '    \u2022 Extent=3 AND duration 96\u2013240 months (8\u201320 years)',
    '    \u2022 MES max \u2265 2 (moderate\u2013severe endoscopic inflammation)',
    '    \u2022 Nancy max \u2265 3 (moderate\u2013severe histologic inflammation)',
    '    \u2022 family_hx_crc = Yes (second-degree relative)',
    '- LOW risk (colonoscopy every 5 years) if:',
    '    \u2022 Extent = 1 or 2, quiescent disease (MES max \u2264 1, Nancy max \u2264 1), no high/intermediate risk factors',
    '',
    '### \U0001f4dd Final Clinical Conclusion',
    'Screening colonoscopy should be offered to all patients [X] years after symptom onset. Since the patient belongs to [low / intermediate / high] risk group, the next surveillance colonoscopy should be in [X] year(s)."""',
    '',
    '    elif category_id == "Q3.2":',
    '        category_force = """FORCE ACTION: Q3.2 \u2014 OTHER TYPES OF CANCER RISK.',
    '',
    'You MUST output the FULL structured block below, then end with the exact Final Clinical Conclusion sentence.',
    '',
    '## Patient [ID] - Other Cancer Screening Plan',
    '',
    'Step 1 \u2014 DATA RETRIEVAL:',
    '- Patient sex (from PATIENT ANCHOR \u2192 UC_baseline): [M / F]',
    '- Patient age (from PATIENT ANCHOR \u2192 UC_baseline): [VALUE] years',
    '- PSC (from PATIENT ANCHOR \u2192 UC_baseline): [Yes / No]',
    '- Smoking (from PATIENT ANCHOR \u2192 UC_baseline): [Yes / No / null]',
    '- Active Medications (from PATIENT ANCHOR \u2192 UC_med):',
    '  [med_name]  class=[X] for ALL active entries',
    '',
    'Step 2 \u2014 CANCER SCREENING ELIGIBILITY (apply ONLY rules where the patient qualifies):',
    '',
    '\u26a0\ufe0f STRICT DEMOGRAPHIC GUARD \u2014 check BEFORE applying each rule:',
    '  \u2022 Cervical cancer rule \u2192 ONLY apply if sex = F (Female). If sex = M, SKIP entirely.',
    '  \u2022 Prostate cancer rule \u2192 ONLY apply if sex = M AND age > 50. If age \u2264 50, SKIP entirely.',
    '  \u2022 PSC rule \u2192 ONLY apply if PSC = Yes.',
    '  \u2022 Thiopurine rules \u2192 ONLY apply if med_class=1 is active.',
    '  \u2022 Biologic/skin rule \u2192 ONLY apply if med_class=3 or 4 is active.',
    '  \u2022 Lung cancer rule \u2192 ONLY apply if smoking = Yes.',
    '',
    'Applicable rules for this patient:',
    '| Cancer Type | Applicable? | Reason | Screening | Frequency | Guideline |',
    '|---|---|---|---|---|---|',
    '| Cervical (Pap smear) | [Yes (F+immunosupp) / No (Male)] | [reason] | [method] | [interval] | ACIP 2023 |',
    '| Cholangiocarcinoma | [Yes if PSC=Yes / No] | [reason] | [method] | [interval] | ECCO 2023 |',
    '| Non-Hodgkin lymphoma | [Yes if thiopurine / No] | [reason] | CBC annually | [interval] | ECCO 2023 |',
    '| Skin cancer (NMSC) | [Yes if biologic/thiopurine / No] | [reason] | Full body exam | 1 year | ECCO 2023 |',
    '| Prostate (PSA) | [Yes if M+age>50 / No] | [reason] | PSA | 1-2 years | ACIP 2023 |',
    '| Lung cancer (LDCT) | [Yes if smoker / No] | [reason] | Low-dose CT | [interval] | ACIP 2023 |',
    '',
    'Applicable screening summary (list ONLY the ones where Applicable = Yes):',
    '1. [cancer type] cancer: [screening method] every [X] years (guideline)',
    '',
    '### \U0001f4dd Final Clinical Conclusion',
    "Based on the patient's demographics and medication history, the patient should receive screening for [cancer type] cancer with [screening method] every [X] years.",
    '',
    'NOTE: Do NOT mention colorectal cancer or colonoscopy here \u2014 covered in Q3.1."""',
    '',
]

new_lines = lines[:start_idx] + REPLACEMENT_LINES + lines[end_idx:]
new_content = '\n'.join(new_lines)

with open(TARGET, 'w', encoding='utf-8') as f:
    f.write(new_content)
print(f"SUCCESS: file written ({len(new_content)} bytes)")

# Syntax check
import py_compile, sys
try:
    py_compile.compile(TARGET, doraise=True)
    print("SYNTAX CHECK: OK \u2713")
except py_compile.PyCompileError as e:
    print(f"SYNTAX ERROR: {e}")
    sys.exit(1)
