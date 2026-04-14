import pandas as pd
import os
from datetime import datetime

os.makedirs('internal_docs', exist_ok=True)
FILE_PATH = 'internal_docs/4DEADFE0FD06EA10E459256A2E85237AB43BD9EB_UC_20260304(follow_up_20260211)_long.xlsx'

with pd.ExcelWriter(FILE_PATH) as writer:
    # 1. UC_baseline
    df_baseline = pd.DataFrame([{
        'id': 1,
        'bl_mayo_total': 4,
        'bl_mayo_s': 2,
        'bl_mayo_b': 1,
        'bl_mayo_p': 1,
        'date_onset': '2023-01-15',
        'birthday': '1990-05-12',
        'extent': 3,
        'psc': 0,
        'family_hx_crc': 0,
        'sex': 'M',
        'age': 36
    }])
    # Account for QA header mapping: UC_baseline uses header=1, so add a dummy row first
    pd.DataFrame(columns=df_baseline.columns).to_excel(writer, sheet_name='UC_baseline', index=False)
    df_baseline.to_excel(writer, sheet_name='UC_baseline', index=False, startrow=1)

    # 2. UC_cpy
    df_cpy = pd.DataFrame([{
        'id': 1,
        'date_cpy': '2025-10-15',
        'mes_a': 1,
        'mes_t': 2,
        'mes_d': 3,
        'mes_s': 2,
        'mes_r': 1
    }])
    
    # UC_cpy uses header=0
    df_cpy.to_excel(writer, sheet_name='UC_cpy', index=False)

    # 3. UC_lab
    df_lab = pd.DataFrame([
        {'id': 1, 'lab_date': '2026-01-10', 'lab_item': 'crp', 'lab_value': 5.8},
        {'id': 1, 'lab_date': '2026-01-10', 'lab_item': 'fc', 'lab_value': 310},
        {'id': 1, 'lab_date': '2026-01-10', 'lab_item': 'alb', 'lab_value': 3.8}
    ])
    df_lab.to_excel(writer, sheet_name='UC_lab', index=False)

    # 4. UC_histo
    df_histo = pd.DataFrame([{
        'id': 1,
        'date_cpy': '2025-10-15',
        'nancy_a': 0,
        'nancy_t': 1,
        'nancy_d': 2,
        'nancy_s': 1,
        'nancy_r': 1
    }])
    df_histo.to_excel(writer, sheet_name='UC_histo', index=False)

    # 5. UC_med
    df_med = pd.DataFrame([{
        'id': 1,
        'med_name': 'Infliximab',
        'med_class': 3,
        'route': 'IV',
        'dose': '5mg/kg',
        'interval': '8 wks',
        'start_date': '2025-08-01',
        'end_date': None
    }])
    # UC_med uses header=1
    pd.DataFrame(columns=df_med.columns).to_excel(writer, sheet_name='UC_med', index=False)
    df_med.to_excel(writer, sheet_name='UC_med', index=False, startrow=1)

print('Mock data generated successfully at:', FILE_PATH)

