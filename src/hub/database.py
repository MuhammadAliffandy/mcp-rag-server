"""
EHR Database Abstraction Layer
Supports SQL (SQLite) and CSV backends for MIMIC-IV style patient record lookups.
"""

import sqlite3
import pandas as pd
import os
from typing import List, Dict, Any, Optional

class EHRDatabase:
    """
    Unified interface for retrieving patient clinical records.
    Mimics the SQLite search pattern used in EXPRAG original repository.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.is_csv = db_path.endswith('.csv')
        self._df = None
        
        if not os.path.exists(db_path):
            print(f"⚠️ Warning: Database path does not exist: {db_path}")
            
        if self.is_csv:
            self._load_csv()
        else:
            self._connect_sql()

    def _connect_sql(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            # Ensure tables exist (Simplified MIMIC-IV schema)
            cursor = self.conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS patient_records (hadm_id TEXT PRIMARY KEY, text TEXT)")
            self.conn.commit()
        except Exception as e:
            print(f"❌ SQL Connection Error: {e}")

    def _load_csv(self):
        try:
            self._df = pd.read_csv(self.db_path, dtype={'hadm_id': str, 'case_id': str})
        except Exception as e:
            print(f"❌ CSV Loading Error: {e}")

    def get_full_record(self, case_id: str) -> Optional[str]:
        """Retrieve the full clinical text (e.g. discharge summary) for a patient."""
        if self.is_csv:
            if self._df is None: return None
            # Support both common ID names
            id_col = 'hadm_id' if 'hadm_id' in self._df.columns else 'case_id'
            row = self._df[self._df[id_col] == str(case_id)]
            if not row.empty:
                return str(row.iloc[0]['text'])
        else:
            cursor = self.conn.cursor()
            cursor.execute("SELECT text FROM patient_records WHERE hadm_id = ?", (case_id,))
            result = cursor.fetchone()
            return result[0] if result else None
        return None

    def save_record(self, case_id: str, text: str):
        """Save or update a patient record."""
        if self.is_csv:
            print("⚠️ Save operation not supported for CSV backend in read-only mode.")
        else:
            cursor = self.conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO patient_records (hadm_id, text) VALUES (?, ?)", (case_id, text))
            self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()
