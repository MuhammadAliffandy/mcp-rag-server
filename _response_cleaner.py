import re as _re

def _clean_response_for_client(text: str) -> str:
    """
    Post-process LLM output before showing to clients:
    - Convert Python dict notation {mes_a: 1.0} to readable table rows
    - Clean up raw field names (mes_a -> Ascending, etc.)
    - Remove trailing .0 from whole numbers
    - Convert Python True/False to Yes/No
    """
    if not text:
        return text

    SEGMENT_MAP = {
        "mes_a": "Ascending (A)", "mes_t": "Transverse (T)",
        "mes_d": "Descending (D)", "mes_s": "Sigmoid (S)", "mes_r": "Rectum (R)",
        "nancy_a": "Ascending (A)", "nancy_t": "Transverse (T)",
        "nancy_d": "Descending (D)", "nancy_s": "Sigmoid (S)", "nancy_r": "Rectum (R)",
    }
    FIELD_MAP = {
        "bl_mayo_total": "Partial Mayo Score",
        "bl_mayo_s": "Stool Frequency sub-score",
        "bl_mayo_b": "Rectal Bleeding sub-score",
        "bl_mayo_p": "Physician Assessment sub-score",
        "max_mes": "MES Max",
        "max_nancy": "Nancy Max",
    }

    def _dict_to_table(m):
        inner = m.group(1)
        pairs = _re.findall(r"['\"]?(\w+)['\"]?\s*:\s*([\d.]+)", inner)
        if not pairs:
            return m.group(0)
        parts = []
        for k, v in pairs:
            label = SEGMENT_MAP.get(k, k)
            try:
                fval = float(v)
                val_str = str(int(fval)) if fval == int(fval) else v
            except Exception:
                val_str = v
            parts.append(f"{label}: {val_str}")
        return " | ".join(parts)

    # Convert dict notation to pipe-separated table
    text = _re.sub(r"\{([^{}]+)\}", _dict_to_table, text)

    # Replace standalone raw field names
    for k, v in {**SEGMENT_MAP, **FIELD_MAP}.items():
        text = text.replace(k, v)

    # Remove trailing .0 on whole numbers (e.g. "3.0" -> "3", "4.0" -> "4")
    text = _re.sub(r"\b(\d+)\.0\b", r"\1", text)

    # Python booleans -> human readable
    text = text.replace(": True)", ": Yes)").replace(": False)", ": No)")
    text = text.replace(": True\n", ": Yes\n").replace(": False\n", ": No\n")
    text = text.replace("=True", "= Yes").replace("=False", "= No")

    return text
