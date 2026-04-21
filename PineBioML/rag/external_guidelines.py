"""
External Guidelines RAG (Guard RAG)

Architecture: Knowledge-First + Web-Enrich
  1. match_guideline()            — instant lookup from embedded clinical_knowledge.py
  2. fetch_guideline_context()    — optional web search to enrich (DuckDuckGo)
  3. synthesize_guidelines()      — LLM synthesis with citations
  4. query_external_guidelines()  — main entry point

If embedded knowledge matches → returns a reliable, cited answer immediately.
Web search is used only to ENRICH, not as the primary source.
"""

import re
import time
import datetime
from typing import List, Dict, Tuple, Optional
from PineBioML.model.llm_factory import get_llm

from .clinical_knowledge import (
    CLINICAL_GUIDELINES,
    match_guideline,
    format_guideline_answer,
)

# ---------------------------------------------------------------------------
# GUIDELINE SOURCE REGISTRY — for web search enrichment
# ---------------------------------------------------------------------------

GUIDELINE_SOURCES: Dict[str, Dict] = {
    "ACG":      {"domain": "gi.org",                      "specialty": ["gi", "ibd", "colitis", "crohn"]},
    "ECCO":     {"domain": "ecco-ibd.eu",                 "specialty": ["gi", "ibd", "colitis", "crohn"]},
    "BSG":      {"domain": "bsg.org.uk",                  "specialty": ["gi", "ibd", "colonoscopy"]},
    "WGO":      {"domain": "worldgastroenterology.org",   "specialty": ["gi", "endoscopy"]},
    "WHO":      {"domain": "who.int",                     "specialty": ["general", "infectious", "global"]},
    "NICE":     {"domain": "nice.org.uk",                 "specialty": ["general", "all"]},
    "PubMed":   {"domain": "pubmed.ncbi.nlm.nih.gov",    "specialty": ["research", "all"]},
    "Cochrane": {"domain": "cochranelibrary.com",         "specialty": ["evidence", "all"]},
    "ESC":      {"domain": "escardio.org",                "specialty": ["cardiology", "heart", "ecg", "ejection fraction"]},
    "AHA":      {"domain": "heart.org",                   "specialty": ["cardiology", "heart failure", "stroke"]},
    "ACC":      {"domain": "acc.org",                     "specialty": ["cardiology", "coronary"]},
    "IDSA":     {"domain": "idsociety.org",               "specialty": ["infectious", "antibiotic", "sepsis", "hiv"]},
    "CDC":      {"domain": "cdc.gov",                     "specialty": ["infectious", "vaccine", "prevention"]},
    "ADA":      {"domain": "diabetes.org",                "specialty": ["diabetes", "hba1c", "insulin", "glucose"]},
    "ENDO":     {"domain": "endocrine.org",               "specialty": ["endocrine", "thyroid", "adrenal"]},
    "ASCO":     {"domain": "asco.org",                    "specialty": ["cancer", "oncology", "chemotherapy"]},
    "ESMO":     {"domain": "esmo.org",                    "specialty": ["cancer", "oncology", "tumor"]},
    "NCCN":     {"domain": "nccn.org",                    "specialty": ["cancer", "oncology", "staging"]},
    "GOLD":     {"domain": "goldcopd.org",                "specialty": ["copd", "respiratory", "spirometry"]},
    "ATS":      {"domain": "thoracic.org",                "specialty": ["respiratory", "pneumonia", "ards"]},
    "ERS":      {"domain": "ersnet.org",                  "specialty": ["respiratory", "lung", "asthma"]},
    "KDIGO":    {"domain": "kdigo.org",                   "specialty": ["kidney", "nephrology", "renal", "creatinine"]},
    "ASN":      {"domain": "asn-online.org",              "specialty": ["kidney", "dialysis", "glomerular"]},
    "EULAR":    {"domain": "eular.org",                   "specialty": ["rheumatology", "arthritis", "lupus", "spondylitis"]},
    "ACR":      {"domain": "rheumatology.org",            "specialty": ["rheumatology", "gout", "fibromyalgia"]},
    "AAN":      {"domain": "aan.com",                     "specialty": ["neurology", "stroke", "seizure", "dementia", "ms"]},
    "ACOG":     {"domain": "acog.org",                    "specialty": ["obstetrics", "gynecology", "pregnancy", "maternal"]},
    "SCCM":     {"domain": "sccm.org",                    "specialty": ["icu", "critical care", "sepsis", "ventilator"]},
    "ESICM":    {"domain": "esicm.org",                   "specialty": ["icu", "critical", "shock", "organ failure"]},
}

# Specialty keyword detection
SPECIALTY_KEYWORDS: Dict[str, List[str]] = {
    "gi":           ["colitis", "mes", "mayo", "ibd", "crohn", "uc", "ulcerative", "colonoscopy", "ileitis", "gi", "gastro", "rectum", "colon", "bowel", "intestin"],
    "cardiology":   ["ejection fraction", "ef", "heart", "cardiac", "ecg", "ekg", "mi", "stemi", "nstemi", "afib", "hf", "cardiomyopathy", "troponin"],
    "infectious":   ["sepsis", "antibiotic", "infection", "bacteremia", "covid", "pneumonia", "hiv", "fever"],
    "diabetes":     ["hba1c", "diabetes", "glucose", "insulin", "metformin", "hyperglycemia", "dm"],
    "cancer":       ["tumor", "cancer", "malignancy", "chemotherapy", "staging", "biopsy", "oncol"],
    "respiratory":  ["copd", "asthma", "spirometry", "fev1", "oxygen", "spo2", "dyspnea", "respiratory"],
    "kidney":       ["creatinine", "gfr", "egfr", "kidney", "renal", "dialysis", "proteinuria", "hematuria"],
    "rheumatology": ["arthritis", "lupus", "sle", "rheumatoid", "gout", "spondylitis", "vasculitis"],
    "neurology":    ["stroke", "seizure", "epilepsy", "parkinson", "dementia", "ms", "migraine", "neuropathy"],
    "obstetrics":   ["pregnant", "pregnancy", "maternal", "fetal", "delivery", "preeclampsia", "gestational"],
    "icu":          ["icu", "critical care", "ventilator", "shock", "organ failure", "intubation"],
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def detect_medical_domain(question: str) -> List[str]:
    """Classify a clinical question into medical specialties."""
    q_lower = question.lower()
    scores: Dict[str, int] = {}
    for specialty, keywords in SPECIALTY_KEYWORDS.items():
        hit = sum(1 for kw in keywords if kw in q_lower)
        if hit > 0:
            scores[specialty] = hit
    if not scores:
        return ["general"]
    return sorted(scores.keys(), key=lambda s: -scores[s])


def extract_patient_context(question: str) -> str:
    """Extract patient context (MES score, severity, metrics) from the question text."""
    q_lower = question.lower()
    ctx_parts = []

    # MES score
    mes_match = re.search(r'mes\s*(\d)', q_lower)
    if mes_match:
        score = mes_match.group(1)
        severity_map = {"0": "Remission", "1": "Mild", "2": "Moderate", "3": "Severe"}
        ctx_parts.append(f"MES {score} ({severity_map.get(score, '')})")

    # pMayo
    pmayo_match = re.search(r'(?:p?mayo|pmayo)\s*(?:score)?\s*(\d+)', q_lower)
    if pmayo_match:
        ctx_parts.append(f"pMayo {pmayo_match.group(1)}")

    # Severity keywords
    for sev in ["severe", "moderate", "mild", "remission", "fulminant", "acute"]:
        if sev in q_lower and sev not in " ".join(ctx_parts).lower():
            ctx_parts.append(sev.capitalize())

    # HbA1c
    hba1c_match = re.search(r'hba1c\s*(\d+\.?\d*)', q_lower)
    if hba1c_match:
        ctx_parts.append(f"HbA1c {hba1c_match.group(1)}")

    # EF
    ef_match = re.search(r'(?:ef|ejection fraction)\s*(\d+)', q_lower)
    if ef_match:
        ctx_parts.append(f"EF {ef_match.group(1)}%")

    return ", ".join(ctx_parts) if ctx_parts else ""


def get_priority_domains(specialties: List[str], max_domains: int = 6) -> List[Dict]:
    """Given detected specialties, return the top guideline sources to search."""
    priority = []
    seen = set()
    for specialty in specialties:
        for name, info in GUIDELINE_SOURCES.items():
            if name in seen:
                continue
            if any(specialty in s for s in info["specialty"]):
                priority.append({"name": name, **info})
                seen.add(name)
    for name in ["PubMed", "WHO", "NICE"]:
        if name not in seen:
            priority.append({"name": name, **GUIDELINE_SOURCES[name]})
            seen.add(name)
    return priority[:max_domains]


def scrape_page(url: str, max_chars: int = 3000, timeout: int = 8) -> Tuple[str, str]:
    """Scrape text content from a URL. Returns (text_content, status)."""
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0 (compatible; PineBioML-GuardRAG/1.0)"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code in [401, 403, 402]:
            return "", "paywalled"
        if resp.status_code != 200:
            return "", f"error_{resp.status_code}"
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "button"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.find(id="content") or soup.find(class_="content")
        text = (main or soup).get_text(separator=" ", strip=True)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_chars], "ok"
    except Exception as e:
        return "", f"error: {str(e)}"


def fetch_web_guidelines(question: str, patient_context: str = "", max_results: int = 3) -> List[Dict]:
    """
    Web search enrichment (best-effort, not primary source).
    Searches DuckDuckGo for guideline content across priority domains.
    """
    results = []
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        specialties = detect_medical_domain(question)
        priority_domains = get_priority_domains(specialties, max_domains=4)

        clinical_terms = re.sub(r'[^a-zA-Z0-9 \-]', '', question[:200]).strip()
        query = f"{clinical_terms} guidelines protocol".strip()

        fetched_urls = set()
        ddg = DDGS()

        for source_info in priority_domains:
            domain = source_info["domain"]
            source_name = source_info["name"]
            site_query = f"site:{domain} {query}"
            try:
                search_results = list(ddg.text(site_query, max_results=2))
                time.sleep(0.3)
                for r in search_results:
                    url = r.get("href", "")
                    if not url or url in fetched_urls:
                        continue
                    fetched_urls.add(url)
                    content, status = scrape_page(url)
                    results.append({
                        "source_name": source_name,
                        "url": url,
                        "title": r.get("title", ""),
                        "snippet": r.get("body", "")[:300],
                        "content": content,
                        "status": status
                    })
                    if len(results) >= max_results:
                        break
            except Exception:
                pass
            if len(results) >= max_results:
                break
    except Exception:
        pass  # Web search failure is non-fatal
    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def query_external_guidelines(
    question: str,
    patient_context: str = "",
    max_results: int = 3
) -> str:
    """
    Guard RAG main entry point.
    
    Flow:
    1. Extract patient context from question if not provided
    2. Match against embedded clinical knowledge base (PRIMARY SOURCE)
    3. If no matches found in KB, perform web search (ENRICHMENT)
    4. Synthesize final answer with proper citations
    """
    timestamp = datetime.datetime.now().isoformat()
    print(f"[{timestamp}] [GuardRAG] Processing: {question[:100]}...")

    # 1. Extract patient context from question text if not provided
    if not patient_context:
        patient_context = extract_patient_context(question)
    
    # 2. Match embedded clinical knowledge (PRIMARY SOURCE)
    kb_matches = match_guideline(question, patient_context)
    kb_answer = format_guideline_answer(kb_matches, question)

    if kb_matches:
        print(f"[GuardRAG] Internal KB Match: {len(kb_matches)} found.")
        return kb_answer
    
    # 3. Only trigger web search if internal KB is empty
    print(f"[GuardRAG] No internal KB match. Triggering web enrichment...")
    web_results = fetch_web_guidelines(question, patient_context, max_results=max_results)
    
    if not web_results:
        return "No internal SOP or reliable external guideline found for this specific query. I am restricted from providing unverified recommendations."

    # 4. Synthesize from web results
    web_context = "\n\n".join([
        f"Source: {r['source_name']} ({r['url']})\nTitle: {r['title']}\nContent: {r['content'] or r['snippet']}"
        for r in web_results
    ])
    
    return _synthesize_web_only(question, web_context, patient_context)

def _synthesize_combined(question: str, kb_answer: str, web_context: str, patient_context: str) -> str:
    """Combine embedded knowledge (primary) with web enrichment."""
    try:
        llm = get_llm(model_name="gpt-4o-mini", temperature=0.1)

        system = (
            "You are a Clinical Decision Support AI. You have been given a PRIMARY guideline answer "
            "from our embedded knowledge base, plus SUPPLEMENTARY web content. "
            "Your job is to enhance the primary answer with any additional relevant details from the web, "
            "while keeping the primary answer's citations and structure intact. "
            "Keep citations explicit (e.g., 'According to ACG Clinical Guidelines...'). "
            "Mirror the user's language. Be concise and actionable."
        )
        user = (
            f"Question: {question}\n"
            f"Patient Context: {patient_context}\n\n"
            f"PRIMARY GUIDELINE ANSWER:\n{kb_answer}\n\n"
            f"SUPPLEMENTARY WEB CONTENT:\n{web_context[:3000]}\n\n"
            f"Provide the final clinical recommendation. Start with the primary answer, "
            f"add any new relevant details from web content. Keep source citations."
        )
        response = llm.invoke([("system", system), ("human", user)])
        return response.content
    except Exception as e:
        # Fallback to KB answer alone if LLM fails
        return kb_answer


def _synthesize_web_only(question: str, web_context: str, patient_context: str) -> str:
    """Synthesize from web content only (no embedded KB match)."""
    try:
        llm = get_llm(model_name="gpt-4o-mini", temperature=0.1)

        system = (
            "You are a Senior Clinical Decision Support AI. Synthesize medical guideline content "
            "into an actionable clinical recommendation. ALWAYS cite which guideline source. "
            "Use the format: 'According to [SOURCE] Guidelines: [RECOMMENDATION]'. "
            "Mirror the user's language. Be concise."
        )
        user = (
            f"Clinical Question: {question}\n"
            f"Patient Context: {patient_context}\n\n"
            f"Retrieved Guideline Content:\n{web_context[:5000]}\n\n"
            f"Provide a clinical recommendation with citations."
        )
        response = llm.invoke([("system", system), ("human", user)])
        return response.content
    except Exception as e:
        return f"⚠️ Could not synthesize guidelines: {e}"


def _synthesize_fallback(question: str, patient_context: str) -> str:
    """LLM general medical knowledge when no specific sources found."""
    try:
        llm = get_llm(model_name="gpt-4o-mini", temperature=0.1)

        system = (
            "You are a Senior Clinical Decision Support AI with extensive knowledge of "
            "ACG, ECCO, WHO, NICE, ESC, ADA, ASCO, IDSA, and other major medical guidelines. "
            "Answer based on your training knowledge of these guidelines. "
            "ALWAYS cite the specific guideline source. "
            "Use the format: 'According to [SOURCE] Clinical Guidelines: [RECOMMENDATION]'. "
            "Mirror the user's language."
        )
        user = (
            f"Clinical Question: {question}\n"
            f"Patient Context: {patient_context}\n\n"
            f"Provide an evidence-based clinical recommendation citing the relevant guidelines."
        )
        response = llm.invoke([("system", system), ("human", user)])
        answer = response.content

        # Add disclaimer
        answer += (
            "\n\n> ⚠️ *Note: This recommendation is based on the AI's training knowledge of published guidelines. "
            "For the latest updates, please consult the original guideline publications directly.*"
        )
        return answer
    except Exception as e:
        return f"⚠️ Could not generate guideline recommendation: {e}"
