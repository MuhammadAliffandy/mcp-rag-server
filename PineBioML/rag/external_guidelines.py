"""
External Guidelines RAG (Guard RAG)

Fetches live medical guidelines from trusted authorities across ALL specialties,
synthesizes a cited clinical answer using GPT-4o-mini.

Architecture:
  1. detect_medical_domain()  — classify question into specialty
  2. build_search_query()     — build optimized search string
  3. fetch_guideline_context() — DuckDuckGo search + scrape pages
  4. synthesize_guidelines()  — LLM synthesis with citations
"""

import re
import time
import datetime
from typing import List, Dict, Tuple, Optional
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# GUIDELINE SOURCE REGISTRY — All authoritative medical guideline domains
# ---------------------------------------------------------------------------

GUIDELINE_SOURCES: Dict[str, Dict] = {
    # GI / IBD
    "ACG":      {"domain": "gi.org",                      "specialty": ["gi", "ibd", "colitis", "crohn"]},
    "ECCO":     {"domain": "ecco-ibd.eu",                 "specialty": ["gi", "ibd", "colitis", "crohn"]},
    "BSG":      {"domain": "bsg.org.uk",                  "specialty": ["gi", "ibd", "colonoscopy"]},
    "WGO":      {"domain": "worldgastroenterology.org",   "specialty": ["gi", "endoscopy"]},

    # General / Multi-specialty
    "WHO":      {"domain": "who.int",                     "specialty": ["general", "infectious", "global"]},
    "NICE":     {"domain": "nice.org.uk",                 "specialty": ["general", "all"]},
    "PubMed":   {"domain": "pubmed.ncbi.nlm.nih.gov",    "specialty": ["research", "all"]},
    "Cochrane": {"domain": "cochranelibrary.com",         "specialty": ["evidence", "all"]},

    # Cardiology
    "ESC":      {"domain": "escardio.org",                "specialty": ["cardiology", "heart", "ecg", "ejection fraction"]},
    "AHA":      {"domain": "heart.org",                   "specialty": ["cardiology", "heart failure", "stroke"]},
    "ACC":      {"domain": "acc.org",                     "specialty": ["cardiology", "coronary"]},

    # Infectious Disease
    "IDSA":     {"domain": "idsociety.org",               "specialty": ["infectious", "antibiotic", "sepsis", "hiv"]},
    "CDC":      {"domain": "cdc.gov",                     "specialty": ["infectious", "vaccine", "prevention"]},

    # Diabetes / Endocrinology
    "ADA":      {"domain": "diabetes.org",                "specialty": ["diabetes", "hba1c", "insulin", "glucose"]},
    "ENDO":     {"domain": "endocrine.org",               "specialty": ["endocrine", "thyroid", "adrenal"]},

    # Oncology
    "ASCO":     {"domain": "asco.org",                    "specialty": ["cancer", "oncology", "chemotherapy"]},
    "ESMO":     {"domain": "esmo.org",                    "specialty": ["cancer", "oncology", "tumor"]},
    "NCCN":     {"domain": "nccn.org",                    "specialty": ["cancer", "oncology", "staging"]},

    # Respiratory
    "GOLD":     {"domain": "goldcopd.org",                "specialty": ["copd", "respiratory", "spirometry"]},
    "ATS":      {"domain": "thoracic.org",                "specialty": ["respiratory", "pneumonia", "ards"]},
    "ERS":      {"domain": "ersnet.org",                  "specialty": ["respiratory", "lung", "asthma"]},

    # Nephrology
    "KDIGO":    {"domain": "kdigo.org",                   "specialty": ["kidney", "nephrology", "renal", "creatinine"]},
    "ASN":      {"domain": "asn-online.org",              "specialty": ["kidney", "dialysis", "glomerular"]},

    # Rheumatology
    "EULAR":    {"domain": "eular.org",                   "specialty": ["rheumatology", "arthritis", "lupus", "spondylitis"]},
    "ACR":      {"domain": "rheumatology.org",            "specialty": ["rheumatology", "gout", "fibromyalgia"]},

    # Neurology
    "AAN":      {"domain": "aan.com",                     "specialty": ["neurology", "stroke", "seizure", "dementia", "ms"]},
    "ENS":      {"domain": "ens.org",                     "specialty": ["neurology", "parkinson", "headache"]},

    # OB/GYN
    "ACOG":     {"domain": "acog.org",                    "specialty": ["obstetrics", "gynecology", "pregnancy", "maternal"]},
    "FIGO":     {"domain": "figo.org",                    "specialty": ["obstetrics", "fertilization", "maternal"]},

    # Surgery
    "SAGES":    {"domain": "sages.org",                   "specialty": ["surgery", "laparoscopy", "endoscopy", "bariatric"]},

    # Critical Care / ICU
    "SCCM":     {"domain": "sccm.org",                    "specialty": ["icu", "critical care", "sepsis", "ventilator"]},
    "ESICM":    {"domain": "esicm.org",                   "specialty": ["icu", "critical", "shock", "organ failure"]},
}

# Domain keywords for auto-detecting specialty from a clinical question
SPECIALTY_KEYWORDS: Dict[str, List[str]] = {
    "gi":           ["colitis", "mes", "mayo", "ibd", "crohn", "uc", "ulcerative", "colonoscopy", "ileitis", "gi", "gastro", "rectum", "colon", "bowel", "intestin"],
    "cardiology":   ["ejection fraction", "ef", "heart", "cardiac", "ecg", "ekg", "mi", "stemi", "nstemi", "afib", "hf", "cardiomyopathy", "troponin"],
    "infectious":   ["sepsis", "antibiotic", "infection", "bacteremia", "covid", "pneumonia", "hiv", "fever", "crp infect"],
    "diabetes":     ["hba1c", "diabetes", "glucose", "insulin", "metformin", "hyperglycemia", "dm"],
    "cancer":       ["tumor", "cancer", "malignancy", "chemotherapy", "staging", "biopsy", "oncol"],
    "respiratory":  ["copd", "asthma", "spirometry", "fev1", "oxygen", "spo2", "dyspnea", "respiratory"],
    "kidney":       ["creatinine", "gfr", "egfr", "kidney", "renal", "dialysis", "proteinuria", "hematuria"],
    "rheumatology": ["arthritis", "lupus", "sle", "rheumatoid", "gout", "spondylitis", "vasculitis"],
    "neurology":    ["stroke", "seizure", "epilepsy", "parkinson", "dementia", "ms", "migraine", "neuropathy"],
    "obstetrics":   ["pregnant", "pregnancy", "maternal", "fetal", "delivery", "preeclampsia", "gestational"],
    "icu":          ["icu", "critical care", "ventilator", "shock", "organ failure", "intubation"],
}


def detect_medical_domain(question: str) -> List[str]:
    """
    Classify a clinical question into medical specialties.
    Returns a list of detected specialties, ordered by confidence.
    """
    q_lower = question.lower()
    scores: Dict[str, int] = {}

    for specialty, keywords in SPECIALTY_KEYWORDS.items():
        hit = sum(1 for kw in keywords if kw in q_lower)
        if hit > 0:
            scores[specialty] = hit

    if not scores:
        return ["general"]  # Default fallback

    # Sort by score descending
    return sorted(scores.keys(), key=lambda s: -scores[s])


def get_priority_domains(specialties: List[str], max_domains: int = 6) -> List[Dict]:
    """
    Given detected specialties, return the top guideline sources to search.
    Always includes PubMed and WHO as baseline.
    """
    priority = []
    seen = set()

    for specialty in specialties:
        for name, info in GUIDELINE_SOURCES.items():
            if name in seen:
                continue
            if any(specialty in s for s in info["specialty"]):
                priority.append({"name": name, **info})
                seen.add(name)

    # Always include general fallbacks
    for name in ["PubMed", "WHO", "NICE"]:
        if name not in seen:
            priority.append({"name": name, **GUIDELINE_SOURCES[name]})
            seen.add(name)

    return priority[:max_domains]


def build_search_query(question: str, patient_context: str, specialties: List[str]) -> str:
    """
    Build an optimized DuckDuckGo query for medical guideline retrieval.
    """
    # Extract key clinical terms from question
    clinical_terms = re.sub(r'[^a-zA-Z0-9 \-]', '', question[:200]).strip()

    # Add severity terms if patient context present
    context_terms = ""
    if patient_context:
        ctx_lower = patient_context.lower()
        if any(t in ctx_lower for t in ["mes 3", "severe", "pMayo", "high"]):
            context_terms = "severe treatment escalation"
        elif any(t in ctx_lower for t in ["mes 2", "moderate", "moderate-severe"]):
            context_terms = "moderate treatment guidelines"
        elif any(t in ctx_lower for t in ["mes 1", "mes 0", "mild", "remission"]):
            context_terms = "mild remission induction"

    specialty_hint = specialties[0] if specialties else "clinical"
    query = f"{clinical_terms} {context_terms} guidelines protocol {specialty_hint}".strip()
    return query[:400]  # DuckDuckGo query length limit


def scrape_page(url: str, max_chars: int = 3000, timeout: int = 8) -> Tuple[str, str]:
    """
    Scrape text content from a URL.
    Returns (text_content, status) where status is 'ok' | 'paywalled' | 'error'.
    """
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

        # Remove navigation, scripts, styles, ads
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "button"]):
            tag.decompose()

        # Try to get main content areas first
        main = soup.find("main") or soup.find("article") or soup.find(id="content") or soup.find(class_="content")
        text = (main or soup).get_text(separator=" ", strip=True)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text[:max_chars], "ok"

    except Exception as e:
        return "", f"error: {str(e)}"


def fetch_guideline_context(
    question: str,
    patient_context: str = "",
    max_results: int = 5
) -> List[Dict]:
    """
    Main fetch pipeline:
    1. Detect specialty
    2. Build search query
    3. Search DuckDuckGo across priority domains
    4. Scrape top pages
    Returns list of {source_name, url, content, status}
    """
    results = []

    try:
        try:
            from ddgs import DDGS  # New package name
        except ImportError:
            from duckduckgo_search import DDGS  # Legacy fallback

        # 1. Detect specialty
        specialties = detect_medical_domain(question)
        priority_domains = get_priority_domains(specialties, max_domains=8)

        # 2. Build query
        query = build_search_query(question, patient_context, specialties)

        # 3. Search each priority domain
        fetched_urls = set()
        ddg = DDGS()

        for source_info in priority_domains:
            domain = source_info["domain"]
            source_name = source_info["name"]
            site_query = f"site:{domain} {query}"

            try:
                # DuckDuckGo search
                search_results = list(ddg.text(site_query, max_results=2))
                time.sleep(0.3)  # Rate limiting

                for r in search_results:
                    url = r.get("href", "")
                    if not url or url in fetched_urls:
                        continue

                    fetched_urls.add(url)

                    # Scrape the page
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

            except Exception as e:
                # Skip this domain silently (rate limit / no results)
                pass

            if len(results) >= max_results:
                break

    except ImportError:
        return [{
            "source_name": "Error",
            "url": "",
            "content": "duckduckgo-search library not installed. Run: pip install duckduckgo-search",
            "status": "error"
        }]
    except Exception as e:
        return [{"source_name": "Error", "url": "", "content": str(e), "status": "error"}]

    return results


def synthesize_guidelines(
    question: str,
    fetched_results: List[Dict],
    patient_context: str = ""
) -> str:
    """
    Uses GPT-4o-mini to synthesize a cited clinical answer from scraped guideline content.
    """
    try:
        # Build the context block from fetched results
        context_blocks = []
        used_sources = []

        for r in fetched_results:
            if r.get("status") == "ok" and r.get("content"):
                block = f"[{r['source_name']}] ({r['url']})\n{r['content']}"
                context_blocks.append(block)
                used_sources.append(f"- **{r['source_name']}**: {r['url']}")
            elif r.get("snippet"):
                block = f"[{r['source_name']}] ({r['url']}) — SNIPPET ONLY:\n{r['snippet']}"
                context_blocks.append(block)
                used_sources.append(f"- **{r['source_name']}** (snippet): {r['url']}")

        if not context_blocks:
            return "⚠️ No guideline content could be retrieved from external sources. Please check your internet connection or try a more specific clinical question."

        guideline_context = "\n\n---\n\n".join(context_blocks)
        sources_list = "\n".join(used_sources)

        patient_block = f"\n**Current Patient Context:** {patient_context}" if patient_context else ""

        system_prompt = """You are a Senior Clinical Decision Support AI. Your role is to synthesize medical guideline content and provide actionable, evidence-based clinical recommendations with clear source citations.

CRITICAL RULES:
1. ALWAYS cite which guideline your recommendation comes from (e.g., "Per ACG Guidelines...", "According to ECCO 2023...")
2. If multiple guidelines agree, note the consensus
3. If guidelines differ, note the discrepancy clearly
4. Structure your answer: Recommendation → Rationale → Source
5. Use markdown formatting (bold for drug names, bullet points for steps)
6. Be concise but complete. A doctor needs ACTION not theory.
7. Mirror the user's language (Indonesian if they asked in Indonesian, English otherwise)"""

        user_prompt = f"""Clinical Question: {question}
{patient_block}

Retrieved Guideline Content:
{guideline_context[:8000]}

Based ONLY on the above guideline content, provide a clinical recommendation with citations.
End your response with a "📚 Sources Consulted:" section listing the sources used.
"""

        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
        response = llm.invoke([("system", system_prompt), ("human", user_prompt)])
        answer = response.content

        # Append source list if not already present
        if "Sources Consulted" not in answer and used_sources:
            answer += f"\n\n📚 **Sources Consulted:**\n{sources_list}"

        return answer

    except Exception as e:
        return f"❌ Synthesis error: {e}"


def query_external_guidelines(
    question: str,
    patient_context: str = "",
    max_results: int = 5
) -> str:
    """
    Main entry point for Guard RAG.
    Fetches and synthesizes medical guideline content for a clinical question.

    Args:
        question: Clinical question (e.g., "What is the treatment for MES 3 ulcerative colitis?")
        patient_context: Optional patient info string (e.g., "pMayo 7, Hb 9, MES 3")
        max_results: Number of guideline pages to retrieve

    Returns:
        Synthesized clinical answer with citations.
    """
    timestamp = datetime.datetime.now().isoformat()
    print(f"[{timestamp}] [GuardRAG] Fetching guidelines for: {question[:100]}...")

    # Step 1: Fetch from external sources
    fetched = fetch_guideline_context(question, patient_context, max_results=max_results)
    ok_count = sum(1 for r in fetched if r.get("status") == "ok")
    print(f"[GuardRAG] Retrieved {len(fetched)} results ({ok_count} fully scraped)")

    # Step 2: Synthesize with LLM
    answer = synthesize_guidelines(question, fetched, patient_context)

    return answer
