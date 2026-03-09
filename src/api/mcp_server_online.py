"""
MCP Server — Online Guard RAG Edition
======================================
Identical to mcp_server.py except Guard RAG uses ONLINE web search
(DuckDuckGo → ACG, ECCO, BSG, AGA, PubMed, etc.) instead of offline
embedded knowledge + local PDF guidelines.

This file imports the ENTIRE original MCP server, then overrides ONLY
the `query_guard_rag` tool with an online version.
"""

import os
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, ".env"))

# ============================================================================
# Import the ORIGINAL mcp_server — this registers all tools on `mcp`
# ============================================================================
from src.api.mcp_server import mcp, rag_engine, pine_log

# ============================================================================
# OVERRIDE: query_guard_rag → ONLINE web search version
# ============================================================================

# Remove the offline guard_rag tool so we can re-register it
# FastMCP stores tools internally — we remove the old one by name
if hasattr(mcp, '_tool_manager'):
    # FastMCP internal: remove the old tool
    try:
        tools = mcp._tool_manager._tools
        if "query_guard_rag" in tools:
            del tools["query_guard_rag"]
            pine_log("🔄 Removed offline query_guard_rag, replacing with online version")
    except Exception:
        pass


@mcp.tool()
def query_guard_rag(query_intent: str) -> str:
    """
    Guard RAG (ONLINE): Fetches medical guidelines via live web search.
    Uses DuckDuckGo to search authoritative medical sites (ACG, ECCO, BSG, AGA,
    PubMed, WHO, NICE, etc.) and synthesizes the results with LLM.
    """
    try:
        from PineBioML.rag.external_guidelines import (
            detect_medical_domain,
            extract_patient_context,
            fetch_web_guidelines,
            _synthesize_web_only,
            _synthesize_combined,
            _synthesize_fallback,
        )
        from PineBioML.rag.clinical_knowledge import match_guideline, format_guideline_answer

        pine_log(f"🌐 Guard RAG (ONLINE): Searching web for: {query_intent[:80]}...")

        # Extract patient context from the query
        patient_context = extract_patient_context(query_intent)
        pine_log(f"🏥 Patient context detected: {patient_context or 'none'}")

        # Step 1: Try embedded KB for instant match (still useful as a first layer)
        kb_matches = match_guideline(query_intent, patient_context)
        kb_answer = format_guideline_answer(kb_matches, query_intent)

        if kb_matches:
            pine_log(f"📚 Embedded KB matched: {kb_matches[0]['id']}")

        # Step 2: ONLINE — Fetch guidelines from the web (DuckDuckGo)
        pine_log(f"🔍 Searching DuckDuckGo for medical guidelines...")
        web_results = fetch_web_guidelines(query_intent, patient_context, max_results=3)

        if web_results:
            pine_log(f"✅ Web search returned {len(web_results)} results")
            # Build web context string
            web_context_parts = []
            for wr in web_results:
                source = wr.get("source_name", "Unknown")
                title = wr.get("title", "")
                url = wr.get("url", "")
                content = wr.get("content", wr.get("snippet", ""))
                web_context_parts.append(
                    f"**{source}** — {title}\n"
                    f"URL: {url}\n"
                    f"Content: {content}\n"
                )
            web_context = "\n---\n".join(web_context_parts)

            # Synthesize based on what we have
            if kb_answer:
                # Combine embedded KB (primary) + web (enrichment)
                pine_log("🧬 Combining embedded KB + web results")
                answer = _synthesize_combined(query_intent, kb_answer, web_context, patient_context)
            else:
                # Web-only synthesis
                pine_log("🌐 Synthesizing from web results only")
                answer = _synthesize_web_only(query_intent, web_context, patient_context)

            # Append source URLs for transparency
            url_citations = "\n\n🌐 **Online Sources Consulted:**\n"
            for wr in web_results:
                src = wr.get("source_name", "")
                url = wr.get("url", "")
                title = wr.get("title", "")
                status = wr.get("status", "")
                url_citations += f"- **{src}**: [{title}]({url}) ({status})\n"

            answer += url_citations
            pine_log(f"✅ Guard RAG (ONLINE): Final answer ({len(answer)} chars)")
            return answer

        elif kb_answer:
            # No web results but embedded KB matched
            pine_log("⚠️ Web search returned no results, using embedded KB only")
            return kb_answer + "\n\n> ℹ️ *Online search returned no additional results.*"

        else:
            # Nothing from web, nothing from KB — use LLM general knowledge
            pine_log("⚠️ No web results and no KB match. Using LLM fallback.")
            return _synthesize_fallback(query_intent, patient_context)

    except Exception as e:
        pine_log(f"❌ Guard RAG (ONLINE) Error: {e}")
        import traceback
        pine_log(traceback.format_exc())
        return f"⚠️ Online guideline search failed: {str(e)}"


# ============================================================================
# ENTRY POINT — Run this server standalone
# ============================================================================
if __name__ == "__main__":
    pine_log("🚀 Starting MCP Server — ONLINE Guard RAG Edition")
    mcp.run(transport="stdio")
