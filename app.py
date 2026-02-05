import streamlit as st
import os
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Medical MCP RAG & PineBioML", page_icon="🌲", layout="wide")

# MCP Server Parameters
server_params = StdioServerParameters(
    command="./venv/bin/python", # Pastikan path python benar
    args=["src/api/mcp_server.py"],
    env={**os.environ, "PYTHONPATH": "."}
)

# Initialize dark mode state
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# Custom CSS - Clean Modern Medical Theme
dark_bg = "#0f1419" if st.session_state.dark_mode else "#ffffff"
dark_surface = "#1a1f2e" if st.session_state.dark_mode else "#f8f9fa"
dark_text = "#e4e6eb" if st.session_state.dark_mode else "#1a1a1a"
accent_color = "#3b82f6"

st.markdown(f"""
<style>
    /* Clean Modern Theme */
    .stApp {{
        background: {dark_bg};
        color: {dark_text};
    }}
    
    /* Sidebar - Minimal */
    [data-testid="stSidebar"] {{
        background: {dark_surface};
        border-right: 1px solid {'#2d3748' if st.session_state.dark_mode else '#e2e8f0'};
    }}
    
    /* Chat Messages - Clean Cards */
    [data-testid="stChatMessage"] {{
        background: {dark_surface} !important;
        border: 1px solid {'#2d3748' if st.session_state.dark_mode else '#e2e8f0'};
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: none;
    }}
    
    /* Assistant messages - subtle accent */
    [data-testid="stChatMessage"][data-testid*="assistant"] {{
        border-left: 3px solid {accent_color};
    }}
    
    /* User messages - subtle differentiation */
    [data-testid="stChatMessage"][data-testid*="user"] {{
        border-left: 3px solid #6b7280;
    }}
    
    /* Buttons - Clean Minimal */
    .stButton>button {{
        background: {accent_color};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }}
    
    .stButton>button:hover {{
        background: #2563eb;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }}
    
    /* File Uploader - Minimal */
    [data-testid="stFileUploader"] {{
        border: 1px dashed {'#4b5563' if st.session_state.dark_mode else '#d1d5db'};
        border-radius: 6px;
        padding: 1rem;
        background: transparent;
    }}
    
    /* Input - Clean */
    .stTextInput>div>div>input, .stTextArea textarea {{
        background: {dark_surface};
        color: {dark_text};
        border: 1px solid {'#374151' if st.session_state.dark_mode else '#d1d5db'};
        border-radius: 6px;
    }}
    
    /* Headers - Clean Typography */
    h1, h2, h3 {{
        color: {dark_text};
        font-weight: 600;
        letter-spacing: -0.02em;
    }}
    
    h1 {{ font-size: 1.875rem; }}
    h2 {{ font-size: 1.5rem; }}
    h3 {{ font-size: 1.25rem; }}
    
    /* Info/Alert Boxes - Subtle */
    .stAlert {{
        background: {dark_surface};
        border: 1px solid {'#374151' if st.session_state.dark_mode else '#e2e8f0'};
        border-radius: 6px;
        border-left-width: 3px;
    }}
    
    /* Code blocks */
    code {{
        background: {'#374151' if st.session_state.dark_mode else '#f3f4f6'};
        color: {accent_color};
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-size: 0.875rem;
    }}
    
    /* Remove all box shadows and gradients */
    * {{
        box-shadow: none !important;
    }}
    
    /* Chat input */
    .stChatFloatingInputContainer {{
        background: {dark_surface};
        border-top: 1px solid {'#2d3748' if st.session_state.dark_mode else '#e2e8f0'};
    }}
    
    /* Spinner */
    .stSpinner>div {{
        border-top-color: {accent_color} !important;
    }}
    
    /* Links */
    a {{
        color: {accent_color};
        text-decoration: none;
    }}
    
    a:hover {{
        text-decoration: underline;
    }}
</style>
""", unsafe_allow_html=True)

# Helper for MCP tool calls
async def call_mcp_tool(tool_name, arguments):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return result.content[0].text

if "messages" not in st.session_state: st.session_state.messages = []
if "ingested" not in st.session_state: st.session_state.ingested = False

# --- SIDEBAR (Cleaned Up) ---
with st.sidebar:
    # Dark mode toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🌲 PineBioML")
    with col2:
        if st.button("🌓" if st.session_state.dark_mode else "☀️", key="theme_toggle"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    
    # 1. Bagian Upload Data Pasien (Internal otomatis diload di server)
    st.markdown("---")
    st.header("📂 Upload Patient Data")
    st.info("Upload records specifically for the **current patient(s)** analysis.")
    
    uploaded_files = st.file_uploader("Drop PDF/CSV/Excel here", accept_multiple_files=True)
    
    if st.button("🚀 Ingest to Patient Context"):
        if uploaded_files:
            with st.spinner("Processing patient data..."):
                import shutil
                temp_dir = "temp_uploads"
                if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
                os.makedirs(temp_dir, exist_ok=True)
                
                for f in uploaded_files:
                    with open(os.path.join(temp_dir, f.name), "wb") as out:
                        out.write(f.getbuffer())
                
                # REVISI: Hardcode doc_type jadi 'internal_patient' 
                # karena SOP/Knowledge sudah dihandle otomatis server.
                res = asyncio.run(call_mcp_tool("ingest_medical_files", {
                    "directory_path": os.path.abspath(temp_dir),
                    "doc_type": "internal_patient"
                }))
                st.session_state.ingested = True
                st.success(res)
        else:
            st.warning("Please upload files first.")

    st.markdown("---")
    st.header("🆔 Patient ID Filter")
    patient_filter = st.text_input("Active Patient ID (e.g., 001)", "")
    
    if st.button("🗑️ Reset All Data"):
        # Reset Logic
        pass # (Gunakan logic reset tombol lama Anda)

# --- MAIN PAGE ---
st.markdown('<h1 class="header-style">🌲 PineBioML RAG Assistant</h1>', unsafe_allow_html=True)
st.markdown("**System Status:** Internal SOPs & Medical Guidelines are loaded.")

# Report Display
if st.session_state.get("show_report") and "last_plot" in st.session_state:
    st.image(st.session_state.last_plot, caption="Generated PineBioML PCA Plot")
    if "last_interpretation" in st.session_state:
        st.markdown(st.session_state.last_interpretation)
    if st.button("Close Report"):
        st.session_state.show_report = False

# Chat Interface
if st.session_state.ingested:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask DoctorGPT (e.g., 'Summary for ID 8')"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Medical AI is planning and executing..."):
                # Call the Smart Brain via MCP Tool
                dispatch_res = asyncio.run(call_mcp_tool("smart_intent_dispatch", {
                    "question": prompt,
                    "patient_id_filter": patient_filter,
                    "chat_history": st.session_state.messages
                }))
                
                # ERROR HANDLING: Check if response is valid
                if not dispatch_res or dispatch_res.strip() == "":
                    st.error("❌ Error: No response from AI engine. Please try again or check server logs.")
                    st.stop()
                
                import json
                try:
                    decision = json.loads(dispatch_res)
                except json.JSONDecodeError as e:
                    st.error(f"❌ Error parsing AI response: {e}")
                    st.write("**Raw response:**")
                    st.code(dispatch_res)
                    st.stop()
                
                answer = decision.get("answer")
                tool = decision.get("tool")
                tasks = decision.get("tasks", [])

                if tool == "rag":
                    st.markdown(answer)
                    res = answer
                elif tool == "multi_task":
                    # Display the AI's explanation/plan first
                    if answer:
                        st.markdown(answer)
                    
                    res = f"{answer}\n\n" if answer else ""
                    
                    tool_outputs = [] # Collect results for synthesis

                    for i, task in enumerate(tasks):
                        t_name = task.get("tool")
                        if not t_name:
                            t_name = "unknown_task"
                        t_args = task.get("args", {})
                        
                        st.write(f"**Step {i+1}: {t_name.title()}**")
                        
                        if t_name in ["clean", "clean_medical_data"]:
                            with st.spinner("Cleaning medical data..."):
                                m_res = asyncio.run(call_mcp_tool("clean_medical_data", t_args))
                                st.success(m_res)
                                res += f"\n- {m_res}"
                                tool_outputs.append(f"Clean Data: {m_res}")
                        
                        elif t_name in ["plot", "generate_medical_plot"]:
                            with st.spinner(f"Generating plot..."):
                                p_args = {
                                    "plot_type": t_args.get("plot_type", "pca"),
                                    "patient_ids": patient_filter,
                                    "target_column": t_args.get("target_column")
                                }
                                m_res = asyncio.run(call_mcp_tool("generate_medical_plot", p_args))
                                if "|||" in m_res:
                                    path, interp = m_res.split("|||")
                                    st.image(path)
                                    st.markdown(interp)
                                    res += f"\n- {interp}"
                                    tool_outputs.append(f"Plot Findings: {interp}")
                                else:
                                    st.error(m_res)

                        elif t_name == "run_pls_analysis":
                            with st.spinner("Running PLS-DA..."):
                                m_res = asyncio.run(call_mcp_tool("run_pls_analysis", {}))
                                if "|||" in m_res:
                                    path, txt = m_res.split("|||")
                                    st.image(path)
                                    st.markdown(txt)
                                    res += f"\n- PLS-DA Analysis complete."
                                    tool_outputs.append(f"PLS-DA Findings: {txt}")

                        elif t_name == "run_umap_analysis":
                            with st.spinner("Running UMAP..."):
                                m_res = asyncio.run(call_mcp_tool("run_umap_analysis", {}))
                                if "|||" in m_res:
                                    path, txt = m_res.split("|||")
                                    st.image(path)
                                    st.markdown(txt)
                                    res += f"\n- UMAP Analysis complete."
                                    tool_outputs.append(f"UMAP Findings: {txt}")

                        elif t_name == "run_correlation_heatmap":
                            with st.spinner("Generating Heatmap..."):
                                m_res = asyncio.run(call_mcp_tool("run_correlation_heatmap", {}))
                                if "|||" in m_res:
                                    path, txt = m_res.split("|||")
                                    st.image(path)
                                    st.markdown(txt)
                                    res += f"\n- Heatmap complete."
                                    tool_outputs.append(f"Heatmap Findings: {txt}")
                        
                        elif t_name in ["train", "train_medical_model"]:
                            with st.spinner("Training model..."):
                                m_res = asyncio.run(call_mcp_tool("train_medical_model", t_args))
                                st.markdown(m_res)
                                res += f"\n- Training complete."
                                tool_outputs.append(f"Training Results: {m_res}")
                        
                        elif t_name == "inspect_knowledge_base":
                            with st.spinner("Inspecting knowledge base..."):
                                m_res = asyncio.run(call_mcp_tool("inspect_knowledge_base", {}))
                                st.info("Knowledge Base Inventory")
                                st.markdown(m_res)
                                res += f"\n- Knowledge inspection complete."
                                tool_outputs.append(f"Knowledge Inventory: {m_res}")

                        elif t_name in ["discover", "discover_markers"]:
                            with st.spinner("Discovering biomarkers..."):
                                m_res = asyncio.run(call_mcp_tool("discover_markers", t_args))
                                st.markdown(m_res)
                                res += f"\n- Biomarkers identified."
                                tool_outputs.append(f"Discovery Results: {m_res}")
                        
                        elif t_name in ["report", "generate_medical_report"]:
                            with st.spinner("Generating report..."):
                                m_res = asyncio.run(call_mcp_tool("generate_medical_report", {}))
                                if "|||" in m_res:
                                    path, txt = m_res.split("|||")
                                    st.image(path)
                                    st.markdown(txt)
                                    res += f"\n- Report generated."
                                    tool_outputs.append(f"Report Summary: {txt}")
                        
                        elif t_name in ["rag", "query_medical_rag"]:
                            m_res = asyncio.run(call_mcp_tool("query_medical_rag", {"question": prompt}))
                            st.markdown(m_res)
                            res += f"\n- {m_res}"
                            tool_outputs.append(f"RAG Knowledge: {m_res}")

                        elif t_name == "exact_identifier_search":
                            with st.spinner("Finding exact matches..."):
                                m_res = asyncio.run(call_mcp_tool("exact_identifier_search", {"query": prompt, "patient_id_filter": patient_filter}))
                                st.markdown("### 🔍 Exact Match Results")
                                st.markdown(m_res) # This will render the markdown + code blocks
                                res += f"\n- Exact search completed."
                                tool_outputs.append(f"Exact Search Results: {m_res}")

                        elif t_name in ["describe", "get_data_context"]:
                            with st.spinner("Analyzing data summary..."):
                                m_res = asyncio.run(call_mcp_tool("get_data_context", {}))
                                st.markdown(m_res)
                                res += f"\n- Data Summary Loaded."
                                tool_outputs.append(f"Data Summary: {m_res}")

                    # Final Step: Clinical Synthesis
                    if tool_outputs:
                        with st.spinner("Dr. AI is analyzing the results..."):
                            combined_findings = "\n".join(tool_outputs)
                            synth_res = asyncio.run(call_mcp_tool("synthesize_medical_results", {
                                "question": prompt,
                                "results": combined_findings
                            }))
                            st.markdown("### 📝 Clinical Synthesis")
                            st.info(synth_res)
                            res += f"\n\n### Clinical Synthesis\n{synth_res}"
                else:
                    st.markdown(answer)
                    res = answer
                
        st.session_state.messages.append({"role": "assistant", "content": res})
else:
    st.info("👈 Ingest medical files via MCP to start.")
