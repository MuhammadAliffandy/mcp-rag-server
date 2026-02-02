import streamlit as st
import os
import asyncio
import pandas as pd
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Page Config
st.set_page_config(page_title="Medical MCP RAG & PineBioML", page_icon="🌲", layout="wide")

# MCP Server Parameters
server_params = StdioServerParameters(
    command="./venv/bin/python",
    args=["mcp_server.py"],
)

# Custom CSS
st.markdown("""
<style>
    .stChatFloatingInputContainer { background-color: #f8f9fa; }
    .header-style { color: #2e7d32; font-family: 'Inter', sans-serif; }
</style>
""", unsafe_allow_html=True)

# Helper for MCP tool calls
async def call_mcp_tool(tool_name, arguments):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return result.content[0].text

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested" not in st.session_state:
    st.session_state.ingested = False

# Sidebar
with st.sidebar:
    st.title("🌲 PineBioML Config (MCP)")
    
    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ OPENAI_API_KEY not found in .env")
    else:
        st.success("✅ OpenAI API Key Loaded")

    st.markdown("---")
    st.header("📂 Data Ingestion")
    doc_type = st.radio("Select Document Type", ["Patient Records (Internal)", "Medical Guidelines (External)"])
    uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)
    
    if st.button("🚀 Ingest Files (via MCP)"):
        if uploaded_files:
            with st.spinner("Processing documents via MCP Server..."):
                import shutil
                temp_dir = "temp_uploads"
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                os.makedirs(temp_dir, exist_ok=True)
                
                for f in uploaded_files:
                    with open(os.path.join(temp_dir, f.name), "wb") as out:
                        out.write(f.getbuffer())
                
                dtype_key = "internal_patient" if "Patient" in doc_type else "external_guideline"
                res = asyncio.run(call_mcp_tool("ingest_medical_files", {
                    "directory_path": os.path.abspath(temp_dir),
                    "doc_type": dtype_key
                }))
                st.session_state.ingested = True
                st.success(res)
        else:
            st.warning("Please upload files first.")

    st.markdown("---")
    st.header("🆔 Patient Filter")
    patient_filter = st.text_input("Filter IDs (e.g., 1-5)", "")
    
    with st.expander("🔍 View Data Context (MCP)"):
        if st.button("Refresh Context"):
            ctx = asyncio.run(call_mcp_tool("get_data_context", {}))
            st.markdown(ctx)
            
    if st.button("📊 Generate PineBioML Report"):
        with st.spinner("Generating plot via MCP Tool..."):
            res = asyncio.run(call_mcp_tool("generate_medical_plot", {
                "plot_type": "pca", # Default for report button
                "patient_ids": patient_filter
            }))
            if "|||" in res:
                path, interpretation = res.split("|||")
                st.session_state.last_plot = path
                st.session_state.last_interpretation = interpretation
                st.session_state.show_report = True
            else:
                st.error(res)
    st.markdown("---")
    if st.button("🗑️ Reset System (Clear Cache)"):
        with st.spinner("Clearing all data..."):
            res = asyncio.run(call_mcp_tool("reset_medical_database", {}))
            st.session_state.messages = []
            st.session_state.ingested = False
            st.session_state.show_report = False
            if "last_plot" in st.session_state:
                del st.session_state.last_plot
            st.success(res)
            st.rerun()

# Main App Header
st.markdown('<h1 class="header-style">🌲 PineBioML RAG Assistant (MCP)</h1>', unsafe_allow_html=True)
st.markdown("Standardized Medical Context Protocol (MCP) Server Integration.")

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
                    "patient_id_filter": patient_filter
                }))
                
                import json
                decision = json.loads(dispatch_res)
                answer = decision.get("answer")
                tool = decision.get("tool")
                tasks = decision.get("tasks", [])

                if tool == "rag":
                    st.markdown(answer)
                    res = answer
                elif tool == "multi_task":
                    res = ""
                    for i, task in enumerate(tasks):
                        t_name = task.get("tool")
                        t_args = task.get("args", {})
                        
                        st.write(f"**Step {i+1}: {t_name.title()}**")
                        
                        if t_name == "clean":
                            with st.spinner("Cleaning medical data..."):
                                m_res = asyncio.run(call_mcp_tool("clean_medical_data", t_args))
                                st.success(m_res)
                                res += f"\n- {m_res}"
                        
                        elif t_name == "plot":
                            with st.spinner(f"Generating {t_args.get('plot_type', 'plot')}..."):
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
                                else:
                                    st.error(m_res)
                        
                        elif t_name == "train":
                            with st.spinner("Training model..."):
                                m_res = asyncio.run(call_mcp_tool("train_medical_model", t_args))
                                st.markdown(m_res)
                                res += f"\n- Training complete."
                        
                        elif t_name == "discover":
                            with st.spinner("Discovering biomarkers..."):
                                m_res = asyncio.run(call_mcp_tool("discover_markers", t_args))
                                st.markdown(m_res)
                                res += f"\n- Biomarkers identified."
                        
                        elif t_name == "report":
                            with st.spinner("Generating report..."):
                                m_res = asyncio.run(call_mcp_tool("generate_medical_report", {}))
                                if "|||" in m_res:
                                    path, txt = m_res.split("|||")
                                    st.image(path)
                                    st.markdown(txt)
                                    res += f"\n- Report generated."
                        
                        elif t_name == "rag":
                            m_res = asyncio.run(call_mcp_tool("query_medical_rag", {"question": prompt}))
                            st.markdown(m_res)
                            res += f"\n- {m_res}"
                else:
                    st.markdown(answer)
                    res = answer
                
        st.session_state.messages.append({"role": "assistant", "content": res})
else:
    st.info("👈 Ingest medical files via MCP to start.")
