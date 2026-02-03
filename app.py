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

if "messages" not in st.session_state: st.session_state.messages = []
if "ingested" not in st.session_state: st.session_state.ingested = False

# --- SIDEBAR (Cleaned Up) ---
with st.sidebar:
    st.title("🌲 PineBioML Config")
    
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
                        
                        if t_name == "clean":
                            with st.spinner("Cleaning medical data..."):
                                m_res = asyncio.run(call_mcp_tool("clean_medical_data", t_args))
                                st.success(m_res)
                                res += f"\n- {m_res}"
                                tool_outputs.append(f"Clean Data: {m_res}")
                        
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
                                    tool_outputs.append(f"Plot ({p_args['plot_type']}) Findings: {interp}")
                                else:
                                    st.error(m_res)
                        
                        elif t_name == "train":
                            with st.spinner("Training model..."):
                                m_res = asyncio.run(call_mcp_tool("train_medical_model", t_args))
                                st.markdown(m_res)
                                res += f"\n- Training complete."
                                tool_outputs.append(f"Training Results: {m_res}")
                        
                        elif t_name == "discover":
                            with st.spinner("Discovering biomarkers..."):
                                m_res = asyncio.run(call_mcp_tool("discover_markers", t_args))
                                st.markdown(m_res)
                                res += f"\n- Biomarkers identified."
                                tool_outputs.append(f"Discovery Results: {m_res}")
                        
                        elif t_name == "report":
                            with st.spinner("Generating report..."):
                                m_res = asyncio.run(call_mcp_tool("generate_medical_report", {}))
                                if "|||" in m_res:
                                    path, txt = m_res.split("|||")
                                    st.image(path)
                                    st.markdown(txt)
                                    res += f"\n- Report generated."
                                    tool_outputs.append(f"Report Summary: {txt}")
                        
                        elif t_name == "rag":
                            m_res = asyncio.run(call_mcp_tool("query_medical_rag", {"question": prompt}))
                            st.markdown(m_res)
                            res += f"\n- {m_res}"
                            tool_outputs.append(f"RAG Knowledge: {m_res}")

                        elif t_name == "describe":
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
