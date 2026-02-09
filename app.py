import streamlit as st
import os
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
import base64
import json
import re

def get_img_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

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

# Theme Colors
if st.session_state.dark_mode:
    bg_color = "#000000"
    sidebar_bg = "#0A0A0A"
    card_bg = "#111111"
    text_color = "#FFFFFF"
    subtext_color = "#666666"
    border_color = "#1A1A1A"
    bubble_bg = "#1E1E1E"
else:
    bg_color = "#FFFFFF"
    sidebar_bg = "#F8F9FA"
    card_bg = "#F0F2F6"
    text_color = "#000000"
    subtext_color = "#555555"
    border_color = "#E0E0E0"
    bubble_bg = "#F0F2F6"

# Custom CSS - Premium Monochrome Aesthetic (Dynamic)
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700&family=Poppins:wght@300;400;500&display=swap');

    /* Global Typography & Colors */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {bg_color} !important;
        font-family: 'Poppins', sans-serif !important;
        color: {text_color} !important;
    }}
    
    .stApp {{
        background: {bg_color};
    }}

    h1, h2, h3, .header-style {{
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 700 !important;
        color: {text_color} !important;
        letter-spacing: -0.02em;
    }}

    /* Sidebar Styling - Enhanced Dashboard Look */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {border_color};
        padding-top: 20px;
    }}
    
    .brand-header {{
        background: {sidebar_bg};
        border: 1px solid {border_color};
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    
    .brand-icon {{
        background: {border_color};
        color: {text_color};
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
    }}
    
    .nav-label {{
        font-size: 0.75rem;
        color: {subtext_color};
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 20px 0 10px 10px;
        font-weight: 600;
    }}
    
    .nav-item {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 12px;
        border-radius: 8px;
        color: {subtext_color};
        cursor: pointer;
        transition: all 0.2s;
        text-decoration: none;
        margin-bottom: 4px;
    }}
    
    .nav-item:hover {{
        background: {card_bg};
        color: {text_color};
    }}
    
    /* Input Styling - Search Type */
    .stTextInput input {{
        background-color: {bg_color} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px !important;
        color: {text_color} !important;
    }}

    /* Chat Elements - New Flexbox System */
    .user-container {{
        display: flex !important;
        flex-direction: row-reverse !important;
        align-items: flex-start !important;
        justify-content: flex-start !important; /* Starts from right in row-reverse */
        width: 100% !important;
        margin: 1.5rem 0 !important;
        gap: 12px !important;
    }}

    .user-bubble {{
        background-color: {bubble_bg} !important;
        border: 1px solid {border_color} !important;
        color: {text_color} !important;
        padding: 12px 18px !important;
        border-radius: 20px 20px 4px 20px !important;
        max-width: 75% !important;
        font-size: 1.1rem !important; /* Increased font size */
        box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
    }}

    .assistant-container {{
        display: flex !important;
        flex-direction: row !important;
        align-items: flex-start !important;
        width: 100% !important;
        margin: 1.5rem 0 !important;
        gap: 12px !important;
    }}

    .assistant-content {{
        color: {text_color} !important;
        max-width: 85% !important;
        font-size: 1.1rem !important; /* Increased font size */
        line-height: 1.6 !important;
        margin-top: 4px !important;
    }}

    .msg-avatar {{
        width: 32px !important;
        height: 32px !important;
        border-radius: 50% !important; /* Circle avatars */
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex-shrink: 0 !important;
        background: {card_bg}; /* Blends with cards */
        border: 1px solid {border_color};
    }}

    .user-avatar {{ 
        /* Specific override if needed, but shared style is good for blending */
    }}
    
    .assistant-avatar {{ 
        /* Specific override if needed */
    }}
    
    .avatar-icon {{
        width: 18px;
        height: 18px;
        fill: {subtext_color}; /* Subtle icon color */
    }}
    
    .user-avatar .avatar-icon {{
        fill: {text_color}; /* White/Text Color for User as requested */
    }}
    
    .assistant-avatar .avatar-icon {{
        fill: #10B981; /* Muted Emerald for AI (PineBio) */
    }}

    /* Disable default Streamlit chat padding/backgrounds */
    [data-testid="stChatMessage"] {{
        padding: 0 !important;
        background: transparent !important;
    }}
    
    [data-testid="stChatMessage"] > div:first-child {{
        display: none !important; /* Hide original avatar */
    }}
    
    [data-testid="stChatMessageContent"] {{
        padding: 0 !important;
        background: transparent !important;
    }}
    
    /* Input Area - Seamless Integration */
    [data-testid="stChatInput"] {{
        background-color: {bg_color} !important;
        border: 1px solid {border_color} !important;
        border-radius: 14px !important;
        padding: 0.3rem !important;
    }}
    
    /* Force pure black for the bottom container */
    .stChatFloatingInputContainer, [data-testid="stBottomBlockContainer"] {{
        background-color: {bg_color} !important;
        border-top: none !important;
    }}
    
    /* Ensure no background bleed from sidebar or main view */
    [data-testid="stAppViewBlockContainer"] {{
        background-color: {bg_color} !important;
    }}
    
    /* Prompt Cards (Empty State) */
    .prompt-card {{
        background: {sidebar_bg};
        border: 1px solid {border_color};
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: all 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        color: {subtext_color};
        font-size: 0.85rem;
        cursor: pointer;
    }}
    
    .prompt-card:hover {{
        border-color: {subtext_color};
        background: {card_bg};
        color: {text_color};
    }}
    
    .prompt-icon {{
        font-size: 1.5rem;
        margin-bottom: 10px;
        color: {text_color};
        opacity: 0.6;
    }}

    /* Buttons - Refined Minimal */
    .stButton>button {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        font-weight: 500 !important;
        border-radius: 10px !important;
        border: 1px solid {border_color} !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.2s;
    }}
    
    .stButton>button:hover {{
        background-color: {sidebar_bg} !important;
        border-color: {subtext_color} !important;
    }}
    
    /* File Uploader */
    [data-testid="stFileUploader"] {{
        background: {sidebar_bg};
        border: 1px dashed {border_color};
        border-radius: 10px;
    }}

    /* Hide standard Streamlit elements for "Pure" look */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    [data-testid="stHeader"] {{background: rgba(0,0,0,0);}}
    
    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 5px; }}
    ::-webkit-scrollbar-track {{ background: {bg_color}; }}
    ::-webkit-scrollbar-thumb {{ background: {border_color}; border-radius: 10px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {subtext_color}; }}
</style>
""", unsafe_allow_html=True)

# SVG Icons
icon_user = """<svg class="avatar-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M20 21C20 19.6044 20 18.9067 19.8278 18.3389C19.44 17.0605 18.4395 16.06 17.1611 15.6722C16.5933 15.5 15.8956 15.5 14.5 15.5H9.5C8.10444 15.5 7.40665 15.5 6.83886 15.6722C5.56045 16.06 4.56004 17.0605 4.17224 18.3389C4 18.9067 4 19.6044 4 21M16.5 7.5C16.5 9.98528 14.4853 12 12 12C9.51472 12 7.5 9.98528 7.5 7.5C7.5 5.01472 9.51472 3 12 3C14.4853 3 16.5 5.01472 16.5 7.5Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>"""

icon_ai = """<svg class="avatar-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M9.813 15.904L9 18.75L8.187 15.904C7.79 14.512 6.696 13.418 5.304 13.021L2.25 12.25L5.304 11.479C6.696 11.082 7.79 9.988 8.187 8.596L9 5.75L9.813 8.596C10.21 9.988 11.304 11.082 12.696 11.479L15.75 12.25L12.696 13.021C11.304 13.418 10.21 14.512 9.813 15.904Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M16.89 19.782L16.5 21.125L16.11 19.782C15.922 19.125 15.405 18.608 14.748 18.42L13.25 18.062L14.748 17.705C15.405 18.517 15.922 18.032 16.11 17.375L16.5 16.032L16.89 17.375C17.078 18.032 17.595 18.517 18.252 18.705L19.75 19.062L18.252 19.42C17.595 19.608 17.078 20.125 16.89 19.782Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M19.488 7.379L19.25 8.25L19.012 7.379C18.887 6.942 18.543 6.598 18.106 6.473L17.25 6.25L18.106 6.027C18.543 5.902 18.887 5.558 19.012 5.121L19.25 4.25L19.488 5.121C19.613 5.558 19.957 5.902 20.394 6.027L21.25 6.25L20.394 6.473C19.957 6.598 19.613 6.942 19.488 7.379Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>"""

# Helper for MCP tool calls
async def call_mcp_tool(tool_name, arguments):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return result.content[0].text

if "messages" not in st.session_state: st.session_state.messages = []
if "ingested" not in st.session_state: st.session_state.ingested = False

# --- SIDEBAR (Premium Dashboard) ---
with st.sidebar:
    # 1. Brand Header (ChatGPT Style)
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("""
        <div class="brand-header">
            <div class="brand-icon">P</div>
            <div>
                <div style="font-weight:600; font-size:1rem; color:#FFFFFF;">PineBioML-4</div>
                <div style="font-size:0.75rem; color:#666;">Medical Intelligence Core</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("🌓" if st.session_state.dark_mode else "☀️", key="theme_toggle"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    
    # 2. Search / Filter (Minimal)
    patient_filter = st.text_input("Search ID...", "", placeholder="Patient ID (e.g. 001)", label_visibility="collapsed")
    
    # 3. Navigation Section
    st.markdown('<div class="nav-label">Main</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">🏠 Home</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item" style="background:#111; color:#fff;">✨ Chat</div>', unsafe_allow_html=True)
    
    # 4. Data / Workspace Section
    st.markdown('<div class="nav-label">Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">📂 Library</div>', unsafe_allow_html=True)
    
    # Data Management
    uploaded_files = st.file_uploader("Upload Medical Records", accept_multiple_files=True, label_visibility="collapsed")
    
    if st.button("🚀 Ingest Records", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing..."):
                import shutil
                temp_dir = "temp_uploads"
                if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
                os.makedirs(temp_dir, exist_ok=True)
                
                for f in uploaded_files:
                    with open(os.path.join(temp_dir, f.name), "wb") as out:
                        out.write(f.getbuffer())
                
                res = asyncio.run(call_mcp_tool("ingest_medical_files", {
                    "directory_path": os.path.abspath(temp_dir),
                    "doc_type": "session_upload"
                }))
                st.session_state.ingested = True
                st.success("Context loaded.")
        else:
            st.warning("Upload first.")

    # 5. Settings / Control
    st.markdown('<div class="nav-label">Control</div>', unsafe_allow_html=True)
    if st.button("🗑️ Reset Session", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- MAIN PAGE ---

# 1. Capture Input First (Execution Order Fix)
# 1. Capture Input First (Execution Order Fix)
prompt = st.chat_input("Ask about SOPs or patient data (e.g., 'What is in the guidelines?')")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.processing_pending = True
    st.rerun()

# 2. Decide what to render
if not st.session_state.messages:
    # Empty State / Landing Page
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col_c1, col_c2, col_c3 = st.columns([1, 4, 1])
    with col_c2:
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 20px; color:{text_color};">⠿</div>
            <h1 style="font-size: 3.5rem !important; margin-bottom: 0px; font-weight:700;">Intelligence Core</h1>
            <p style="color: {subtext_color}; font-size: 1.1rem; margin-top: 10px; font-weight:300; max-width: 600px; margin-left: auto; margin-right: auto;">Advanced RAG orchestration for clinical multi-omics and precision medical data analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Prompt Cards Grid
        p1, p2, p3, p4 = st.columns(4)
        with p1: 
            st.markdown('<div class="prompt-card"><div class="prompt-icon">📊</div>Analyze my patient data and tell me the findings</div>', unsafe_allow_html=True)
        with p2:
            st.markdown('<div class="prompt-card"><div class="prompt-icon">📜</div>Check medical SOPs for surgery verification</div>', unsafe_allow_html=True)
        with p3:
            st.markdown('<div class="prompt-card"><div class="prompt-icon">🔍</div>Find specific accession codes in documents</div>', unsafe_allow_html=True)
        with p4:
            st.markdown('<div class="prompt-card"><div class="prompt-icon">💡</div>Discover significant markers in data</div>', unsafe_allow_html=True)
else:
    # Render Chat History
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            st.markdown(f"""
            <div class="user-container">
                <div class="msg-avatar user-avatar">{icon_user}</div>
                <div class="user-bubble">{content}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-container">
                <div class="msg-avatar assistant-avatar">{icon_ai}</div>
                <div class="assistant-content">{content}</div>
            </div>
            """, unsafe_allow_html=True)

# 3. Handle AI Response (Triggered by flag)
if st.session_state.get("processing_pending", False):
    st.session_state.processing_pending = False # Reset flag
    
    # Get the latest user prompt
    last_user_prompt = st.session_state.messages[-1]["content"]

    # Placeholder for the new assistant message to keep things clean
    with st.spinner("Medical AI is planning and executing..."):
        # Call the Smart Brain via MCP Tool
        dispatch_res = asyncio.run(call_mcp_tool("smart_intent_dispatch", {
            "question": last_user_prompt,
            "patient_id_filter": patient_filter,
            "chat_history": st.session_state.messages[:-1] # History excluding current
        }))

        # ERROR HANDLING: Check if response is valid
        if not dispatch_res or dispatch_res.strip() == "":
            st.error("❌ Error: No response from AI engine. Please try again or check server logs.")
            st.stop()
        
        import json
        try:
            decision = json.loads(dispatch_res)
            
            # Quiet logging for production persona
            # pine_log(f"Smart Dispatch Result: {decision.get('tool')} with {len(decision.get('tasks', []))} tasks")
            
        except json.JSONDecodeError as e:
            st.error(f"❌ Error parsing AI response: {e}")
            st.write("**Raw response:**")
            st.code(dispatch_res)
            st.stop()
        
        # DEBUG: Check if we have tasks
        if not decision.get("tasks"):
            print("⚠️ DEBUG: No tasks found in Orchestrator response!")
            print(f"Response: {dispatch_res}")
        else:
            print(f"✅ DEBUG: Orchestrator returned {len(decision.get('tasks'))} tasks.")
            for t in decision.get("tasks"):
                print(f"  - Tool: {t.get('tool')}, Args: {t.get('args')}")

        answer = decision.get("answer")
        tool = decision.get("tool", "").strip().lower() # Normalize immediately
        raw_tasks = decision.get("tasks", [])
        rag_context = decision.get("rag_context", "")

        # REDUNDANT FAILSAFE: Task Flattener for LLM Hallucinations
        tasks = []
        _tasks_to_process = raw_tasks if isinstance(raw_tasks, list) else [raw_tasks]
        for t in _tasks_to_process:
            if not isinstance(t, dict): continue
            
            # Detect nested "0", "1", etc. keys (Screenshot turn 1745 confirmed this structure)
            numeric_keys = [k for k in t.keys() if str(k).isdigit()]
            if len(t) == 1 and numeric_keys:
                inner = t[numeric_keys[0]]
                if isinstance(inner, dict):
                    print(f"✨ Redundant Flattening in App: Unpacked key '{numeric_keys[0]}'")
                    tasks.append(inner)
                else:
                    tasks.append(t)
            else:
                tasks.append(t)

        # SURGICAL DEBUG for Visualization Fix
        print(f"DEBUG: tool='{tool}', tasks_len={len(tasks)}, raw_len={len(raw_tasks)}")
        # SURGICAL DEBUG for Visualization Fix (Sidebar Only for cleanliness)
        with st.sidebar:
            st.divider()
            st.caption("🔍 Orchestrator Status")
            st.code(f"Tool: {tool}\nTasks: {len(tasks)}")
            if tasks:
                st.json(tasks)

        # Store rag_context in session state for later synthesis
        st.session_state.current_rag_context = rag_context

        # Wrap assistant output in custom container
        st.markdown(f'<div class="assistant-container"><div class="msg-avatar assistant-avatar">{icon_ai}</div><div class="assistant-content">', unsafe_allow_html=True)
        
        if tool == "rag":
            st.markdown(answer)
            res = answer
        elif tool == "multi_task":
            # Display the AI's explanation/plan first
            if answer:
                st.markdown(answer)
            
            # Clinical Execution Plan
            with st.expander("🛠️ Clinical Execution Plan", expanded=False):
                st.json(tasks)
            
            res = f"{answer}\n\n" if answer else ""
            
            tool_outputs = [] # Collect results for synthesis

            # Mapping technical tool names to professional clinical titles
            TOOL_LABEL_MAP = {
                "extract_data_from_rag": "Retrieving Clinical Records",
                "generate_medical_plot": "Generating Diagnostic Visualization",
                "clean_medical_data": "Pre-processing Clinical Data",
                "run_pls_analysis": "Performing PLS-DA Separation Analysis",
                "run_umap_analysis": "Executing UMAP Clustering Analysis",
                "discover_markers": "Identifying Significant Biomarkers",
                "train_medical_model": "Developing Predictive Model",
                "generate_data_overview": "Preparing Comprehensive Data Report",
                "run_correlation_heatmap": "Analyzing Feature Correlations",
                "query_medical_rag": "Consulting Medical Knowledge Base",
                "get_data_context": "Analyzing Data Schema",
                "exact_identifier_search": "Searching Patient Records",
                "query_exprag_hybrid": "Searching Similar Cases & Knowledge"
            }

            for i, task in enumerate(tasks):
                t_name = task.get("tool")
                if not t_name:
                    t_name = "unknown_task"
                
                # Normalize tool name
                t_name = t_name.strip().lower()

                t_args = task.get("args", {})
                
                # Use mapped label or title-cased tool name
                display_label = TOOL_LABEL_MAP.get(t_name, t_name.replace("_", " ").title())
                st.markdown(f"**Step {i+1}: {display_label}**")
                
                if t_name in ["clean", "clean_medical_data"]:
                    with st.spinner("Cleaning medical data..."):
                        m_res = asyncio.run(call_mcp_tool("clean_medical_data", t_args))
                        st.success(m_res)
                        res += f"\n- {m_res}"
                        tool_outputs.append(f"Clean Data: {m_res}")
                
                elif t_name in ["extract", "extract_data_from_rag"]:
                    with st.spinner("Dr. AI is meticulously retrieving clinical records..."):
                        m_res = asyncio.run(call_mcp_tool("extract_data_from_rag", t_args))
                        if "success|||" in m_res:
                            _, summary = m_res.split("|||")
                            # Extract common label for clean display
                            clean_summary = summary.split(". Columns:")[0] if ". Columns:" in summary else summary
                            st.success(f"✅ {clean_summary}")
                            
                            # Move technical column info to hidden expander
                            if ". Columns:" in summary:
                                with st.expander("🔍 View Technical Data Schema"):
                                    st.write(summary.split(". Columns:")[1])
                                    
                            res += f"\n- Medical data preparation complete: {clean_summary}"
                            tool_outputs.append(f"Data Extraction: {summary}")
                        else:
                            st.error(f"❌ Record retrieval challenge: {m_res}")
                            res += f"\n- Extraction failed: {m_res}"
                
                elif t_name in ["plot", "generate_medical_plot"]:
                    with st.spinner(f"Generating plot..."):
                        # Fix: Pass ALL arguments from LLM, injecting context
                        p_args = {k: v for k, v in t_args.items() if v is not None} # Sanitize immediately
                        p_args["patient_ids"] = patient_filter
                        if "plot_type" not in p_args: p_args["plot_type"] = "pca"
                        
                        # Fix argument types
                        if "styling" in p_args and isinstance(p_args["styling"], dict):
                            p_args["styling"] = json.dumps(p_args["styling"])
                        elif "styling" not in p_args:
                            p_args["styling"] = "{}" # Default empty JSON
                            
                        # Fix data_source (LLM hallucinations)
                        p_args["data_source"] = "session"
                        
                        # Final conversion to ensure NO None values reach the server
                        # (Server now supports Optional, but passing empty string is safer for Pydantic)
                        for opt_col in ["x_column", "y_column", "target_column"]:
                            if opt_col not in p_args or p_args[opt_col] is None:
                                p_args[opt_col] = "" # Map to empty string instead of None

                        try:
                            # 1. CALL THE TOOL
                            m_res = asyncio.run(call_mcp_tool("generate_medical_plot", p_args))
                            
                            # 2. DEBUG: Show RAW Response immediately
                            with st.expander("🛠️ Tool Execution Debug", expanded=True):
                                st.code(f"Raw Response: {m_res}")
                                if not m_res:
                                    st.error("Received EMPTY response from server!")
                            
                            # 3. Handle Result
                            if "|||" in m_res:
                                path, interp = m_res.split("|||")
                                
                                # Verify file existence
                                if os.path.exists(path):
                                    st.image(path)
                                    
                                    # Persist image
                                    img_b64 = get_img_base64(path)
                                    res += f'<br><img src="data:image/png;base64,{img_b64}" width="100%" style="border-radius: 10px; margin: 10px 0;">'
                                    
                                    st.markdown(interp)
                                    res += f"\n- {interp}"
                                    tool_outputs.append(f"Plot Findings: {interp}")
                                else:
                                    st.error(f"❌ Plot generated but file not found at: {path}")
                                    st.caption("Common cause: Relative path issue. Try restarting the server.")
                            else:
                                st.error(f"❌ Server Error: {m_res}")
                        except Exception as e:
                            st.error(f"❌ Crash during plot execution: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

                elif t_name == "run_pls_analysis":
                    with st.spinner("Running PLS-DA..."):
                        m_res = asyncio.run(call_mcp_tool("run_pls_analysis", t_args))
                        if "|||" in m_res:
                            path, txt = m_res.split("|||")
                            st.image(path)
                            
                            # Persist image
                            img_b64 = get_img_base64(path)
                            res += f'<br><img src="data:image/png;base64,{img_b64}" width="100%" style="border-radius: 10px; margin: 10px 0;">'
                            
                            st.markdown(txt)
                            res += f"\n- PLS-DA Analysis complete."
                            tool_outputs.append(f"PLS-DA Findings: {txt}")

                elif t_name == "run_umap_analysis":
                    with st.spinner("Running UMAP..."):
                        m_res = asyncio.run(call_mcp_tool("run_umap_analysis", t_args))
                        if "|||" in m_res:
                            path, txt = m_res.split("|||")
                            st.image(path)
                            
                            # Persist image
                            img_b64 = get_img_base64(path)
                            res += f'<br><img src="data:image/png;base64,{img_b64}" width="100%" style="border-radius: 10px; margin: 10px 0;">'
                            
                            st.markdown(txt)
                            res += f"\n- UMAP Analysis complete."
                            tool_outputs.append(f"UMAP Findings: {txt}")

                elif t_name == "run_correlation_heatmap":
                    with st.spinner("Generating Heatmap..."):
                        m_res = asyncio.run(call_mcp_tool("run_correlation_heatmap", t_args))
                        if "|||" in m_res:
                            path, txt = m_res.split("|||")
                            st.image(path)
                            
                            # Persist image
                            img_b64 = get_img_base64(path)
                            res += f'<br><img src="data:image/png;base64,{img_b64}" width="100%" style="border-radius: 10px; margin: 10px 0;">'
                            
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
                            
                            # Persist image
                            img_b64 = get_img_base64(path)
                            res += f'<br><img src="data:image/png;base64,{img_b64}" width="100%" style="border-radius: 10px; margin: 10px 0;">'
                            
                            st.markdown(txt)
                            res += f"\n- Report generated."
                            tool_outputs.append(f"Report Summary: {txt}")
                
                elif t_name in ["rag", "query_medical_rag"]:
                    with st.spinner("Consulting internal clinical documentation..."):
                        m_res = asyncio.run(call_mcp_tool("query_medical_rag", {"question": last_user_prompt}))
                        with st.expander("🔍 Clinical Knowledge Base Records", expanded=False):
                            st.markdown(m_res)
                        res += f"\n- Medical records and clinical documentation retrieved."
                        tool_outputs.append(f"RAG Knowledge: {m_res}")

                elif t_name == "exact_identifier_search":
                    with st.spinner("Searching for specific patient identifiers..."):
                        m_res = asyncio.run(call_mcp_tool("exact_identifier_search", {"query": last_user_prompt, "patient_id_filter": patient_filter}))
                        with st.expander("🔍 Patient Identifier Search Results", expanded=False):
                            st.markdown(m_res) # This will render the markdown + code blocks
                        res += f"\n- Precise clinical data points identified."
                        tool_outputs.append(f"Exact Search Results: {m_res}")

                elif t_name in ["describe", "get_data_context"]:
                    with st.spinner("Analyzing patient data context..."):
                        m_res = asyncio.run(call_mcp_tool("get_data_context", {}))
                        with st.expander("🔍 Detailed Data Schema Overview", expanded=False):
                            st.markdown(m_res)
                        res += f"\n- Data Summary Loaded."
                        tool_outputs.append(f"Data Summary: {m_res}")

                elif t_name == "query_exprag_hybrid":
                    with st.spinner("Finding similar cases and SOPs..."):
                        # Handle potential JSON string from LLM
                        p_data = t_args.get("patient_data", {})
                        if isinstance(p_data, str):
                            # Already a string, use as is (try to validate if valid JSON first?)
                            try:
                                json.loads(p_data) # Check validity
                                p_data_str = p_data
                            except:
                                p_data_str = "{}"
                        else:
                            # Is a dict, dump to string
                            p_data_str = json.dumps(p_data)

                        exprag_args = {"question": last_user_prompt, "patient_data": p_data_str}
                        
                        # DEBUG
                        print(f"📡 Calling EXPRAG with: {exprag_args}")
                        
                        m_res = asyncio.run(call_mcp_tool("query_exprag_hybrid", exprag_args))
                        
                        # DEBUG
                        print(f"📥 EXPRAG Raw Response: {m_res[:200]}...")

                        try:
                            hybrid_result = json.loads(m_res)
                            if "error" in hybrid_result:
                                st.error(f"❌ EXPRAG Error: {hybrid_result['error']}")
                            else:
                                with st.expander("🔍 Similar Patient Cohort", expanded=False):
                                    st.write(f"**Similar Cases:** {hybrid_result.get('cohort_ids', [])}")
                                    st.json(hybrid_result.get('profile', {}))
                                st.markdown(hybrid_result.get('answer', ''))
                                res += f"\n- Hybrid EXPRAG analysis complete."
                                tool_outputs.append(f"EXPRAG: {hybrid_result.get('answer', '')}")
                        except json.JSONDecodeError:
                            st.error(f"❌ Failed to parse EXPRAG result: {m_res}")

                else:
                    st.warning(f"⚠️ Unrecognized tool: **{t_name}**")
                    res += f"\n- Skipped unrecognized tool: {t_name}"

            # Final Step: Clinical Synthesis
            if tool_outputs:
                with st.spinner("Dr. AI is analyzing the results..."):
                    combined_findings = "\n".join(tool_outputs)
                    synth_res = asyncio.run(call_mcp_tool("synthesize_medical_results", {
                        "question": last_user_prompt,
                        "results": combined_findings,
                        "rag_context": st.session_state.get("current_rag_context", "")
                    }))
                    st.markdown("### 📝 Clinical Synthesis")
                    st.info(synth_res)
                    res += f"\n\n### Clinical Synthesis\n{synth_res}"
        else:
            st.markdown(answer)
            res = answer
        
        st.markdown('</div></div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": res})
        st.rerun()
