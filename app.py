import os
import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai
from google.genai import types
from engine import PPAEngine

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SiliconArch: PPA & SoC Advisor", layout="wide", page_icon="⚡")

# --- 1. INITIALIZE & CACHE ML ENGINE ---
# We use @st.cache_resource so the RandomForest model only trains once when the app starts
@st.cache_resource
def load_engine():
    engine = PPAEngine()
    engine.train()
    return engine

engine = load_engine()

# --- 2. SIDEBAR CONFIGURATION ---
st.sidebar.title("⚙️ Architecture Setup")

api_key = st.sidebar.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
model_name = st.sidebar.selectbox("Gemini Model", ["gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-1.5-flash"])

st.sidebar.markdown("---")
st.sidebar.subheader("Target Envelope Constraints")
max_pwr = st.sidebar.slider("Max Power Budget (W)", 2.0, 150.0, 25.0, 1.0)
max_ar = st.sidebar.slider("Max Die Area (mm²)", 10.0, 350.0, 80.0, 5.0)
min_prf = st.sidebar.slider("Min Performance Score", 200.0, 8000.0, 1500.0, 100.0)

# Apply user constraints to find optimal designs
pareto_df = engine.find_pareto_designs(max_power=max_pwr, max_area=max_ar, min_perf=min_prf)

# --- 3. DASHBOARD METRICS & 3D VISUALIZATION ---
st.title("⚡ SiliconArch: PPA & SoC Trade-off Advisor")
st.caption("AI-assisted Design Space Exploration & Pareto Optimization Platform")

# Top-level metric cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Explored Designs", len(engine.dataset))
col2.metric("Pareto Optimal Designs", len(pareto_df))
col3.metric("Power Ceiling", f"{max_pwr} W")
col4.metric("Area Budget", f"{max_ar} mm²")

# Data Visualization Tabs
tab_viz, tab_data = st.tabs(["📊 3D Design Space & Trade-offs", "📋 Pareto Frontier Table"])

with tab_viz:
    if not pareto_df.empty:
        # Clone dataset to mark which points are Pareto Optimal
        full_df = engine.dataset.copy()
        full_df["Classification"] = "Sub-optimal"
        full_df.loc[full_df.index.isin(pareto_df.index), "Classification"] = "Pareto Optimal"
        
        # Render a 3D Plotly Scatter Plot
        fig = px.scatter_3d(
            full_df,
            x="area_mm2",
            y="power_w",
            z="perf_score",
            color="Classification",
            color_discrete_map={"Sub-optimal": "#475569", "Pareto Optimal": "#06b6d4"},
            hover_data=["node_nm", "cores", "freq_ghz", "l3_cache_mb", "npu_tops"],
            title="SoC Design Space: Area (mm²) vs Power (W) vs Performance",
            labels={"area_mm2": "Die Area (mm²)", "power_w": "Power (W)", "perf_score": "Performance"},
            opacity=0.7
        )
        fig.update_layout(height=520, margin=dict(l=0, r=0, b=0, t=30))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No designs found meeting these constraints. Try increasing power or area limits.")

with tab_data:
    if not pareto_df.empty:
        # Display the data frame
        st.dataframe(pareto_df, use_container_width=True)
        
        # Added Enterprise CSV Export Functionality
        csv_data = pareto_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Pareto Frontier (CSV)",
            data=csv_data,
            file_name='pareto_optimal_soc_designs.csv',
            mime='text/csv',
        )
    else:
        st.write("No matching configurations.")

# --- 4. AGENTIC AI ADVISOR WITH TOOL CALLING ---
st.markdown("---")
st.subheader("💬 AI Silicon Architect Advisor")

# 4a. Define the Python tools the AI can execute
def simulate_specific_soc(node_nm: int, cores: int, freq_ghz: float, l3_cache_mb: int, npu_tops: int) -> dict:
    """Predicts power (W), area (mm2), and performance score for a custom SoC configuration using the trained ML model."""
    return engine.predict(node_nm, cores, freq_ghz, l3_cache_mb, npu_tops)

def query_pareto_designs(max_power_w: float, max_area_mm2: float, min_perf_score: float = 0.0) -> list:
    """Finds non-dominated Pareto-optimal SoC architectures matching power, area, and performance constraints."""
    df = engine.find_pareto_designs(max_power=max_power_w, max_area=max_area_mm2, min_perf=min_perf_score)
    return df.head(5).to_dict(orient="records")

tools_list = [simulate_specific_soc, query_pareto_designs]

# 4b. Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome. I am your Senior Silicon Architect. You can ask me to evaluate trade-offs, compare process nodes (3nm vs 7nm), or find optimal configurations for your thermal and area envelope."}
    ]

# Render previous conversation blocks
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4c. Chat Input and Execution Logic
user_input = st.chat_input("Ask about architecture trade-offs or request a specific design...")

if user_input:
    # Print the user's message to the UI
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Halt if no API key is provided
    if not api_key:
        st.warning("Please provide your Gemini API Key in the sidebar to chat with the advisor.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Executing simulation tools & analyzing trade-offs..."):
                try:
                    # Initialize the new standard SDK client
                    client = genai.Client(api_key=api_key)
                    
                    system_instruction = """
                    You are a Principal Silicon Architect and SoC Planning Advisor. 
                    Your job is to provide rigorous, authoritative technical advice on Semiconductor PPA (Power, Performance, Area) trade-offs.
                    You have access to simulation tools:
                    1. 'simulate_specific_soc': To compute metrics for any architectural combination.
                    2. 'query_pareto_designs': To search the Pareto frontier for optimal setups.
                    Always cite realistic physics rationale (e.g., dynamic CV^2f scaling, sub-threshold leakage, SRAM area footprint, Amdahl's law) when explaining trade-offs.
                    """

                    config = types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        tools=tools_list,
                        temperature=0.2
                    )

                    # Reconstruct past text history so the model remembers context
                    history = []
                    for msg in st.session_state.messages[:-1]: 
                        role = "user" if msg["role"] == "user" else "model"
                        history.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
                    
                    # Create the chat session
                    chat = client.chats.create(
                        model=model_name,
                        config=config,
                        history=history
                    )
                    
                    # Send the new prompt. The SDK will automatically detect and execute function calls if needed.
                    response = chat.send_message(user_input)
                    reply_text = response.text or "Analysis completed based on the retrieved telemetry."
                    
                    # Print and save the AI's response
                    st.markdown(reply_text)
                    st.session_state.messages.append({"role": "assistant", "content": reply_text})
                    
                except Exception as e:
                    st.error(f"Execution Error: {e}")