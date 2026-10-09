import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google import genai
from google.genai import types
from engine import PPAEngine

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SiliconArch: PPA & SoC Advisor", layout="wide", page_icon="⚡")

# --- 1. INITIALIZE & CACHE ML ENGINE ---
@st.cache_resource
def load_engine():
    engine = PPAEngine()
    engine.train()
    return engine

engine = load_engine()

# --- 2. SIDEBAR CONFIGURATION & APPLICATION PRESETS ---
st.sidebar.title("⚙️ Architecture Setup")

api_key = st.sidebar.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
model_name = st.sidebar.selectbox("Gemini Model", ["gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-1.5-flash"])

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Application-Specific Presets")

# Hardware domain presets matching industry standard constraints
PRESETS = {
    "Custom / Manual": None,
    "Ultra-Low Power IoT & MCU": {"power": 3.0, "area": 15.0, "perf": 300.0},
    "Wearable & Smartwatch": {"power": 8.0, "area": 30.0, "perf": 1200.0},
    "Edge AI & Embedded Vision": {"power": 16.0, "area": 55.0, "perf": 3200.0},
    "High-Performance Mobile": {"power": 25.0, "area": 80.0, "perf": 5000.0},
    "Compute Server / Automotive": {"power": 95.0, "area": 240.0, "perf": 7500.0},
}

# Initialize session state bounds if not set
if "max_pwr" not in st.session_state:
    st.session_state.max_pwr = 25.0
if "max_ar" not in st.session_state:
    st.session_state.max_ar = 80.0
if "min_prf" not in st.session_state:
    st.session_state.min_prf = 1500.0

def update_preset():
    selected = st.session_state.preset_selection
    if selected in PRESETS and PRESETS[selected] is not None:
        st.session_state.max_pwr = float(PRESETS[selected]["power"])
        st.session_state.max_pwr = float(PRESETS[selected]["power"])
        st.session_state.max_ar = float(PRESETS[selected]["area"])
        st.session_state.min_prf = float(PRESETS[selected]["perf"])

st.sidebar.selectbox(
    "Target Domain",
    list(PRESETS.keys()),
    index=0,
    key="preset_selection",
    on_change=update_preset,
    help="Instantly configure realistic physical envelopes for specific hardware domains."
)

st.sidebar.subheader("Target Envelope Constraints")
max_pwr = st.sidebar.slider("Max Power Budget (W)", 2.0, 150.0, step=1.0, key="max_pwr")
max_ar = st.sidebar.slider("Max Die Area (mm²)", 10.0, 350.0, step=5.0, key="max_ar")
min_prf = st.sidebar.slider("Min Performance Score", 200.0, 8000.0, step=100.0, key="min_prf")

# Filter Pareto designs based on active constraints
pareto_df = engine.find_pareto_designs(max_power=max_pwr, max_area=max_ar, min_perf=min_prf)

# --- 3. DASHBOARD METRICS & VISUALIZATIONS ---
st.title("⚡ SiliconArch: PPA & SoC Trade-off Advisor")
st.caption("AI-assisted Design Space Exploration & Pareto Optimization Platform")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Explored Designs", len(engine.dataset))
col2.metric("Pareto Optimal Designs", len(pareto_df))
col3.metric("Power Ceiling", f"{max_pwr} W")
col4.metric("Area Budget", f"{max_ar} mm²")

tab_viz, tab_data, tab_compare = st.tabs(["📊 3D Design Space & Trade-offs", "📋 Pareto Frontier Table", "⚖️ Compare Chips"])

with tab_viz:
    if not pareto_df.empty:
        full_df = engine.dataset.copy()
        full_df["Classification"] = "Sub-optimal"
        full_df.loc[full_df.index.isin(pareto_df.index), "Classification"] = "Pareto Optimal"
        
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
        st.dataframe(pareto_df, use_container_width=True)
        
        csv_data = pareto_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Pareto Frontier (CSV)",
            data=csv_data,
            file_name='pareto_optimal_soc_designs.csv',
            mime='text/csv',
        )
    else:
        st.write("No matching configurations.")

with tab_compare:
    st.subheader("⚖️ Head-to-Head Architecture Comparison")
    
    if len(pareto_df) >= 2:
        compare_df = pareto_df.copy()
        compare_df["Chip_Name"] = (
            "Node: " + compare_df["node_nm"].astype(str) + "nm | Cores: " + 
            compare_df["cores"].astype(str) + " | Freq: " + 
            compare_df["freq_ghz"].astype(str) + "GHz"
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            chip_a_name = st.selectbox("Select Chip Option A", compare_df["Chip_Name"], key="compare_a")
        with col_b:
            chip_b_name = st.selectbox("Select Chip Option B", compare_df["Chip_Name"], index=1, key="compare_b")
            
        chip_a = compare_df[compare_df["Chip_Name"] == chip_a_name].iloc[0]
        chip_b = compare_df[compare_df["Chip_Name"] == chip_b_name].iloc[0]
        
        # Scoring normalization out of 100%
        max_perf = max(chip_a['perf_score'], chip_b['perf_score'])
        perf_a = chip_a['perf_score'] / max_perf
        perf_b = chip_b['perf_score'] / max_perf
        
        min_pwr = min(chip_a['power_w'], chip_b['power_w'])
        pwr_a = min_pwr / chip_a['power_w']
        pwr_b = min_pwr / chip_b['power_w']
        
        min_area = min(chip_a['area_mm2'], chip_b['area_mm2'])
        area_a = min_area / chip_a['area_mm2']
        area_b = min_area / chip_b['area_mm2']
        
        categories = ['Performance', 'Power Efficiency', 'Area Efficiency']
        stats_a = [perf_a, pwr_a, area_a, perf_a] 
        stats_b = [perf_b, pwr_b, area_b, perf_b]
        cats = categories + [categories[0]]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=stats_a, theta=cats, fill='toself', name='Option A', marker=dict(color='#06b6d4')))
        fig.add_trace(go.Scatterpolar(r=stats_b, theta=cats, fill='toself', name='Option B', marker=dict(color='#f43f5e')))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
            showlegend=True,
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Widen your constraints to provide at least 2 non-dominated designs for head-to-head comparison.")

# --- 4. AGENTIC AI ADVISOR WITH TOOL CALLING ---
st.markdown("---")
st.subheader("💬 AI Silicon Architect Advisor")

def simulate_specific_soc(node_nm: int, cores: int, freq_ghz: float, l3_cache_mb: int, npu_tops: int) -> dict:
    """Predicts power (W), area (mm2), and performance score for a custom SoC configuration using the trained ML model."""
    return engine.predict(node_nm, cores, freq_ghz, l3_cache_mb, npu_tops)

def query_pareto_designs(max_power_w: float, max_area_mm2: float, min_perf_score: float = 0.0) -> list:
    """Finds non-dominated Pareto-optimal SoC architectures matching power, area, and performance constraints."""
    df = engine.find_pareto_designs(max_power=max_power_w, max_area=max_area_mm2, min_perf=min_perf_score)
    return df.head(5).to_dict(orient="records")

def goal_seek_architecture(target_perf: float, max_power_w: float, max_area_mm2: float) -> dict:
    """Reverse-engineers a hypothetical chip design to hit a specific performance target without exceeding power and area limits."""
    best_design = None
    
    # The AI tests realistic hardware combinations to find a valid blueprint
    nodes = [3, 5, 7, 10] 
    core_options = [4, 8, 12, 16, 24, 32]
    freq_options = [2.0, 2.5, 3.0, 3.5, 4.0]
    npu_options = [0, 20, 45, 80]
    
    for node in nodes:
        for cores in core_options:
            for freq in freq_options:
                for npu in npu_options:
                    # Test this hypothetical combination using your ML engine
                    metrics = engine.predict(node, cores, freq, 16, npu) # standardizing cache to 16MB
                    
                    # Check if it hits the exact business goals
                    if (metrics['perf_score'] >= target_perf and 
                        metrics['power_w'] <= max_power_w and 
                        metrics['area_mm2'] <= max_area_mm2):
                        
                        # Save the design that uses the least amount of power
                        if best_design is None or metrics['power_w'] < best_design['metrics']['power_w']:
                            best_design = {
                                "specs": {"node_nm": node, "cores": cores, "freq_ghz": freq, "l3_cache_mb": 16, "npu_tops": npu},
                                "metrics": metrics
                            }
                            
    if best_design:
        return best_design
    else:
        return {"error": "Scientific limit reached. No physical silicon architecture can achieve this performance target within these thermal and area constraints."}

# Register all three tools
tools_list = [simulate_specific_soc, query_pareto_designs, goal_seek_architecture]

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome. I am your Senior Silicon Architect. You can select an application preset on the sidebar, query pareto trade-offs, or ask me to goal-seek a completely custom architecture to hit your specific performance target."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about architecture trade-offs or request a specific design...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if not api_key:
        st.warning("Please provide your Gemini API Key in the sidebar to chat with the advisor.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Executing simulation tools & analyzing trade-offs..."):
                try:
                    client = genai.Client(api_key=api_key)
                    
                    system_instruction = """
                    You are a Principal Silicon Architect and SoC Planning Advisor. 
                    Your job is to provide rigorous, authoritative technical advice on Semiconductor PPA (Power, Performance, Area) trade-offs.
                    You have access to simulation tools:
                    1. 'simulate_specific_soc': To compute metrics for any architectural combination.
                    2. 'query_pareto_designs': To search the Pareto frontier for optimal setups.
                    3. 'goal_seek_architecture': To reverse-engineer an architecture based on strict business goals.
                    Always cite realistic physics rationale (dynamic CV^2f scaling, sub-threshold leakage, SRAM area footprint, Amdahl's law) when explaining trade-offs.
                    """

                    config = types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        tools=tools_list,
                        temperature=0.2
                    )

                    history = []
                    for msg in st.session_state.messages[:-1]: 
                        role = "user" if msg["role"] == "user" else "model"
                        history.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
                    
                    chat = client.chats.create(
                        model=model_name,
                        config=config,
                        history=history
                    )
                    
                    response = chat.send_message(user_input)
                    reply_text = response.text or "Analysis completed based on the retrieved telemetry."
                    
                    st.markdown(reply_text)
                    st.session_state.messages.append({"role": "assistant", "content": reply_text})
                    
                except Exception as e:
                    st.error(f"Execution Error: {e}")
