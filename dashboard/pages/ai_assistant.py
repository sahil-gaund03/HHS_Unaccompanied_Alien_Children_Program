"""
dashboard/pages/ai_assistant.py
===============================
AI Policy Copilot page.
An interactive chat assistant powered by Gemini 1.5 Flash, pre-loaded with
the UAC program dataset and policy reports for answering operational and strategic questions.
"""

import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from pathlib import Path
from dashboard.components.theme import render_page_header

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_api_key():
    """Retrieve the Gemini API key from session state, environment, or Streamlit secrets."""
    if "gemini_api_key" in st.session_state and st.session_state.gemini_api_key:
        return st.session_state.gemini_api_key

    # Check env var
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        st.session_state.gemini_api_key = env_key
        return env_key

    # Check streamlit secrets
    try:
        sec_key = st.secrets.get("GEMINI_API_KEY")
        if sec_key:
            st.session_state.gemini_api_key = sec_key
            return sec_key
    except Exception:
        pass

    # Check local secrets.toml manually (robust fallback)
    try:
        secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            with open(secrets_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "=" in line:
                        parts = line.split("=", 1)
                        key_name = parts[0].strip()
                        if key_name == "GEMINI_API_KEY":
                            val = parts[1].strip().strip('"').strip("'")
                            st.session_state.gemini_api_key = val
                            return val
    except Exception:
        pass

    return None


@st.cache_data
def load_context_data():
    """Load cleaned dataset and generated reports to inject into the LLM system prompt."""
    # Load Cleaned Data
    cleaned_csv_path = PROJECT_ROOT / "data" / "cleaned_data.csv"
    cleaned_csv_content = ""
    if cleaned_csv_path.exists():
        try:
            with open(cleaned_csv_path, "r", encoding="utf-8") as f:
                cleaned_csv_content = f.read()
        except Exception as e:
            cleaned_csv_content = f"Error loading CSV data: {str(e)}"

    # Load Executive Summary
    exec_path = PROJECT_ROOT / "reports" / "executive_summary.md"
    exec_content = ""
    if exec_path.exists():
        try:
            with open(exec_path, "r", encoding="utf-8") as f:
                exec_content = f.read()
        except Exception as e:
            exec_content = f"Error loading executive summary: {str(e)}"

    # Load Policy Recommendations
    policy_path = PROJECT_ROOT / "reports" / "policy_recommendations.md"
    policy_content = ""
    if policy_path.exists():
        try:
            with open(policy_path, "r", encoding="utf-8") as f:
                policy_content = f.read()
        except Exception as e:
            policy_content = f"Error loading policy recommendations: {str(e)}"

    return cleaned_csv_content, exec_content, policy_content


def render(df: pd.DataFrame):
    """Render the AI Policy Copilot page."""
    render_page_header(
        "🤖 AI Policy Copilot",
        "Operational intelligence assistant. Query trends, bottlenecks, predictions, and policy recommendations."
    )

    # Fetch context data
    cleaned_csv_content, exec_content, policy_content = load_context_data()

    # Sidebar: API Key Configuration
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔑 Gemini Configuration")

    api_key = get_api_key()
    selected_model_name = "gemini-pro"

    if not api_key:
        st.sidebar.warning("⚠️ No Gemini API Key found.")
        input_key = st.sidebar.text_input(
            "Enter Gemini API Key",
            type="password",
            placeholder="AIzaSy...",
            help="Get an API key from Google AI Studio",
        )
        if input_key:
            st.session_state.gemini_api_key = input_key
            st.sidebar.success("API Key saved for session!")
            st.rerun()
    else:
        st.sidebar.success("✅ Gemini API Connected")
        model_options = [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-pro",
            "gemini-1.0-pro"
        ]
        selected_model_name = st.sidebar.selectbox(
            "Select Model",
            options=model_options,
            index=2,
            help="If you get a 404 error (e.g. key has model restrictions), select 'gemini-pro'."
        )
        if st.sidebar.button("Reset API Key"):
            st.session_state.gemini_api_key = None
            if "chat_history" in st.session_state:
                del st.session_state.chat_history
            if "chat_session" in st.session_state:
                del st.session_state.chat_session
            st.rerun()

    # System instructions construct
    system_instruction = f"""You are the UAC Operational Intelligence Policy Copilot, a Senior AI Architect, Principal Data Scientist, and Government Analytics Expert for the HHS Unaccompanied Alien Children (UAC) Program.

Your purpose is to assist policymakers, operational managers, and analysts in understanding the pipeline from CBP custody to HHS placement and discharge. You have full visibility into the dataset, model performances, SHAP explainability insights, and policy recommendations.

Here is the operational dataset (Jan 2023 - Dec 2025):
---
Columns format: date, apprehended, cbp_custody, transferred_out, hhs_care, discharged
{cleaned_csv_content}
---

Here is the Executive Summary of the predictive modeling, SHAP explainability, and anomaly detection results:
---
{exec_content}
---

Here is the current Policy Recommendations report:
---
{policy_content}
---

Operational Rules & Domain Context:
1. Apprehension to Discharge pipeline:
   - Apprehended: Children apprehended and placed in CBP custody.
   - CBP Custody: Current custody load at CBP.
   - Transferred Out: Children transferred out of CBP custody (transitioning to HHS).
   - HHS Care: Current occupancy of children in HHS shelter network.
   - Discharged: Children discharged from HHS care (mostly reunified with sponsors).
2. Major Trend: The operational volumes saw a significant drop starting in mid-2024 and persisting through 2025 (HHS Care dropped from ~11k to ~2k).
3. Modeling: Stacking ensemble regression achieves R2 score of 0.8127, using base models: XGBoost, Gradient Boosting, Random Forest, Extra Trees, AdaBoost, with a Ridge meta-regressor.
4. Bottlenecks: A key metric is transfer efficiency (transferred_out / cbp_custody) and discharge effectiveness (discharged / hhs_care). 

Guidelines for your responses:
- Be highly professional, data-driven, objective, and policy-informed.
- Do not make up data. Use the provided CSV and summary stats.
- When performing analysis (e.g., averages, sums, trend comparisons), compute them accurately using the provided CSV.
- For date ranges, make sure to cross-reference them accurately (e.g., Jan 2025 saw lower numbers than Jan 2023).
- Format all equations, comparisons, lists, and numbers clearly in Markdown tables or lists.
- Proactively offer insights on bottlenecks, efficiency improvements, or risk factors based on the user's question.
"""

    # Stop page render if key is missing
    if not api_key:
        st.info("💡 **Welcome to the AI Policy Copilot!** Please configure your Google Gemini API key in the sidebar to begin querying the operational intelligence agent.")
        
        # Display sample dashboard metrics to show the page is alive
        st.markdown("### 📊 What you can ask the Copilot")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            * **Operational Performance:** "How did the average discharge effectiveness change between 2023, 2024, and 2025?"
            * **Bottleneck Analysis:** "What are the main causes of custody surges and backlog accumulation?"
            """)
        with col2:
            st.markdown("""
            * **ML Model Performance:** "Explain the stacking ensemble's performance compared to the individual models."
            * **Explainability:** "Which features are most predictive of discharge volume based on SHAP values?"
            """)
        return

    # Initialize Gemini Client
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=selected_model_name,
            system_instruction=system_instruction
        )
    except Exception as e:
        st.error(f"Failed to configure Gemini client: {e}")
        return

    # Chat history state initialization
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_session" not in st.session_state or st.session_state.get("current_model") != selected_model_name:
        # Start a new chat session with the model
        st.session_state.chat_session = model.start_chat(history=[])
        st.session_state.current_model = selected_model_name

    # Suggestion Chips / Buttons
    st.markdown("### 💡 Recommended Queries")
    suggestions = [
        "What are the key policy recommendations from the reports?",
        "Explain the main bottlenecks in the UAC care transition pipeline.",
        "What is the average discharge effectiveness for 2025 compared to 2023?",
        "Explain the performance of the predictive stacking ensemble model.",
    ]

    cols = st.columns(4)
    suggested_query = None
    for idx, sug in enumerate(suggestions):
        with cols[idx]:
            if st.button(sug, key=f"sug_{idx}", use_container_width=True):
                suggested_query = sug

    st.markdown("---")

    # Render past chat history
    for message in st.session_state.chat_history:
        avatar = "🤖" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Chat input and processing
    user_input = st.chat_input("Ask the UAC Policy Copilot...")

    # If suggested query was clicked, override user_input
    if suggested_query:
        user_input = suggested_query

    if user_input:
        # Render user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Generate model response
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            with st.spinner("Analyzing operational context..."):
                try:
                    chat = st.session_state.chat_session
                    response = chat.send_message(user_input)
                    assistant_response = response.text
                    response_placeholder.markdown(assistant_response)
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": assistant_response
                    })
                except Exception as e:
                    error_msg = f"⚠️ Gemini API Error: {str(e)}\n\nPlease verify your API key and network connection."
                    response_placeholder.error(error_msg)
                    
    # Add a clear chat button in the sidebar
    if st.sidebar.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        # Restart chat session
        st.session_state.chat_session = model.start_chat(history=[])
        st.rerun()
