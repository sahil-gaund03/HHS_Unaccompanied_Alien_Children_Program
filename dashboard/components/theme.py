"""
dashboard/components/theme.py
=============================
Premium dark theme with glassmorphism, government-grade aesthetics,
animated KPI panels, and professional typography.
"""

import streamlit as st


def apply_theme():
    """Inject custom CSS for the premium dark dashboard theme."""
    st.markdown("""
    <style>
    /* ============================================================
       IMPORTS — Premium Typography
       ============================================================ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ============================================================
       ROOT VARIABLES — Government Analytics Palette
       ============================================================ */
    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #111827;
        --bg-card: rgba(17, 24, 39, 0.8);
        --bg-glass: rgba(255, 255, 255, 0.03);
        --border-glass: rgba(255, 255, 255, 0.08);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-teal: #06b6d4;
        --accent-blue: #3b82f6;
        --accent-indigo: #6366f1;
        --accent-emerald: #10b981;
        --accent-amber: #f59e0b;
        --accent-rose: #f43f5e;
        --gradient-primary: linear-gradient(135deg, #06b6d4, #3b82f6);
        --gradient-success: linear-gradient(135deg, #10b981, #06b6d4);
        --gradient-warning: linear-gradient(135deg, #f59e0b, #f97316);
        --gradient-danger: linear-gradient(135deg, #f43f5e, #e11d48);
        --shadow-glow: 0 0 30px rgba(6, 182, 212, 0.15);
        --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.3);
        --radius: 16px;
        --radius-sm: 8px;
    }

    /* ============================================================
       GLOBAL STYLES
       ============================================================ */
    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-glass) !important;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary) !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ============================================================
       GLASSMORPHISM CARD
       ============================================================ */
    .glass-card {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius);
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: var(--shadow-card);
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(6, 182, 212, 0.3);
        box-shadow: var(--shadow-glow);
        transform: translateY(-2px);
    }

    /* ============================================================
       KPI CARDS
       ============================================================ */
    .kpi-card {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius);
        padding: 20px 24px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-primary);
        border-radius: var(--radius) var(--radius) 0 0;
    }

    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-glow);
        border-color: rgba(6, 182, 212, 0.4);
    }

    .kpi-card.success::before { background: var(--gradient-success); }
    .kpi-card.warning::before { background: var(--gradient-warning); }
    .kpi-card.danger::before { background: var(--gradient-danger); }

    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1.1;
        margin: 8px 0;
        letter-spacing: -0.02em;
    }

    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 4px;
    }

    .kpi-delta {
        font-size: 0.85rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
        display: inline-block;
        margin-top: 6px;
    }

    .kpi-delta.positive {
        color: #10b981;
        background: rgba(16, 185, 129, 0.12);
    }

    .kpi-delta.negative {
        color: #f43f5e;
        background: rgba(244, 63, 94, 0.12);
    }

    .kpi-delta.neutral {
        color: var(--text-secondary);
        background: rgba(148, 163, 184, 0.12);
    }

    /* ============================================================
       SECTION HEADERS
       ============================================================ */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid var(--border-glass);
        letter-spacing: -0.01em;
    }

    .section-subheader {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-secondary);
        margin: 16px 0 12px 0;
    }

    /* ============================================================
       PAGE TITLE
       ============================================================ */
    .page-title {
        font-size: 2rem;
        font-weight: 800;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }

    .page-subtitle {
        font-size: 0.95rem;
        color: var(--text-muted);
        margin-bottom: 24px;
        font-weight: 400;
    }

    /* ============================================================
       SIDEBAR BRANDING
       ============================================================ */
    .sidebar-brand {
        text-align: center;
        padding: 20px 0 24px 0;
        border-bottom: 1px solid var(--border-glass);
        margin-bottom: 20px;
    }

    .sidebar-brand h1 {
        font-size: 1.3rem;
        font-weight: 800;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.01em;
    }

    .sidebar-brand p {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin: 4px 0 0 0;
    }

    /* ============================================================
       METRICS TABLE
       ============================================================ */
    .metrics-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: var(--radius-sm);
        overflow: hidden;
        font-size: 0.85rem;
    }

    .metrics-table th {
        background: rgba(6, 182, 212, 0.1);
        color: var(--accent-teal);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 12px 16px;
        text-align: left;
        border-bottom: 1px solid var(--border-glass);
    }

    .metrics-table td {
        padding: 10px 16px;
        color: var(--text-primary);
        border-bottom: 1px solid var(--border-glass);
    }

    .metrics-table tr:hover td {
        background: rgba(255, 255, 255, 0.02);
    }

    /* ============================================================
       STATUS BADGES
       ============================================================ */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge-success { background: rgba(16, 185, 129, 0.15); color: #10b981; }
    .badge-warning { background: rgba(245, 158, 11, 0.15); color: #f59e0b; }
    .badge-danger { background: rgba(244, 63, 94, 0.15); color: #f43f5e; }
    .badge-info { background: rgba(59, 130, 246, 0.15); color: #3b82f6; }

    /* ============================================================
       PLOTLY CHART CONTAINERS
       ============================================================ */
    .stPlotlyChart {
        border-radius: var(--radius) !important;
        overflow: hidden;
    }

    /* ============================================================
       ANIMATIONS
       ============================================================ */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .animate-in {
        animation: fadeInUp 0.5s ease forwards;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    .pulse { animation: pulse 2s ease-in-out infinite; }

    /* ============================================================
       STREAMLIT COMPONENT OVERRIDES
       ============================================================ */
    .stSelectbox > div > div {
        background: var(--bg-secondary) !important;
        border-color: var(--border-glass) !important;
        color: var(--text-primary) !important;
    }

    .stDateInput > div > div > input {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }

    .stButton > button {
        background: var(--gradient-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        padding: 8px 24px !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.3) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--bg-glass);
        border-radius: var(--radius-sm);
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: var(--text-secondary);
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(6, 182, 212, 0.15) !important;
        color: var(--accent-teal) !important;
    }

    div[data-testid="stMetric"] {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-sm);
        padding: 16px;
    }

    /* ============================================================
       SIDEBAR RADIO NAVIGATION STYLING
       ============================================================ */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 6px;
        background: transparent;
    }
    
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: var(--radius-sm) !important;
        padding: 10px 14px !important;
        margin: 2px 0 !important;
        color: var(--text-secondary) !important;
        transition: all 0.25s ease !important;
        cursor: pointer !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
    }

    /* Hide the default circular radio selector button */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    /* Target the text container inside the label */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p {
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }

    /* Hover State */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background: rgba(6, 182, 212, 0.08) !important;
        border-color: rgba(6, 182, 212, 0.3) !important;
        color: var(--text-primary) !important;
        transform: translateX(4px);
    }

    /* Selected State */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        background: rgba(6, 182, 212, 0.15) !important;
        border-color: var(--accent-teal) !important;
        color: var(--accent-teal) !important;
        box-shadow: 0 0 12px rgba(6, 182, 212, 0.2) !important;
    }
    
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
        font-weight: 700 !important;
        color: #06b6d4 !important;
    }

    /* ============================================================
       SIDEBAR LOGO
       ============================================================ */
    .sidebar-logo {
        width: 85px;
        height: 85px;
        border-radius: 50%;
        object-fit: cover;
        margin: 0 auto 12px auto;
        display: block;
        border: 2px solid var(--accent-teal);
        box-shadow: 0 0 20px rgba(6, 182, 212, 0.4);
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .sidebar-logo:hover {
        transform: rotate(5deg) scale(1.08);
    }
    </style>
    """, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str = ""):
    """Render a styled page header."""
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_sidebar_brand():
    """Render the sidebar branding section with base64 encoded logo."""
    import base64
    from pathlib import Path
    
    # Resolve the path to assets/logo.png
    logo_path = Path(__file__).resolve().parents[2] / "assets" / "logo.png"
    logo_base64 = None
    
    if logo_path.exists():
        try:
            with open(logo_path, "rb") as f:
                logo_base64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            pass
            
    if logo_base64:
        st.sidebar.markdown(f"""
        <div class="sidebar-brand">
            <img src="data:image/png;base64,{logo_base64}" class="sidebar-logo" alt="UAC Analytics Logo">
            <h1>🏛️ UAC Analytics</h1>
            <p>Operational Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="sidebar-brand">
            <div style="font-size: 3rem; margin-bottom: 8px;">🏛️</div>
            <h1>🏛️ UAC Analytics</h1>
            <p>Operational Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)
