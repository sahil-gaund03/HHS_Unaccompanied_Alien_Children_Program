"""
app.py — Top-level Streamlit entry point
==========================================
Launch with: streamlit run app.py
Delegates to dashboard/app.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dashboard.app import main

if __name__ == "__main__":
    main()
