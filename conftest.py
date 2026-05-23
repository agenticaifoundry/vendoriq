"""
VendorIQ Test Configuration
============================
This file is automatically loaded by pytest before any tests run.
It adds the project root to the Python path so all modules are importable.
"""

import sys
import os

# Add the project root to Python path
# This allows tests to import from agents/, rag/, graph/, etc.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
