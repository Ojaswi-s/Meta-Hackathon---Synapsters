"""
conftest.py — pytest root configuration.
Auto-adds the project root to sys.path so that `from meeting_env import ...`
works during pytest without needing `pip install -e .` first.
"""
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
