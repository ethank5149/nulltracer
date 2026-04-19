"""Shared test configuration."""

import sys
import os

# Add server directory to path so tests can import server modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
