"""Cross-platform launcher: starts the xrdkit Streamlit GUI.

Usage:
    python launch.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

needed = ["streamlit", "pymatgen", "matplotlib", "numpy", "requests",
          "PIL", "pandas"]
missing = []
for m in needed:
    try:
        __import__(m)
    except ImportError:
        missing.append(m)

if missing:
    print("Missing Python packages:", ", ".join(missing))
    print()
    print("Install everything with:")
    print(f"    {sys.executable} -m pip install -r {HERE / 'requirements.txt'}")
    print("Or, if you have cloned the repo:")
    print(f"    {sys.executable} -m pip install -e {HERE}")
    sys.exit(1)

streamlit = shutil.which("streamlit")
if streamlit is None:
    cmd = [sys.executable, "-m", "streamlit"]
else:
    cmd = [streamlit]

port = os.environ.get("XRDKIT_PORT", "8501")
print()
print(f"  Starting xrdkit on http://localhost:{port}")
print("  Your browser should open automatically; close this window to stop.")
print()
subprocess.run(cmd + ["run", str(HERE / "app.py"),
                       f"--server.port={port}",
                       "--browser.gatherUsageStats=false",
                       "--theme.base=light"])
