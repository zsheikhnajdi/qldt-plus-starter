
import os, shutil
from pathlib import Path

def ensure_graphviz_on_path():
    """Ensure Graphviz 'dot' is on PATH (Windows-friendly)."""
    if not shutil.which("dot"):
        os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"
