import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.web import app

if __name__ == "__main__":
    app.run(debug=True, port=5000)
