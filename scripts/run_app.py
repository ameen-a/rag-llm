import os
import sys
from pathlib import Path

# add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# import the flask app
from app.web import app

if __name__ == "__main__":
    # run the flask app
    app.run(debug=True, port=5000)
