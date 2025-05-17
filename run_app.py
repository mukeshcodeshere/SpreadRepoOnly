# run_app.py

from login import login
import subprocess

if login():
    print("Launching Application...")
    subprocess.run(["python", "-m", "streamlit", "run", "streamlit_app.py"])
else:
    print("Login failed. Exiting.")
