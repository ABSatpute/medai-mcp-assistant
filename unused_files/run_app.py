import subprocess
import sys
import os

# Change to app directory
os.chdir('/mnt/c/Users/ADMIN/Documents/medical_app_langgraph')

# Run the app
print("Starting MedAI Flask App...")
print("Open browser to: http://localhost:5000")
print("Press Ctrl+C to stop")

try:
    subprocess.run([
        'medlgenv/Scripts/python.exe', 
        'app.py'
    ], check=True)
except KeyboardInterrupt:
    print("\nApp stopped by user")
except Exception as e:
    print(f"Error running app: {e}")
