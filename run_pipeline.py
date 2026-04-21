"""
run_pipeline.py — runs the full pipeline in order:
  1. Generate synthetic demand + event data
  2. Train forecast models and evaluate
  3. Calculate dynamic inventory plan
"""
import subprocess, sys

steps = [
    ("Generating data",         ["python", "data/generate_data.py"]),
    ("Training forecast models", ["python", "src/forecast.py"]),
    ("Inventory optimization",   ["python", "src/inventory.py"]),
]

for label, cmd in steps:
    print(f"\n{'='*50}\n{label}\n{'='*50}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Step failed: {label}")
        sys.exit(result.returncode)

print("\n" + "="*50)
print("Pipeline complete.")
print("Run the dashboard:  streamlit run app/dashboard.py")
print("="*50)
