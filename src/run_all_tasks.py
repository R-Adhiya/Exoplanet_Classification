"""
Run all academic rubric extension tasks in order.
Requires packages in requirements.txt PLUS mlflow and fastapi:
  pip install mlflow fastapi uvicorn

Usage (from project root, in your own terminal):
  python src/run_all_tasks.py

Each task is independent -- re-run individually if needed:
  python src/task1_dimensionality_reduction.py
  python src/task2_mlflow_tracking.py
  python src/task3_benchmarking.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

tasks = [
    ("TASK 1: Dimensionality Reduction + Visualizations",
     ROOT / "src" / "task1_dimensionality_reduction.py"),
    ("TASK 2: MLflow Experiment Tracking",
     ROOT / "src" / "task2_mlflow_tracking.py"),
    ("TASK 3: Benchmark Comparison",
     ROOT / "src" / "task3_benchmarking.py"),
]

for title, script in tasks:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"\n[FAIL] {title} exited with code {result.returncode}")
        print("  Fix the error above and re-run this script, or run the task directly.")
        sys.exit(result.returncode)

print("\n" + "="*70)
print("  ALL TASKS COMPLETE")
print("="*70)
print()
print("Next steps:")
print("  mlflow ui --backend-store-uri mlruns   (view experiment tracking)")
print("  uvicorn api:app --reload --port 8000   (start FastAPI inference API)")
print("  docker build -t exoplanet-api .        (build Docker image)")
