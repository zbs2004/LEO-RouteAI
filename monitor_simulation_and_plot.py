import os
import time
import subprocess
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_PATH = os.path.join(ROOT, 'training_logs.csv')
PLOT_SCRIPT = os.path.join(ROOT, 'Python', 'plot_training.py')
OUT_DIR = os.path.join(ROOT, 'simulation_results', 'figures')

print('Monitor started. Watching:', LOG_PATH)
last_mtime = None
try:
    while True:
        if os.path.exists(LOG_PATH):
            mtime = os.path.getmtime(LOG_PATH)
            if last_mtime is None or mtime > last_mtime:
                print('Detected update to', LOG_PATH)
                os.makedirs(OUT_DIR, exist_ok=True)
                try:
                    subprocess.run([sys.executable, PLOT_SCRIPT, '--log', LOG_PATH, '--out', OUT_DIR], check=True)
                    print('Updated plots in', OUT_DIR)
                except Exception as ex:
                    print('Plotting failed:', ex)
                last_mtime = mtime
        time.sleep(10)
except KeyboardInterrupt:
    print('Monitor stopped by user')
